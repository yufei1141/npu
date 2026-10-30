"""hw_pack.py : pack an fp32 weight matrix into the GEMV cluster's W4 layout.

The hardware contract (docs/npu_integration/weight_path.md, tensorcore pe.v):

  one beat = 16 K values, 8 column groups, 4 output columns per group
  group n     -> output columns 4n..4n+3
  group n's 256 bits = 4 PEs x 64 bits
  PE p        -> K rows 4p..4p+3, one E8M0 scale for the whole PE
  PE p row i  -> 16 bits = {w_i0, w_i1, w_i2, w_i3}, signed int4, MSB first,
                 carried little-endian in the byte stream

so the quantization granularity is one shared power-of-two exponent per
(K4 block x 4 output columns).  The int4 codes are then chosen per element.

Tile byte layout, group-major (docs/.../weight_path.md section 4):

  weight(g,s) = W_base    + 32*(g*S+s)              low half word first
  scale(g,s)  = scale_base + 16*g*ceil(S/4) + 4*s    padded to 16 B per group

This file is the offline counterpart of the weight_loader: it produces the
exact byte stream a LOAD_W command must move, plus the reconstruction the
hardware will compute, so the simulation error can be budgeted separately
from a real implementation bug.
"""
import numpy as np

# E8M0: stored byte is the biased exponent of 2^0, so 127 means 1.0
E8M0_ONE = 127


def block_scale(block):
    """shared E8M0 exponent for one (K4 x 4) block; codes land in [-8, 7]

    `max/7` is the smallest step that never clips; one or two coarser steps
    can still win on squared error when a block has one large and many small
    values, so the best of three candidates is taken.  Saturation is allowed
    at -8/7, which keeps the search cheap and bounded.
    """
    m = np.max(np.abs(block))
    if m == 0:
        return E8M0_ONE
    e0 = int(np.ceil(np.log2(m / 7.0)))
    best, best_err = None, None
    for e in (e0, e0 + 1, e0 + 2):
        step = np.ldexp(1.0, max(-127, min(127, e)))
        q = np.clip(np.rint(block / step), -8, 7)
        err = float(np.sum((q * step - block) ** 2))
        if best_err is None or err < best_err:
            best, best_err = e, err
    return max(-127, min(127, best)) + E8M0_ONE


def pack_block(block):
    """[4 K rows][4 cols] fp32 -> (bytes 8 per PE row, E8M0 byte)"""
    e8 = block_scale(block)
    step = np.ldexp(1.0, e8 - E8M0_ONE)
    q = np.rint(block / step).astype(np.int32)
    q = np.clip(q, -8, 7)
    q &= 0xF
    out = bytearray()
    for i in range(4):                      # K row i
        w0, w1, w2, w3 = (int(v) for v in q[i])
        # The row is one 16-bit word {w_i0,w_i1,w_i2,w_i3}, MSB (column 0)
        # first, and the weight frame is a byte-addressed stream: the PE reads
        # nibble 3-j of the word for column j, so the word must go out
        # little-endian.  Emitting it big-endian rotates the columns by two
        # (col0 <- w2, col1 <- w3, col2 <- w0, col3 <- w1).
        word = (w0 << 12) | (w1 << 8) | (w2 << 4) | w3
        out += word.to_bytes(2, "little")
    return bytes(out), e8


def pack_tile(W, k_step=16, n_grp=8):
    """W [K, N] fp32 -> (weight bytes, scale bytes, reconstruction)

    The returned bytes are exactly one tile: `weight` is group-major
    (n_grp * S * 32 B), `scale` is the padded 16*ceil(S/4) bytes per group.
    """
    K, N = W.shape
    S = (K + k_step - 1) // k_step
    n_col = 4 * n_grp
    if N > n_col:
        raise ValueError("tile holds %d columns, matrix has %d" % (n_col, N))

    Wp = np.zeros((S * k_step, n_col), dtype=np.float32)
    Wp[:K, :N] = W

    wbytes = bytearray()
    sbytes = bytearray()
    recon = np.zeros((S * k_step, n_col), dtype=np.float32)

    for g in range(n_grp):
        cols = slice(4 * g, 4 * g + 4)
        gs = bytearray()
        for s in range(S):
            for p in range(4):
                rows = slice(s * k_step + 4 * p, s * k_step + 4 * p + 4)
                blk = Wp[rows, cols]
                b, e8 = pack_block(blk)
                wbytes += b                              # PE p: 8 bytes
                gs.append(e8)
                step = np.ldexp(1.0, e8 - E8M0_ONE)
                words = [b[2 * i] | (b[2 * i + 1] << 8) for i in range(4)]
                codes = np.array([[(wd >> 12) & 0xF, (wd >> 8) & 0xF,
                                   (wd >> 4) & 0xF, wd & 0xF]
                                  for wd in words], dtype=np.float32)
                codes = np.where(codes >= 8, codes - 16, codes)
                recon[rows, cols] = codes * step
        # pad the group's scale area to 16 B per four steps
        pad = (-len(gs)) % 16
        sbytes += gs + bytes(pad)

    return bytes(wbytes), bytes(sbytes), recon[:K, :N]


def pack_matrix(W, k_step=16, n_grp=8, tiles=None):
    """split N into tiles of 4*n_grp columns; returns per-tile byte blocks"""
    K, N = W.shape
    n_col = 4 * n_grp
    if tiles is None:
        tiles = (N + n_col - 1) // n_col
    out = []
    for t in range(tiles):
        sub = W[:, t * n_col:(t + 1) * n_col]
        w, s, r = pack_tile(sub, k_step, n_grp)
        out.append({"tile": t, "w": w, "s": s, "recon": r,
                    "col0": t * n_col, "ncol": sub.shape[1]})
    return out


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    W = rng.standard_normal((64, 32)).astype(np.float32) * 0.02
    tiles = pack_matrix(W, n_grp=8, tiles=1)
    t = tiles[0]
    err = np.abs(t["recon"] - W).max()
    scale = np.abs(W).max()
    print("weight bytes", len(t["w"]), "scale bytes", len(t["s"]))
    print("max abs err %.5f  (%.2f%% of max |W|)"
          % (err, 100.0 * err / scale))

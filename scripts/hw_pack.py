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

    `max/7` is the smallest step that never clips, but both coarser steps
    (one large value among small ones) and one *finer* step (a block with a
    single outlier, where clipping it costs less than the step it would
    otherwise force on the other fifteen) win on squared error in practice,
    so the best of four candidates is taken.  Saturation is allowed at
    -8/7, which keeps the search cheap and bounded.
    """
    m = np.max(np.abs(block))
    if m == 0:
        return E8M0_ONE
    e0 = int(np.ceil(np.log2(m / 7.0)))
    best, best_err = None, None
    for e in (e0 - 1, e0, e0 + 1, e0 + 2):
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


def pack_tile_fast(W, k_step=16, n_grp=8):
    """pack_tile without the Python loop over blocks: same bytes, ~500x faster.

    pack_tile calls pack_block once per (K4 x 4) block, which is fine for the
    seven matrices of a layer (a few hundred tiles) and hopeless for the
    lm_head (151936 x 1024 = 4748 tiles, ~1 s each).  This evaluates the same
    four-exponent search for every block of the tile at once.  The blocks are
    laid out as [step, PE, K row, group, column], which is Wp's row order.
    """
    K, N = W.shape
    S = (K + k_step - 1) // k_step
    n_col = 4 * n_grp
    if N > n_col:
        raise ValueError("tile holds %d columns, matrix has %d" % (n_col, N))

    Wp = np.zeros((S * k_step, n_col), dtype=np.float64)
    Wp[:K, :N] = W
    B = Wp.reshape(S, k_step // 4, 4, n_grp, 4)
    m = np.max(np.abs(B), axis=(2, 4))                  # [S, PE, group]
    nz = m > 0
    with np.errstate(divide="ignore"):
        e0 = np.ceil(np.log2(np.where(nz, m, 1.0) / 7.0))
    best_e = np.zeros(m.shape, np.int64)
    best_err = np.full(m.shape, np.inf)
    for d in (-1, 0, 1, 2):
        e = np.clip(e0 + d, -127, 127)
        step = np.ldexp(np.ones_like(e, np.float64), e.astype(np.int32))[:, :, None, :, None]
        q = np.clip(np.rint(B / step), -8, 7)
        err = np.sum((q * step - B) ** 2, axis=(2, 4))
        take = err < best_err
        best_err = np.where(take, err, best_err)
        best_e = np.where(take, e, best_e)
    best_e = np.where(nz, best_e, 0)        # an all-zero block is 2^0 = 127
    step = np.ldexp(np.ones_like(best_e, np.float64), best_e.astype(np.int32))[:, :, None, :, None]
    q = np.clip(np.rint(B / step), -8, 7)               # [-8, 7]
    codes = q.astype(np.int64) & 0xF
    # the word {w0,w1,w2,w3} goes out little-endian, so the low byte carries
    # columns 2 and 3 -- same trick pack_block's comment warns about
    wb = np.stack([(codes[..., 2] << 4) | codes[..., 3],
                   (codes[..., 0] << 4) | codes[..., 1]], axis=-1)
    wbytes = wb.astype(np.uint8).transpose(3, 0, 1, 2, 4).tobytes()
    e8 = (best_e + E8M0_ONE).astype(np.uint8)           # [S, PE, group]
    sbytes = e8.transpose(2, 0, 1).reshape(n_grp, -1)   # [group][step,PE]
    pad = (-sbytes.shape[1]) % 16                       # 16 B per group
    if pad:
        sbytes = np.concatenate([sbytes, np.zeros((n_grp, pad), np.uint8)],
                                axis=1)
    sbytes = sbytes.tobytes()
    recon = (q * step).reshape(S * k_step, n_col)[:K, :N]
    return wbytes, sbytes, recon


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

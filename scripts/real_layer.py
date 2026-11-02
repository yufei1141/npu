"""real_layer.py : one whole Qwen3-0.6B layer through the NPU, end to end.

Every operator of the layer after its input row runs in RTL:

    h1  = RMSNorm(x, ln1)                 VECTOR + REDUCE + SFU
    q   = h1 @ Wq                         MATMUL, int4 weights
    k   = h1 @ Wk                         (one job set per key position)
    v   = h1 @ Wv
    q  <- RMSNorm(q, qn) * hd^-0.5, RoPE  VECTOR + SFU
    k  <- RMSNorm(k, kn), RoPE
    a   = softmax(q K^T) V                activation-B MATMULs, SFU softmax
    o   = a @ Wo                          MATMUL, K = 2048
    xm  = x + o                           VECTOR
    h2  = RMSNorm(xm, ln2)
    g,u = h2 @ Wg, h2 @ Wu
    m   = silu(g) * u
    d   = m @ Wd                          MATMUL, K = 3072
    out = xm + d

The reference is the checkpoint's own fp32 forward; the weights are the real
AWQ weights requantized to the hardware's int4 + E8M0 tiles and the
activations are rounded to fp16, so the comparison is a tolerance check.
Each stage is dumped and compared separately so a mismatch localises.

    python scripts/real_layer.py --layer 0 --upto 7
    python scripts/real_layer.py --layer 0 --token 1 --upto 3
"""
import argparse
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qwen3_ref as qr
import hw_pack
import npu_cmd as nc

# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# external memory map (bytes), relative to the layer's base.  A hidden row
# is 1024 elements: 4 KiB fp32, 2 KiB fp16.  Everything the layer touches
# lives below the weights.  A chained run puts each layer at its own base,
# so two layers can share one image.
# ----------------------------------------------------------------------
_EXT = {
    "X":    0x01000,   # fp32 x_in rows
    "H1":   0x10000,   # fp16 h1 rows
    "QP":   0x14000,   # fp32 q projection of the query token
    "KP":   0x16000,   # fp32 k projection rows
    "VP":   0x1B000,   # fp32 v projection rows
    "QF":   0x20000,   # fp16 q after norm + rope
    "KF":   0x21000,   # fp16 k after norm + rope rows
    "V16":  0x24000,   # fp16 v rows
    "OIN":  0x27000,   # fp16 attention output
    "OP":   0x28000,   # fp32 o projection
    "XM":   0x29000,   # fp32 x_mid
    "OUT":  0x2A000,   # fp32 x_out
    "G1":   0x30000,   # ln1 gamma
    "G2":   0x31000,   # ln2 gamma
    "QN":   0x32000,   # q_norm gamma (hd fp32 = 512 B)
    "KN":   0x32400,   # k_norm gamma
    "COS":  0x33000,   # cos[hd/2] then sin[hd/2], one entry per position
    "H2":   0x34000,   # fp16 h2
    "MA":   0x35000,   # fp16 mlp activation (3072 = 6 KiB)
    "DW":   0x37000,   # fp32 down_proj
    "CL":   0x3F000,   # silu clamp constant
    "W":    0x40000,   # weights
}
LAYER_STRIDE = 0x1000000       # 16 MiB of address space per chained layer


def set_base(base=0):
    """point the layer memory map at `base` (0 for a single-layer run)"""
    g = globals()
    for k, v in _EXT.items():
        g[k + "_EXT"] = base + v


set_base()


# scratchpad logical buffers 0..7 (one 8 KiB bank window each)
B_X, B_H, B_G, B_A, B_D, B_T, B_S0, B_S1 = 0, 1, 2, 3, 4, 5, 6, 7

F32, F16 = nc.F_FP32, nc.F_FP16

CHUNK_TILES, NGATE = 2, 8             # 64 output columns per MATMUL
CHUNK_COLS = 4 * NGATE * CHUNK_TILES
D_TILES, NGD = 4, 4                   # down_proj: K = 3072 shrinks a tile
DOWN_COLS = 4 * NGD * D_TILES
SILU_CLAMP = -100.0
LOG2E = 1.4426950408889634

# gate workspace in B_G, temps in B_T (see scripts/real_mlp.py)
G_GATE, G_UP, G_E, G_R = 0, 512, 1024, 1536
T_T, T_P1, T_S = 0, 512, 1024
CLAMP_OFF = 32


def f32(x):
    return int(np.float32(x).view(np.uint32))


class Script:
    """command stream with job ids, a reusable slot pool and window release"""

    def __init__(self, job0=100):
        self.lines = []
        self.map = []
        self.job = job0
        self.slot = -1
        self.win = []

    def njob(self):
        # The command word carries job ids in 16 bits ([47:32], and the same
        # width for dep_job and a RELEASE target in the RTL), while one
        # position of the full network is 108k commands.  Wrap the allocator
        # here so *every* consumer -- CMD, dep_job, release target and the
        # harness WAIT -- sees the same truncated value; CLEAR after each
        # window keeps the harness from matching a completion from the
        # previous pass through the id space.
        self.job = (self.job + 1) & 0xFFFF
        return self.job

    def nslot(self):
        self.slot += 1
        if self.slot >= 16:
            raise RuntimeError("window needs more than 16 event slots")
        return self.slot

    def put(self, c, j, s):
        self.lines.append("CMD " + nc.hex128(c))
        self.win.append((j, s))
        # job id -> (opcode, dep job, dep slot) for post-mortem of a CPL
        self.map.append("%d op=%02x dep=%d/%d" %
                        (j, (c >> 8) & 0xFF, (c >> 48) & 0xFFFF,
                         (c >> 104) & 0xFF))

    def add(self, build):
        j, s = self.njob(), self.nslot()
        self.put(build(j, s), j, s)
        return (j, s)

    def close(self):
        if not self.win:
            return
        self.lines.append("WAIT %d" % self.win[-1][0])
        # job ids are 16 bit, and a full-network position is far past 65536
        # commands, so the harness has to forget this window's completions
        # before the ids come round again
        self.lines.append("CLEAR")
        for j, s in self.win:
            self.lines.append("CMD " + nc.hex128(nc.release(self.njob(), s, j)))
        self.win = []
        self.slot = -1


def dep(prev, flags=nc.FLAG_DEP):
    return dict(flags=flags, dep_job=prev[0], dep_slot=prev[1])


class WMat:
    """a weight matrix packed into per-chunk tiles in the memory image"""

    def __init__(self, name, W, base, img, ngrp=NGATE, ntiles=CHUNK_TILES,
                 recon=None):
        self.name = name
        self.K, self.N = W.shape
        self.ngrp = ngrp
        self.ntiles = ntiles
        self.cols = 4 * ngrp * ntiles
        self.steps = (self.K + 15) // 16
        self.nchunk = (self.N + self.cols - 1) // self.cols
        self.tiles = []
        if img is None:                 # cache hit: the layout is arithmetic
            wb = ngrp * self.steps * 32
            tb = wb + ngrp * ((self.steps + 3) // 4) * 16
            for c in range(self.nchunk):
                self.tiles.append(
                    [(base + (c * ntiles + t) * tb,
                      base + (c * ntiles + t) * tb + wb, tb)
                     for t in range(ntiles)])
            self.end = base + self.nchunk * ntiles * tb
            self.recon = recon
            return
        self.recon = np.zeros(W.shape, np.float32)
        off = base
        for c in range(self.nchunk):
            ch = []
            for t in range(ntiles):
                col0 = c * self.cols + t * 4 * ngrp
                w, s, r = hw_pack.pack_tile_fast(
                    W[:, col0:col0 + 4 * ngrp].astype(np.float32), n_grp=ngrp)
                img[off:off + len(w)] = w
                img[off + len(w):off + len(w) + len(s)] = s
                ch.append((off, off + len(w), len(w) + len(s)))
                self.recon[:self.K, col0:col0 + 4 * ngrp] = r[:self.K]
                off += len(w) + len(s)
            self.tiles.append(ch)
        self.end = off

    def emit(self, S, c, a_buf, d_buf, d_ext):
        """one chunk: stage its tiles, run the MATMUL, drain the columns"""
        mm = S.njob()
        sl = S.nslot()
        for t, (wa, sa, nb) in enumerate(self.tiles[c]):
            S.add(lambda j, s, t=t, wa=wa, sa=sa, nb=nb: nc.load_w(
                j, wa, sa, mm, t, 0, self.ngrp, self.steps, nb, slot=s))
        S.put(nc.matmul(mm, 1, self.ngrp, 0, self.ntiles, self.steps,
                        self.K, self.cols, 0, 2 * self.K, 0, 4 * self.cols,
                        a_buf, d_buf, sl), mm, sl)
        S.add(lambda j, s: nc.store(j, d_ext + 4 * self.cols * c, d_buf, 0,
                                    4 * self.cols, slot=s, **dep((mm, sl))))


def metrics(got, ref):
    # ravel first: rows of a frame are compared as one vector
    got = np.asarray(got, np.float64).reshape(-1)
    ref = np.asarray(ref, np.float64).reshape(-1)
    e = got - ref
    rr = float(np.sqrt(np.mean(e ** 2)))
    n = float(np.sqrt(np.mean(ref ** 2))) + 1e-30
    c = float(got @ ref
              / (np.linalg.norm(got) * np.linalg.norm(ref) + 1e-30))
    return rr, n, c


def stat(name, got, ref):
    rr, n, c = metrics(np.asarray(got, np.float64), np.asarray(ref, np.float64))
    print("  %-6s RMS err %.4e (%5.1f%% of RMS)  cos %.6f" %
          (name, rr, 100.0 * rr / n, c))
    return c


def norm_block(S, x_ext, gam_ext, out_ext, K):
    """RMSNorm(x, gam) -> fp16 row at out_ext, using B_X/B_G/B_T/B_H."""
    S.add(lambda j, s: nc.load(j, x_ext, B_X, row_bytes=4 * K, slot=s))
    S.add(lambda j, s: nc.load(j, gam_ext, B_G, row_bytes=4 * K, slot=s))
    f0 = S.add(lambda j, s: nc.fence(j, s))
    sq = S.add(lambda j, s: nc.vector(j, nc.V_MUL, K, B_X, 0, B_T, 0, s,
                                      s1=B_X, o1=0, **dep(f0)))
    sm = S.add(lambda j, s: nc.reduce(j, nc.R_SUM, K, B_T, 0, B_S0, 0, s,
                                      **dep(sq)))
    ms = S.add(lambda j, s: nc.vector(j, nc.V_MUL_SCALAR, 1, B_S0, 0, B_S1, 0,
                                      s, scalar=f32(1.0 / K), **dep(sm)))
    ve = S.add(lambda j, s: nc.vector(j, nc.V_ADD_SCALAR, 1, B_S1, 0, B_S0, 0,
                                      s, scalar=f32(1e-6), **dep(ms)))
    rs = S.add(lambda j, s: nc.sfu(j, nc.S_RSQRT, 1, B_S0, 0, B_S1, 0, s,
                                   **dep(ve)))
    xr = S.add(lambda j, s: nc.vector(j, nc.V_MUL | nc.V_BCAST_S, K,
                                      B_X, 0, B_T, 0, s, s1=B_S1, o1=0,
                                      **dep(rs)))
    out = S.add(lambda j, s: nc.vector(j, nc.V_MUL, K, B_T, 0, B_H, 0, s,
                                       s1=B_G, o1=0, fd=F16, **dep(xr)))
    S.add(lambda j, s: nc.store(j, out_ext, B_H, 0, 2 * K, slot=s, **dep(out)))


def qk_norm_rope(S, x_ext, dst_ext, idx, hd, sc):
    """RMSNorm over one head, optional scale, then rotate-half RoPE.

    The caller stages  B_G gamma, B_H cos|sin (fp32, hd/2 each);  this uses
    B_X x, B_A y, B_T temps, B_D fp16 out, B_S0 the sum, B_S1 the rsqrt.
    """
    hs = hd // 2
    S.add(lambda j, s: nc.load(j, x_ext + 4 * hd * idx, B_X,
                               row_bytes=4 * hd, slot=s))
    f0 = S.add(lambda j, s: nc.fence(j, s))
    sq = S.add(lambda j, s: nc.vector(j, nc.V_MUL, hd, B_X, 0, B_T, 0, s,
                                      s1=B_X, o1=0, **dep(f0)))
    sm = S.add(lambda j, s: nc.reduce(j, nc.R_SUM, hd, B_T, 0, B_S0, 0, s,
                                      **dep(sq)))
    ms = S.add(lambda j, s: nc.vector(j, nc.V_MUL_SCALAR, 1, B_S0, 0, B_S1, 0,
                                      s, scalar=f32(1.0 / hd), **dep(sm)))
    ve = S.add(lambda j, s: nc.vector(j, nc.V_ADD_SCALAR, 1, B_S1, 0, B_S0, 0,
                                      s, scalar=f32(1e-6), **dep(ms)))
    rs = S.add(lambda j, s: nc.sfu(j, nc.S_RSQRT, 1, B_S0, 0, B_S1, 0, s,
                                   **dep(ve)))
    S.close()

    # the previous window was WAITed and released, so no dep is possible (the
    # release drops the (job, slot) record); the barrier already ordered them
    xg = S.add(lambda j, s: nc.vector(j, nc.V_MUL, hd, B_X, 0, B_A, 0, s,
                                      s1=B_G, o1=0))
    yv = S.add(lambda j, s: nc.vector(j, nc.V_MUL | nc.V_BCAST_S, hd,
                                      B_A, 0, B_A, 0, s, s1=B_S1, o1=0,
                                      flags=nc.FLAG_INPLACE | nc.FLAG_DEP,
                                      dep_job=xg[0], dep_slot=xg[1]))
    if sc is not None:
        yv = S.add(lambda j, s: nc.vector(j, nc.V_MUL_SCALAR, hd, B_A, 0,
                                          B_A, 0, s, scalar=f32(sc),
                                          flags=nc.FLAG_INPLACE | nc.FLAG_DEP,
                                          dep_job=yv[0], dep_slot=yv[1]))
    t1 = S.add(lambda j, s: nc.vector(j, nc.V_MUL, hs, B_A, 0, B_T, 0, s,
                                      s1=B_H, o1=0, **dep(yv)))
    t2 = S.add(lambda j, s: nc.vector(j, nc.V_MUL, hs, B_A, 4 * hs,
                                      B_T, 4 * hs, s, s1=B_H, o1=4 * hs,
                                      **dep(t1)))
    o1 = S.add(lambda j, s: nc.vector(j, nc.V_SUB, hs, B_T, 0, B_D, 0, s,
                                      s1=B_T, o1=4 * hs, fd=F16, **dep(t2)))
    t3 = S.add(lambda j, s: nc.vector(j, nc.V_MUL, hs, B_A, 4 * hs, B_T, 0,
                                      s, s1=B_H, o1=0, **dep(o1)))
    t4 = S.add(lambda j, s: nc.vector(j, nc.V_MUL, hs, B_A, 0, B_T, 4 * hs,
                                      s, s1=B_H, o1=4 * hs, **dep(t3)))
    o2 = S.add(lambda j, s: nc.vector(j, nc.V_ADD, hs, B_T, 0, B_D, 2 * hs,
                                      s, s1=B_T, o1=4 * hs, fd=F16,
                                      **dep(t4)))
    S.add(lambda j, s: nc.store(j, dst_ext + 2 * hd * idx, B_D, 0, 2 * hd,
                                slot=s, **dep(o2)))
    S.close()


def attention_head(S, h, hd, H, nk, kh):
    """one head: q K^T -> softmax -> P V, fp16 output at OIN_EXT."""
    ngrp_k = (nk + 3) // 4
    ksteps = hd // 4
    ngrp_v = 8
    nblk = (hd + 4 * ngrp_v - 1) // (4 * ngrp_v)
    vsteps = (nk + 3) // 4

    bQ, bK, bV, bKT, bKF, bSC, bP = 0, 1, 2, 3, 4, 5, 6
    bEX, bSS, bSS2 = bQ, bKT, bKF      # q, K^T and the K frame die in turn

    S.add(lambda j, s: nc.load(j, QF_EXT + 2 * hd * h, bQ,
                               row_bytes=2 * hd, slot=s))
    S.add(lambda j, s: nc.load(j, KF_EXT + 2 * hd * kh, bK,
                               row_bytes=2 * hd, rows=nk,
                               ext_stride=2 * H, local_stride=2 * hd, slot=s))
    S.add(lambda j, s: nc.load(j, V16_EXT + 2 * hd * kh, bV,
                               row_bytes=2 * hd, rows=nk,
                               ext_stride=2 * H, local_stride=2 * hd, slot=s))
    f0 = S.add(lambda j, s: nc.fence(j, s))
    tr = S.add(lambda j, s: nc.transpose2d(j, nk, hd, bK, 0, bKT, 0,
                                           2 * hd, 2 * nk, s, **dep(f0)))
    pk = S.add(lambda j, s: nc.pack_w16(j, hd, nk, bKT, 0, bKF, 0, 2 * nk,
                                        ngrp_k, s, **dep(tr)))
    S.add(lambda j, s: nc.matmul(j, 1, ngrp_k, 1, 1, ksteps, hd, nk,
                                 0, 2 * hd, 0, 4 * nk, bQ, bSC, s,
                                 a_fmt=F16, b_buf=bKF, b_base=0,
                                 b_stride=32 * ksteps, b_fmt=F16,
                                 **dep(pk)))
    S.close()

    mx = S.add(lambda j, s: nc.reduce(j, nc.R_MAX, nk, bSC, 0, bSS, 0, s))
    sb = S.add(lambda j, s: nc.vector(j, nc.V_SUB | nc.V_BCAST_S, nk,
                                      bSC, 0, bSC, 0, s, s1=bSS, o1=0,
                                      flags=nc.FLAG_INPLACE | nc.FLAG_DEP,
                                      dep_job=mx[0], dep_slot=mx[1]))
    ml = S.add(lambda j, s: nc.vector(j, nc.V_MUL_SCALAR, nk, bSC, 0, bSC, 0,
                                      s, scalar=f32(LOG2E),
                                      flags=nc.FLAG_INPLACE | nc.FLAG_DEP,
                                      dep_job=sb[0], dep_slot=sb[1]))
    ex = S.add(lambda j, s: nc.sfu(j, nc.S_EXP2, nk, bSC, 0, bEX, 0, s,
                                   **dep(ml)))
    sm = S.add(lambda j, s: nc.reduce(j, nc.R_SUM, nk, bEX, 0, bSS, 64, s,
                                      **dep(ex)))
    rc = S.add(lambda j, s: nc.sfu(j, nc.S_RCP, 1, bSS, 64, bSS2, 0, s,
                                   **dep(sm)))
    S.add(lambda j, s: nc.vector(j, nc.V_MUL | nc.V_BCAST_S, nk, bEX, 0, bP, 0,
                                 s, s1=bSS2, o1=0, fd=F16, **dep(rc)))
    S.close()

    # src1_buf != 0 is what marks the activation-B path, so the frame cannot
    # live in buffer 0; bK is free once the transpose has copied K
    bVF = bK
    prev = None
    for i in range(nblk):
        prev = S.add(lambda j, s, i=i, p=prev: nc.pack_w16(
            j, nk, 4 * ngrp_v, bV, 4 * ngrp_v * 2 * i, bVF,
            ngrp_v * vsteps * 32 * i, 2 * hd, ngrp_v, s,
            **(dep(p) if p else {})))
    f1 = S.add(lambda j, s: nc.fence(j, s))
    mm = S.njob()
    sl = S.nslot()
    S.put(nc.matmul(mm, 1, ngrp_v, 1, nblk, vsteps, nk, hd, 0, 32, 0,
                    4 * hd, bP, bKT, sl, a_fmt=F16, b_buf=bVF, b_base=0,
                    b_stride=32 * vsteps, b_fmt=F16, **dep(f1)), mm, sl)
    ov = S.add(lambda j, s: nc.vector(j, nc.V_CONVERT, hd, bKT, 0, bKF, 0, s,
                                      fs0=F32, fd=F16, **dep((mm, sl))))
    S.add(lambda j, s: nc.store(j, OIN_EXT + 2 * hd * h, bKF, 0, 2 * hd,
                                slot=s, **dep(ov)))
    S.close()



def fill_image(img, L, cos, sin, nk, hd, HX, x_rows):
    """gammas, per-position RoPE angles, the silu clamp and, unless the layer
    is fed by another one, the x_in rows"""
    img[G1_EXT:G1_EXT + 4 * HX] = L["ln1"].astype(np.float32).tobytes()
    img[G2_EXT:G2_EXT + 4 * HX] = L["ln2"].astype(np.float32).tobytes()
    img[QN_EXT:QN_EXT + 4 * hd] = L["qn"].astype(np.float32).tobytes()
    img[KN_EXT:KN_EXT + 4 * hd] = L["kn"].astype(np.float32).tobytes()
    for t in range(nk):          # RoPE angles are per position, not per token
        cs = np.concatenate([cos[t][:hd // 2], sin[t][:hd // 2]])
        img[COS_EXT + 4 * hd * t:COS_EXT + 4 * hd * (t + 1)] = \
            cs.astype(np.float32).tobytes()
    img[CL_EXT:CL_EXT + 4] = np.float32(SILU_CLAMP).tobytes()
    if x_rows is not None:
        for t in range(nk):
            img[X_EXT + 4 * HX * t:X_EXT + 4 * HX * (t + 1)] = \
                x_rows[t].astype(np.float32).tobytes()


def stage_weights(img, li, L, outdir, repack):
    """pack this layer's seven matrices at the current W_EXT, cache to npz"""
    spec = [("wq", NGATE, CHUNK_TILES), ("wk", NGATE, CHUNK_TILES),
            ("wv", NGATE, CHUNK_TILES), ("wo", NGATE, CHUNK_TILES),
            ("wg", NGATE, CHUNK_TILES), ("wu", NGATE, CHUNK_TILES),
            ("wd", NGD, D_TILES)]
    cache = os.path.join(outdir, "pack_L%d.npz" % li)
    mats = {}
    if os.path.exists(cache) and not repack:
        z = np.load(cache)
        blob = z["blob"]
        img[W_EXT:W_EXT + len(blob)] = blob.tobytes()
        base = W_EXT
        for name, ngrp, ntiles in spec:
            mats[name] = WMat(name, L[name], base, None, ngrp, ntiles,
                              recon=z["r_" + name])
            base = mats[name].end
        print("layer %d weights from cache: %.2f MB" % (li, len(blob) / 1e6))
    else:
        base = W_EXT
        for name, ngrp, ntiles in spec:
            mats[name] = WMat(name, L[name], base, img, ngrp, ntiles)
            base = mats[name].end
        np.savez(cache,
                 blob=np.frombuffer(bytes(img[W_EXT:base]), np.uint8),
                 **{"r_" + k: v.recon for k, v in mats.items()})
        print("layer %d weights packed: %.2f MB" % (li, (base - W_EXT) / 1e6))
    return mats


def emit_layer(S, mats, up, nk, tok, hd, nh, nkv, HX, HQ, HK, rep, scale,
               t0=0):
    """stages 1..7 of one layer, through the memory map at the current base.

    Positions t0..nk-1 are projected and rotated afresh; attention still spans
    every position 0..nk-1.  t0 = 0 is the whole prompt, t0 = tok is one
    decode step reading the k/v cache the earlier steps left behind.
    """
    Wq, Wk, Wv, Wo, Wg, Wu, Wd = (mats[k] for k in
                                  ("wq", "wk", "wv", "wo", "wg", "wu", "wd"))

    # ---- 1. h1 for every freshly projected position --------------------
    if up >= 1:
        for t in range(t0, nk):
            norm_block(S, X_EXT + 4 * HX * t, G1_EXT, H1_EXT + 2 * HX * t, HX)
            S.close()

    # ---- 2. q, k, v ----------------------------------------------------
    if up >= 2:
        S.add(lambda j, s: nc.load(j, H1_EXT + 2 * HX * tok, B_H,
                                   row_bytes=2 * HX, slot=s))
        S.close()
        for c in range(Wq.nchunk):
            Wq.emit(S, c, B_H, B_D, QP_EXT)
            S.close()
        for t in range(t0, nk):
            S.add(lambda j, s, t=t: nc.load(j, H1_EXT + 2 * HX * t, B_H,
                                            row_bytes=2 * HX, slot=s))
            S.close()
            for c in range(Wk.nchunk):
                Wk.emit(S, c, B_H, B_D, KP_EXT + 4 * HK * t)
                S.close()
            for c in range(Wv.nchunk):
                Wv.emit(S, c, B_H, B_D, VP_EXT + 4 * HK * t)
                S.close()

    # ---- 3. q_norm/k_norm + RoPE --------------------------------------
    if up >= 3:
        S.add(lambda j, s: nc.load(j, COS_EXT + 4 * hd * tok, B_H,
                                   row_bytes=4 * hd, slot=s))
        S.close()
        S.add(lambda j, s: nc.load(j, QN_EXT, B_G, row_bytes=4 * hd, slot=s))
        S.close()
        for h in range(nh):
            qk_norm_rope(S, QP_EXT, QF_EXT, h, hd, scale)
        # every key position is normalised and rotated at its own angle; the
        # window above has overwritten B_H with the q angles by then, so each
        # position restages its cos|sin first
        S.add(lambda j, s: nc.load(j, KN_EXT, B_G, row_bytes=4 * hd, slot=s))
        S.close()
        for t in range(t0, nk):
            S.add(lambda j, s, t=t: nc.load(j, COS_EXT + 4 * hd * t, B_H,
                                            row_bytes=4 * hd, slot=s))
            S.close()
            for h in range(nkv):
                qk_norm_rope(S, KP_EXT + 4 * HK * t, KF_EXT + 2 * HK * t, h,
                             hd, None)
        # v only needs rounding to fp16, which attention consumes
        for t in range(t0, nk):
            S.add(lambda j, s, t=t: nc.load(j, VP_EXT + 4 * HK * t, B_X,
                                            row_bytes=4 * HK, slot=s))
            cv = S.add(lambda j, s, t=t: nc.vector(j, nc.V_CONVERT, HK, B_X, 0,
                                                   B_H, 0, s, fd=F16))
            S.add(lambda j, s, t=t: nc.store(j, V16_EXT + 2 * HK * t, B_H, 0,
                                             2 * HK, slot=s, **dep(cv)))
            S.close()

    # ---- 4. attention ---------------------------------------------------
    if up >= 4:
        for h in range(nh):
            attention_head(S, h, hd, HK, nk, h // rep)

    # ---- 5. o_proj and the first residual --------------------------------
    if up >= 5:
        S.add(lambda j, s: nc.load(j, OIN_EXT, B_H, row_bytes=2 * HQ, slot=s))
        S.close()
        for c in range(Wo.nchunk):
            Wo.emit(S, c, B_H, B_D, OP_EXT)
            S.close()
        S.add(lambda j, s: nc.load(j, X_EXT + 4 * HX * tok, B_X,
                                   row_bytes=4 * HX, slot=s))
        S.add(lambda j, s: nc.load(j, OP_EXT, B_D, row_bytes=4 * HX, slot=s))
        ad = S.add(lambda j, s: nc.vector(j, nc.V_ADD, HX, B_X, 0, B_A, 0, s,
                                          s1=B_D, o1=0))
        S.add(lambda j, s: nc.store(j, XM_EXT, B_A, 0, 4 * HX, slot=s,
                                    **dep(ad)))
        S.close()

    # ---- 6. the MLP block ------------------------------------------------
    if up >= 6:
        norm_block(S, XM_EXT, G2_EXT, H2_EXT, HX)
        S.close()
        S.add(lambda j, s: nc.load(j, CL_EXT, B_S1, off=CLAMP_OFF,
                                   row_bytes=4, slot=s))
        S.close()
        for c in range(Wg.nchunk):
            mm_g = S.njob()
            sl_g = S.nslot()
            for t, (wa, sa, nb) in enumerate(Wg.tiles[c]):
                S.add(lambda j, s, t=t, wa=wa, sa=sa, nb=nb: nc.load_w(
                    j, wa, sa, mm_g, t, 0, NGATE, Wg.steps, nb, slot=s))
            S.put(nc.matmul(mm_g, 1, NGATE, 0, CHUNK_TILES, Wg.steps, HX,
                            CHUNK_COLS, 0, 2 * HX, G_GATE, 4 * CHUNK_COLS,
                            B_H, B_G, sl_g), mm_g, sl_g)

            mm_u = S.njob()
            sl_u = S.nslot()
            for t, (wa, sa, nb) in enumerate(Wu.tiles[c]):
                S.add(lambda j, s, t=t, wa=wa, sa=sa, nb=nb: nc.load_w(
                    j, wa, sa, mm_u, t, 0, NGATE, Wu.steps, nb, slot=s))
            S.put(nc.matmul(mm_u, 1, NGATE, 0, CHUNK_TILES, Wu.steps, HX,
                            CHUNK_COLS, 0, 2 * HX, G_UP, 4 * CHUNK_COLS,
                            B_H, B_G, sl_u), mm_u, sl_u)

            fn = S.add(lambda j, s: nc.fence(j, s))
            v1 = S.add(lambda j, s: nc.vector(j, nc.V_MUL_SCALAR, CHUNK_COLS,
                                              B_G, G_GATE, B_T, T_T, s,
                                              scalar=f32(LOG2E), **dep(fn)))
            vc = S.add(lambda j, s: nc.vector(j, nc.V_MAX | nc.V_BCAST_S,
                                              CHUNK_COLS, B_T, T_T, B_T, T_T,
                                              s, s1=B_S1, o1=CLAMP_OFF,
                                              flags=nc.FLAG_INPLACE |
                                              nc.FLAG_DEP,
                                              dep_job=v1[0], dep_slot=v1[1]))
            vn = S.add(lambda j, s: nc.vector(j, nc.V_MUL_SCALAR, CHUNK_COLS,
                                              B_T, T_T, B_T, T_T, s,
                                              scalar=f32(-1.0),
                                              flags=nc.FLAG_INPLACE |
                                              nc.FLAG_DEP,
                                              dep_job=vc[0], dep_slot=vc[1]))
            v2 = S.add(lambda j, s: nc.sfu(j, nc.S_EXP2, CHUNK_COLS,
                                           B_T, T_T, B_G, G_E, s, **dep(vn)))
            v3 = S.add(lambda j, s: nc.vector(j, nc.V_ADD_SCALAR, CHUNK_COLS,
                                              B_G, G_E, B_T, T_P1, s,
                                              scalar=f32(1.0), **dep(v2)))
            v4 = S.add(lambda j, s: nc.sfu(j, nc.S_RCP, CHUNK_COLS, B_T, T_P1,
                                           B_G, G_R, s, **dep(v3)))
            v5 = S.add(lambda j, s: nc.vector(j, nc.V_MUL, CHUNK_COLS,
                                              B_G, G_GATE, B_T, T_S, s,
                                              s1=B_G, o1=G_R, **dep(v4)))
            S.add(lambda j, s: nc.vector(j, nc.V_MUL, CHUNK_COLS, B_T, T_S,
                                         B_A, c * CHUNK_COLS * 2, s,
                                         s1=B_G, o1=G_UP, fd=F16, **dep(v5)))
            S.close()

        # the activation stays in B_A for down_proj, but dump it to memory too
        S.add(lambda j, s: nc.store(j, MA_EXT, B_A, 0, 2 * Wu.N, slot=s))
        S.close()

        for c in range(Wd.nchunk):
            mm = S.njob()
            sl = S.nslot()
            for t, (wa, sa, nb) in enumerate(Wd.tiles[c]):
                S.add(lambda j, s, t=t, wa=wa, sa=sa, nb=nb: nc.load_w(
                    j, wa, sa, mm, t, 0, NGD, Wd.steps, nb, slot=s))
                if t == 1:
                    S.put(nc.matmul(mm, 1, NGD, 0, D_TILES, Wd.steps, Wd.K,
                                    DOWN_COLS, 0, 2 * Wd.K, 0, 4 * DOWN_COLS,
                                    B_A, B_D, sl), mm, sl)
            S.add(lambda j, s, c=c: nc.store(j, DW_EXT + c * DOWN_COLS * 4,
                                             B_D, 0, 4 * DOWN_COLS, slot=s,
                                             **dep((mm, sl))))
            S.close()

    # ---- 7. the second residual -----------------------------------------
    if up >= 7:
        S.add(lambda j, s: nc.load(j, XM_EXT, B_X, row_bytes=4 * HX, slot=s))
        S.add(lambda j, s: nc.load(j, DW_EXT, B_D, row_bytes=4 * HX, slot=s))
        ad = S.add(lambda j, s: nc.vector(j, nc.V_ADD, HX, B_X, 0, B_A, 0, s,
                                          s1=B_D, o1=0))
        S.add(lambda j, s: nc.store(j, OUT_EXT, B_A, 0, 4 * HX, slot=s,
                                    **dep(ad)))
        S.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=qr.DEFAULT_MODEL)
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--token", type=int, default=-1)
    ap.add_argument("--upto", type=int, default=7,
                    help="1 norms, 2 projections, 3 qk-norm+rope, 4 attention,"
                         " 5 o_proj+residual, 6 mlp, 7 residual")
    ap.add_argument("--chain", type=int, default=1,
                    help="run N consecutive layers back to back in one image;"
                         " each takes the previous layer's x_out as its x_in")
    ap.add_argument("--bin", default="build/vsim/qk/Vnpu_top.exe")
    ap.add_argument("--outdir", default="build/real_layer")
    ap.add_argument("--repack", action="store_true",
                    help="ignore the packed-weight cache")
    ap.add_argument("--analyze", action="store_true",
                    help="skip the simulation and read the dumps already in "
                         "--outdir")
    ap.add_argument("--build-only", action="store_true",
                    help="write mem.bin/cmds.txt and stop before simulating")
    args = ap.parse_args()
    up = args.upto

    m = qr.Qwen3(args.model)
    ids = m.tok.encode(args.prompt)
    T = len(ids)
    tok = args.token if args.token >= 0 else T - 1
    nk = tok + 1
    hd, nh, nkv = m.hd, m.nh, m.nkv
    rep = nh // nkv
    scale = hd ** -0.5
    cos, sin = m.cos_sin(T)
    layers = [args.layer + i for i in range(args.chain)]

    ds = {}
    for li in layers:
        _, ds[li] = m.forward(ids, dump=li)
    x_all = ds[layers[0]]["x_in"].astype(np.float32)
    HX = x_all.shape[1]          # residual width (1024)
    HQ = nh * hd                 # q / attention output width (2048)
    HK = nkv * hd                # k, v width (1024)

    print("layer %s token %d: %d keys, %d heads x %d, upto %d" %
          ("+".join(str(x) for x in layers), tok, nk, nh, hd, up))

    os.makedirs(args.outdir, exist_ok=True)
    img = bytearray(1 << (24 + max(0, len(layers) - 1)))
    S = Script()
    paths_by, mats_by = {}, {}
    for i, li in enumerate(layers):
        set_base(i * LAYER_STRIDE)
        L = m.L[li]
        mats = stage_weights(img, li, L, args.outdir, args.repack)
        mats_by[li] = mats
        # Layer 0 is fed by the embedding, taken from the model.  A chained
        # layer needs x_in at every key position (its k/v projections read all
        # of them) but the NPU only wrote the query row: the other rows are
        # the network's own layer-(i-1) output, and the query row is
        # overwritten below with the bytes the previous layer really wrote.
        prev = ds[layers[i - 1]]["x"] if i else x_all
        fill_image(img, L, cos, sin, nk, hd, HX, prev[:nk])
        if i:
            src = (i - 1) * LAYER_STRIDE + _EXT["OUT"]
            dst = i * LAYER_STRIDE + _EXT["X"] + 4 * HX * tok
            ld = S.add(lambda j, s: nc.load(j, src, B_X, row_bytes=4 * HX,
                                            slot=s))
            S.add(lambda j, s: nc.store(j, dst, B_X, 0, 4 * HX, slot=s,
                                        **dep(ld)))
            S.close()
        emit_layer(S, mats, up, nk, tok, hd, nh, nkv, HX, HQ, HK, rep, scale)

        sfx = "" if len(layers) == 1 else "_L%d" % li
        dumps = [
            (H1_EXT, 2 * HX * nk, "h1.bin"),
            (QP_EXT, 4 * HQ, "q.bin"),
            (KP_EXT, 4 * HK * nk, "k.bin"),
            (VP_EXT, 4 * HK * nk, "v.bin"),
            (QF_EXT, 2 * HQ, "qf.bin"),
            (KF_EXT, 2 * HK * nk, "kf.bin"),
            (V16_EXT, 2 * HK * nk, "v16.bin"),
            (OIN_EXT, 2 * HQ, "oin.bin"),
            (OP_EXT, 4 * HX, "o.bin"),
            (XM_EXT, 4 * HX, "xm.bin"),
            (H2_EXT, 2 * HX, "h2.bin"),
            (MA_EXT, 2 * mats["wu"].N, "a.bin"),
            (DW_EXT, 4 * HX, "down.bin"),
            (OUT_EXT, 4 * HX, "x.bin"),
        ]
        paths = paths_by[li] = {}
        for ext, n, name in dumps:
            p = os.path.join(args.outdir, name.replace(".bin", sfx + ".bin"))
            paths[name] = p
            S.lines.append("DUMP %x %x %s" % (ext, n, p))

    mem_bin = os.path.join(args.outdir, "mem.bin")
    if args.analyze:
        print("analyzing the dumps already in %s" % args.outdir)
    else:
        open(mem_bin, "wb").write(bytes(img))
        script = os.path.join(args.outdir, "cmds.txt")
        open(script, "w").write("\n".join(S.lines) + "\n")
        open(os.path.join(args.outdir, "jobs.txt"), "w").write(
            "\n".join(S.map) + "\n")
        print("%d commands, %d lines" %
              (sum(1 for l in S.lines if l.startswith("CMD")), len(S.lines)))
        if args.build_only:
            return

        r = subprocess.run([os.path.abspath(args.bin), "--mem",
                            os.path.abspath(mem_bin), "--script",
                            os.path.abspath(script)], capture_output=True,
                           text=True)
        if r.returncode != 0:
            print(r.stdout[-4000:])
            print(r.stderr.strip())
            sys.exit("harness failed rc=%d" % r.returncode)
        out = r.stdout
        cpl = [l for l in out.splitlines() if l.startswith("CPL ")]
        errs = [l for l in cpl if l.split()[4] != "0000"]
        done = [l for l in out.splitlines() if l.startswith("DONE")]
        print(done[0] if done else out.strip().splitlines()[-1])
        if errs:
            print("%d of %d completions not OK, first 10:" %
                  (len(errs), len(cpl)))
            for l in errs[:10]:
                print("  " + l)
            sys.exit(1)

    if len(layers) > 1:
        chain_analysis(m, ids, tok, layers, ds, paths_by, mats_by, HX)
        return

    d, L = ds[layers[0]], m.L[layers[0]]
    mats, paths = mats_by[layers[0]], paths_by[layers[0]]
    Wq, Wk, Wv, Wo, Wg, Wu, Wd = (mats[k] for k in
                                  ("wq", "wk", "wv", "wo", "wg", "wu", "wd"))

    def rd(name, dt, n):
        return np.frombuffer(open(paths[name], "rb").read(), dt)[:n]

    h1 = rd("h1.bin", np.float16, HX * nk).astype(np.float32).reshape(nk, HX)
    qp = rd("q.bin", np.float32, HQ)
    kp = rd("k.bin", np.float32, HK * nk).reshape(nk, HK)
    vp = rd("v.bin", np.float32, HK * nk).reshape(nk, HK)
    qf = rd("qf.bin", np.float16, HQ).astype(np.float32)
    kf = rd("kf.bin", np.float16, HK * nk).astype(np.float32).reshape(nk, HK)
    v16 = rd("v16.bin", np.float16, HK * nk).astype(np.float32
                                                   ).reshape(nk, HK)
    oin = rd("oin.bin", np.float16, HQ).astype(np.float32)
    op = rd("o.bin", np.float32, HX)
    xm = rd("xm.bin", np.float32, HX)
    h2 = rd("h2.bin", np.float16, HX).astype(np.float32)
    ma = rd("a.bin", np.float16, Wu.N).astype(np.float32)
    dn = rd("down.bin", np.float32, HX)
    xo = rd("x.bin", np.float32, HX)

    q_m = qr.rmsnorm(d["q"][tok].reshape(nh, hd).astype(np.float32),
                     L["qn"].astype(np.float32), m.eps) * scale
    q_m = qr.rope(q_m[:, None, :], cos[tok:tok + 1, None, :],
                  sin[tok:tok + 1, None, :])[:, 0].reshape(-1)
    k_m = qr.rmsnorm(d["k"][:nk].reshape(nk, nkv, hd).astype(np.float32),
                     L["kn"].astype(np.float32), m.eps)
    k_m = qr.rope(k_m, cos[:nk, None, :], sin[:nk, None, :]).reshape(nk, HK)

    # the same stages from the hardware's own fp16 h1 and packed weights:
    # what remains is RTL error, not int4/fp16 requantization
    h1_64 = h1[tok].astype(np.float64)
    q_p = (h1_64 @ Wq.recon.astype(np.float64)).astype(np.float32)
    q_pn = qr.rmsnorm(q_p.reshape(nh, hd), L["qn"].astype(np.float32),
                      m.eps) * scale
    q_pn = qr.rope(q_pn[:, None, :], cos[tok:tok + 1, None, :],
                   sin[tok:tok + 1, None, :])[:, 0].reshape(-1)
    k_p = (h1.astype(np.float64)[:nk] @ Wk.recon.astype(np.float64)
           ).astype(np.float32)                       # hw's k, every key
    k_pn = qr.rmsnorm(k_p.reshape(nk, nkv, hd), L["kn"].astype(np.float32),
                      m.eps)
    k_pn = qr.rope(k_pn, cos[:nk, None, :], sin[:nk, None, :]).reshape(nk, HK)

    # same-operand references.  Each takes the hardware's own fp16 / fp32
    # intermediate for the input and evaluates the operator in float64, so
    # what is left is the RTL, not int4 weights or fp16 activations.
    v_p = (h1.astype(np.float64)[:nk] @ Wv.recon.astype(np.float64)
           ).astype(np.float32)                        # hw's v, every key
    qh = qf.astype(np.float64).reshape(nh, hd)
    kh = kf.astype(np.float64).reshape(nk, nkv, hd)    # k per key, per kv head
    vh = v16.astype(np.float64).reshape(nk, nkv, hd)
    oi = np.zeros((nh, hd))
    for h in range(nh):
        s_ = qh[h] @ kh[:, h // rep, :].T              # all keys, one head
        p_ = np.exp(s_ - s_.max())
        oi[h] = (p_ / p_.sum()) @ vh[:, h // rep, :]
    oin_p = oi.reshape(-1).astype(np.float32)
    op_p = (oin.astype(np.float64) @ Wo.recon.astype(np.float64)
            ).astype(np.float32)
    xm_p = x_all[tok] + op
    h2_p = qr.rmsnorm(xm.astype(np.float64), L["ln2"].astype(np.float64),
                      m.eps).astype(np.float32)
    dn_p = (ma.astype(np.float64) @ Wd.recon.astype(np.float64)
            ).astype(np.float32)
    xo_p = xm + dn
    g_i = h2.astype(np.float64) @ Wg.recon.astype(np.float64)
    u_i = h2.astype(np.float64) @ Wu.recon.astype(np.float64)
    a_i = (g_i / (1.0 + np.exp(-g_i)) * u_i).astype(np.float32)

    # (label, stage result, same-operand reference)
    res, same = {}, {}
    if up >= 1:
        print("\n1 RMSNorm(h1)")
        res["h1"] = stat("h1", h1[tok], d["h1"][tok])
        same["h1"] = res["h1"]
    if up >= 2:
        print("\n2 projections")
        res["q"] = stat("q", qp, d["q"][tok])
        res["k"] = stat("k", kp, d["k"][:nk])          # every key position
        res["v"] = stat("v", vp, d["v"][:nk])
        same["q"] = stat("q/pk", qp, q_p)
        same["k"] = stat("k/pk", kp, k_p)
        same["v"] = stat("v/pk", vp, v_p)
    if up >= 3:
        print("\n3 qk-norm + rope")
        res["qn"] = stat("qf/mdl", qf, q_m)
        same["qn"] = stat("qf/pk", qf, q_pn)
        res["kn"] = stat("kf/mdl", kf, k_m)            # every key position
        same["kn"] = stat("kf/pk", kf, k_pn)
    if up >= 4:
        print("\n4 attention")
        res["attn"] = stat("o_in", oin, d["o_in"][tok])
        same["attn"] = stat("o_in/op", oin, oin_p)
    if up >= 5:
        print("\n5 o_proj + residual")
        res["o"] = stat("o", op, d["o"][tok])
        same["o"] = stat("o/pk", op, op_p)
        res["xm"] = stat("x_mid", xm, d["x_mid"][tok])
        same["xm"] = stat("xm/add", xm, xm_p)
    if up >= 6:
        print("\n6 mlp")
        res["h2"] = stat("h2", h2, d["h2"][tok])
        same["h2"] = stat("h2/pk", h2, h2_p)
        res["a"] = stat("mlp_a", ma, d["mlp_a"][tok])
        same["a"] = stat("a/pk", ma, a_i)
        res["down"] = stat("down", dn, d["down"][tok])
        same["down"] = stat("down/pk", dn, dn_p)
    if up >= 7:
        print("\n7 residual")
        res["x"] = stat("x_out", xo, d["x"][tok])
        same["x"] = stat("x/add", xo, xo_p)

    # The end-to-end reference that actually tests the RTL: the model's own
    # forward, in float64, on the weights the hardware really holds (the
    # per-block E8M0 reconstruction).  Whatever the RTL adds on top of the
    # weight format shows up here; the gap to d["x"] is the format's budget.
    if "x" in res:
        save = {k: L[k] for k in mats}
        for k, W in mats.items():
            L[k] = W.recon
        _, dp = m.forward(ids, dump=args.layer)
        for k in mats:
            L[k] = save[k]
        print("\nformat-limited prediction (this model re-run on the packed "
              "weights, float64)")
        res["pred"] = stat("x/pred", xo, dp["x"][tok])

    print("\nsummary")
    for k in ("h1", "q", "k", "v", "qn", "kn", "attn", "o", "xm", "h2", "a",
              "down", "x"):
        if k in res:
            print("  %-6s vs model %.6f   same operands %.6f" %
                  (k, res[k], same.get(k, float("nan"))))
    if "x" in res:
        worst = min(same.values())
        ok = res["pred"] > 0.999 and worst > 0.9995
        print("\n%s: x_out vs the packed weights' own model %.7f "
              "(1-cos %.1e), worst same-operand %.7f; vs the fp32 model "
              "%.6f (4-bit budget)" %
              ("PASS" if ok else "FAIL", res["pred"], 1.0 - res["pred"],
               worst, res["x"]))




def chain_analysis(m, ids, tok, layers, ds, paths_by, mats_by, HX):
    """two layers where the second is fed the first one's x_out.

    The reference is the network's own forward, started at the second layer
    with an input whose query-token row is the bytes the NPU wrote: the other
    rows are the model's own layer-0 output, which is what the NPU would have
    produced at those positions had it been asked for them.
    """
    def rd(li, name):
        return np.frombuffer(open(paths_by[li][name], "rb").read(),
                             np.float32)[:HX]

    first, second = layers[0], layers[1]
    xo0 = rd(first, "x.bin")
    xo1 = rd(second, "x.bin")

    print("\nlayer %d (fed by the model)" % first)
    c0 = stat("x_out", xo0, ds[first]["x"][tok])

    print("\nlayer %d (fed by layer %d's x_out)" % (second, first))
    stat("x_out", xo1, ds[second]["x"][tok])

    # the same reference with the query-token row replaced by what the NPU
    # actually handed over, so layer 0's own error does not count twice
    x_start = ds[first]["x"].astype(np.float32).copy()
    x_start[tok] = xo0
    _, df = m.forward(ids, dump=second, start=second, x_start=x_start)
    print("\nlayer %d from the NPU's own layer-0 row (network re-run, "
          "float64 operands in fp32)" % second)
    res = stat("x_out", xo1, df["x"][tok])

    # and the same with the weights the hardware holds
    save = {li: dict(m.L[li]) for li in layers}   # rebind, do not mutate
    for li in layers:
        for k, W in mats_by[li].items():
            m.L[li][k] = W.recon
    _, dp = m.forward(ids, dump=second, start=second, x_start=x_start)
    for li in layers:
        m.L[li] = save[li]
    pred = stat("x_out", xo1, dp["x"][tok])
    print("\n%s: layer %d fed the NPU's layer-%d row vs the network "
          "%.7f (1-cos %.1e), vs the packed weights %.7f; layer %d vs the "
          "model %.6f" %
          ("PASS" if pred > 0.999 else "FAIL", second, first, res,
           1.0 - res, pred, first, c0))


if __name__ == "__main__":
    main()

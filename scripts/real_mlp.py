"""real_mlp.py : run a whole SwiGLU MLP block of a real Qwen3-0.6B layer in RTL.

The block is the one the network actually computes:

    h2 = RMSNorm(x)            VECTOR + REDUCE + SFU
    g  = h2 @ Wg               MATMUL, int4 weights, fp16 activation
    u  = h2 @ Wu               MATMUL
    a  = silu(g) * u           MUL_SCALAR, SFU.EXP2, ADD_SCALAR, SFU.RCP, MUL
    y  = a @ Wd                MATMUL, K = 3072

The reference is the real fp32 forward of the checkpoint (scripts/qwen3_ref.py),
not a model of the hardware.  The hardware additionally rounds the activation to
fp16 and requantizes the weights to int4 + E8M0, so the comparison is a
tolerance check; the script also prints an exact fp32 matmul over the *packed*
weights, which separates requantization error from RTL error.

    python scripts/real_mlp.py --layer 0 --token -1
    python scripts/real_mlp.py --layer 13 --token 0 --repack
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

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# scratchpad logical buffers (each maps to its own bank, 8 KiB)
B_X, B_H, B_G, B_A, B_D, B_T, B_S0, B_S1 = 0, 1, 2, 3, 4, 5, 6, 7

# external memory map (bytes); weights start above the small operands
X_EXT, GAM_EXT, H_EXT, A_EXT, D_EXT = 0x1000, 0x2000, 0x3000, 0x4000, 0x6000
DBG_EXT = 0xA000
CL_EXT = 0x5000                       # the silu clamp constant
W_EXT = 0x10000

# gate/up: two weight tiles of eight groups = 64 output columns per MATMUL
CHUNK_TILES, NGATE = 2, 8
CHUNK_COLS = 4 * NGATE * CHUNK_TILES
GATE_STEPS = 64                       # K = 1024, W4 = 16 elements per step

# down: K = 3072 needs 192 steps, so a tile shrinks to four groups (16 columns)
D_TILES, NGD = 4, 4
DOWN_COLS = 4 * NGD * D_TILES
DOWN_STEPS = 192

# gate workspace in B_G: g at 0, u at 512, exp2 at 1024, rcp at 1536
G_GATE, G_UP, G_E, G_R = 0, 512, 1024, 1536
# temp workspace in B_T: g*-log2e at 0, 1+exp at 512, silu at 1024,
# the clamped exp2 argument at 2048
T_T, T_P1, T_S, T_C = 0, 512, 1024, 2048

# SiLU needs 1/(1+exp(-g)); hardware exp2 must return a finite value, so the
# exp2 argument is clamped at 126.  For g below about -87 the model's own
# sigmoid is below 1e-38 and silu(g) is zero to fp32, so the clamp costs
# nothing but keeps the SFU out of its NUMERIC status.
# silu is g / (1 + exp(-g)); the hardware evaluates exp(-g) as
# exp2(-g*log2e), and that exponent leaves fp32 above about 127, i.e. for
# g below about -88.  There sigmoid(g) is far under the fp16 floor (6e-8),
# so the exponent is clamped at -100: exp2 stays finite and the reciprocal
# stays well inside the normal range (a -126 clamp lands RCP exactly on the
# smallest normal fp32 and rounds into underflow, which the SFU reports as
# a NUMERIC failure).  VECTOR has no MIN, so the clamp is
# expressed as max(u, -126) on u = g*log2e and the sign is folded back in.
CLAMP_OFF, SILU_CLAMP = 32, -100.0

F32, F16 = nc.F_FP32, nc.F_FP16


def f32(x):
    """the 32-bit pattern of the fp32 nearest to x"""
    return int(np.float32(x).view(np.uint32))


class Script:
    """command stream with job ids, a reusable slot pool and window release"""

    def __init__(self, job0=100):
        self.lines = []
        self.job = job0
        self.slot = -1              # first add() gets slot 0
        self.win = []

    def njob(self):
        self.job += 1
        return self.job

    def nslot(self):
        # EVENT_SLOTS is 16 on the RTL side; a window that asks for more is a
        # scheduling bug, not something to discover from a 0004 completion
        self.slot += 1
        if self.slot >= 16:
            raise RuntimeError("window needs more than 16 event slots")
        return self.slot

    def put(self, c, j, s):
        self.lines.append("CMD " + nc.hex128(c))
        self.win.append((j, s))

    def add(self, build):
        """build(job, slot) -> command; returns (job, slot)"""
        j, s = self.njob(), self.nslot()
        self.put(build(j, s), j, s)
        return (j, s)

    def close(self):
        """everything in the window is complete: WAIT the last, then release"""
        if not self.win:
            return
        self.lines.append("WAIT %d" % self.win[-1][0])
        for j, s in self.win:
            self.lines.append("CMD " + nc.hex128(nc.release(self.njob(), s, j)))
        self.win = []
        self.slot = -1


def dep(prev, flags=nc.FLAG_DEP):
    return dict(flags=flags, dep_job=prev[0], dep_slot=prev[1])


def pack_all(Wg, Wu, Wd, nchunk):
    """pack the three weight matrices into the byte stream LOAD_W moves"""
    K, N = Wg.shape
    gate, up, dblob = bytearray(), bytearray(), bytearray()
    rec_g = np.zeros((K, N), np.float32)
    rec_u = np.zeros((K, N), np.float32)
    rec_d = np.zeros(Wd.shape, np.float32)

    for c in range(nchunk):
        for t in range(CHUNK_TILES):
            col0 = c * CHUNK_COLS + 4 * NGATE * t
            w, s, r = hw_pack.pack_tile(
                Wg[:, col0:col0 + 4 * NGATE].astype(np.float32), n_grp=NGATE)
            gate += w + s
            rec_g[:, col0:col0 + 4 * NGATE] = r
            w, s, r = hw_pack.pack_tile(
                Wu[:, col0:col0 + 4 * NGATE].astype(np.float32), n_grp=NGATE)
            up += w + s
            rec_u[:, col0:col0 + 4 * NGATE] = r
    for c in range(Wd.shape[1] // DOWN_COLS):
        for t in range(D_TILES):
            col0 = c * DOWN_COLS + 4 * NGD * t
            w, s, r = hw_pack.pack_tile(Wd[:, col0:col0 + 4 * NGD], n_grp=NGD)
            dblob += w + s
            rec_d[:, col0:col0 + 4 * NGD] = r
    return bytes(gate), bytes(up), bytes(dblob), rec_g, rec_u, rec_d


def metrics(got, ref):
    e = got.astype(np.float64) - ref.astype(np.float64)
    rr = float(np.sqrt(np.mean(e ** 2)))
    rr_ref = float(np.sqrt(np.mean(ref.astype(np.float64) ** 2)))
    cs = float(got.astype(np.float64) @ ref.astype(np.float64)
               / np.linalg.norm(got) / np.linalg.norm(ref))
    return rr, rr_ref, cs


def stat(name, got, ref):
    rr, rr_ref, cs = metrics(got, ref)
    return "  %-5s RMS err %.4e (%.1f%% of RMS)  cos %.6f" % (
        name, rr, 100 * rr / rr_ref, cs)


def run_block(m, args, ids, layer, token, outdir):
    """run one MLP block (layer, token) and print the comparison"""
    _, d = m.forward(ids, dump=layer)
    tok = token

    x = d["x_mid"][tok].astype(np.float32)          # MLP block input
    gam = m.L[layer]["ln2"].astype(np.float32)
    K = x.shape[0]
    Wg = m.L[layer]["wg"]
    Wu = m.L[layer]["wu"]
    Wd = m.L[layer]["wd"]
    N = Wg.shape[1]

    ref_a = d["mlp_a"][tok]
    ref_y = d["down"][tok]

    nchunk_all = N // CHUNK_COLS
    nchunk = min(args.chunks, nchunk_all) if args.chunks else nchunk_all
    full = (nchunk == nchunk_all)

    os.makedirs(outdir, exist_ok=True)
    cache = os.path.join(args.outdir, "pack_L%d.npz" % layer)
    if os.path.exists(cache) and not args.repack:
        z = np.load(cache)
        gate_b, up_b, down_b = z["gate"], z["up"], z["down"]
        rec_g, rec_u, rec_d = z["rg"], z["ru"], z["rd"]
    else:
        print("packing weights (int4 + E8M0) ...")
        g, u, dn, rec_g, rec_u, rec_d = pack_all(Wg, Wu, Wd, nchunk_all)
        gate_b = np.frombuffer(g, np.uint8)
        up_b = np.frombuffer(u, np.uint8)
        down_b = np.frombuffer(dn, np.uint8)
        np.savez(cache, gate=gate_b, up=up_b, down=down_b,
                 rg=rec_g, ru=rec_u, rd=rec_d)
    gate_b, up_b, down_b = gate_b.tobytes(), up_b.tobytes(), down_b.tobytes()

    # ------------------------------------------------------------------
    # memory image
    # ------------------------------------------------------------------
    img = bytearray(1 << 24)
    img[X_EXT:X_EXT + 4 * K] = x.tobytes()
    img[GAM_EXT:GAM_EXT + 4 * K] = gam.tobytes()
    img[CL_EXT:CL_EXT + 4] = np.float32(SILU_CLAMP).tobytes()
    GW_EXT = W_EXT
    UW_EXT = GW_EXT + len(gate_b)
    DW_EXT = UW_EXT + len(up_b)
    img[GW_EXT:GW_EXT + len(gate_b)] = gate_b
    img[UW_EXT:UW_EXT + len(up_b)] = up_b
    img[DW_EXT:DW_EXT + len(down_b)] = down_b
    mem_bin = os.path.join(outdir, "mem.bin")
    open(mem_bin, "wb").write(bytes(img))

    # one LOAD_W tile = groups * steps * 32 B weights, then the padded scales
    WB_G = NGATE * GATE_STEPS * 32
    TB_G = WB_G + NGATE * ((GATE_STEPS + 3) // 4) * 16
    WB_D = NGD * DOWN_STEPS * 32
    TB_D = WB_D + NGD * ((DOWN_STEPS + 3) // 4) * 16

    S = Script()

    # ------------------------------------------------------------------
    # RMSNorm:  y = x * rsqrt(mean(x^2) + eps) * gamma
    # ------------------------------------------------------------------
    S.add(lambda j, s: nc.load(j, X_EXT, B_X, row_bytes=4 * K, slot=s))
    S.add(lambda j, s: nc.load(j, GAM_EXT, B_D, row_bytes=4 * K, slot=s))
    S.add(lambda j, s: nc.load(j, CL_EXT, B_S1, off=CLAMP_OFF, row_bytes=4,
                               slot=s))
    f0 = S.add(lambda j, s: nc.fence(j, s))
    sq = S.add(lambda j, s: nc.vector(j, nc.V_MUL, K, B_X, 0, B_T, 0, s,
                                      s1=B_X, o1=0, **dep(f0)))
    sm = S.add(lambda j, s: nc.reduce(j, nc.R_SUM, K, B_T, 0, B_S0, 0, s,
                                      **dep(sq)))
    ms = S.add(lambda j, s: nc.vector(j, nc.V_MUL_SCALAR, 1, B_S0, 0, B_S1, 0, s,
                                      scalar=f32(1.0 / K), **dep(sm)))
    ve = S.add(lambda j, s: nc.vector(j, nc.V_ADD_SCALAR, 1, B_S1, 0, B_S0, 0, s,
                                      scalar=f32(1e-6), **dep(ms)))
    rs = S.add(lambda j, s: nc.sfu(j, nc.S_RSQRT, 1, B_S0, 0, B_S1, 0, s,
                                   **dep(ve)))
    xr = S.add(lambda j, s: nc.vector(j, nc.V_MUL | nc.V_BCAST_S, K,
                                      B_X, 0, B_T, 0, s, s1=B_S1, o1=0,
                                      **dep(rs)))
    h2 = S.add(lambda j, s: nc.vector(j, nc.V_MUL, K, B_T, 0, B_H, 0, s,
                                      s1=B_D, o1=0, fd=F16, **dep(xr)))
    S.add(lambda j, s: nc.store(j, H_EXT, B_H, 0, 2 * K, slot=s, **dep(h2)))
    S.close()

    # ------------------------------------------------------------------
    # gate/up per 64 columns, then SiLU * u stored as fp16 into the down A
    # ------------------------------------------------------------------
    for c in range(nchunk):
        # gate: two tiles fill the two banks, then the MATMUL consumes them
        mm_g = S.njob()
        sl_g = S.nslot()
        for t in range(CHUNK_TILES):
            off = (c * CHUNK_TILES + t) * TB_G
            S.add(lambda j, s, off=off: nc.load_w(
                j, GW_EXT + off, GW_EXT + off + WB_G, mm_g, t, 0,
                NGATE, GATE_STEPS, TB_G, slot=s))
        S.put(nc.matmul(mm_g, 1, NGATE, 0, CHUNK_TILES, GATE_STEPS, K,
                        CHUNK_COLS, 0, 2 * K, G_GATE, 4 * CHUNK_COLS,
                        B_H, B_G, sl_g), mm_g, sl_g)

        mm_u = S.njob()
        sl_u = S.nslot()
        for t in range(CHUNK_TILES):
            off = (c * CHUNK_TILES + t) * TB_G
            S.add(lambda j, s, off=off: nc.load_w(
                j, UW_EXT + off, UW_EXT + off + WB_G, mm_u, t, 0,
                NGATE, GATE_STEPS, TB_G, slot=s))
        S.put(nc.matmul(mm_u, 1, NGATE, 0, CHUNK_TILES, GATE_STEPS, K,
                        CHUNK_COLS, 0, 2 * K, G_UP, 4 * CHUNK_COLS,
                        B_H, B_G, sl_u), mm_u, sl_u)

        fn = S.add(lambda j, s: nc.fence(j, s))
        v1 = S.add(lambda j, s: nc.vector(j, nc.V_MUL_SCALAR, CHUNK_COLS,
                                          B_G, G_GATE, B_T, T_T, s,
                                          scalar=f32(1.4426950408889634),
                                          **dep(fn)))
        vc = S.add(lambda j, s: nc.vector(j, nc.V_MAX | nc.V_BCAST_S,
                                          CHUNK_COLS, B_T, T_T, B_T, T_T, s,
                                          s1=B_S1, o1=CLAMP_OFF,
                                          flags=nc.FLAG_INPLACE | nc.FLAG_DEP,
                                          dep_job=v1[0], dep_slot=v1[1]))
        vn = S.add(lambda j, s: nc.vector(j, nc.V_MUL_SCALAR, CHUNK_COLS,
                                          B_T, T_T, B_T, T_T, s,
                                          scalar=f32(-1.0),
                                          flags=nc.FLAG_INPLACE | nc.FLAG_DEP,
                                          dep_job=vc[0], dep_slot=vc[1]))
        v2 = S.add(lambda j, s: nc.sfu(j, nc.S_EXP2, CHUNK_COLS,
                                       B_T, T_T, B_G, G_E, s, **dep(vn)))
        v3 = S.add(lambda j, s: nc.vector(j, nc.V_ADD_SCALAR, CHUNK_COLS,
                                          B_G, G_E, B_T, T_P1, s,
                                          scalar=f32(1.0), **dep(v2)))
        v4 = S.add(lambda j, s: nc.sfu(j, nc.S_RCP, CHUNK_COLS,
                                       B_T, T_P1, B_G, G_R, s, **dep(v3)))
        v5 = S.add(lambda j, s: nc.vector(j, nc.V_MUL, CHUNK_COLS,
                                          B_G, G_GATE, B_T, T_S, s,
                                          s1=B_G, o1=G_R, **dep(v4)))
        v6 = S.add(lambda j, s: nc.vector(j, nc.V_MUL, CHUNK_COLS,
                                          B_T, T_S, B_A, c * CHUNK_COLS * 2, s,
                                          s1=B_G, o1=G_UP, fd=F16, **dep(v5)))

        if args.debug and c == 0:
            S.add(lambda j, s: nc.store(j, DBG_EXT, B_G, G_GATE,
                                        4 * CHUNK_COLS, slot=s, **dep(v6)))
            S.add(lambda j, s: nc.store(j, DBG_EXT + 0x400, B_A, 0,
                                        2 * CHUNK_COLS, slot=s, **dep(v6)))
        if full and c == nchunk - 1:
            S.add(lambda j, s: nc.store(j, A_EXT, B_A, 0, 2 * N,
                                        slot=s, **dep(v6)))
        S.close()

    # ------------------------------------------------------------------
    # down_proj: K = 3072, four groups per tile so the tile fits a bank
    # ------------------------------------------------------------------
    if full:
        # A scratchpad logical buffer is one 8 KiB bank window, and the
        # down_proj result is 3072 fp32 (12 KiB), so each 64-column chunk is
        # drained to memory as soon as it lands instead of being accumulated.
        for c in range(Wd.shape[1] // DOWN_COLS):
            mm = S.njob()
            sl = S.nslot()
            for t in range(D_TILES):
                off = (c * D_TILES + t) * TB_D
                S.add(lambda j, s, off=off: nc.load_w(
                    j, DW_EXT + off, DW_EXT + off + WB_D, mm, t, 0,
                    NGD, DOWN_STEPS, TB_D, slot=s))
                if t == 1:            # the core can start once tile 0 is in
                    S.put(nc.matmul(mm, 1, NGD, 0, D_TILES, DOWN_STEPS,
                                    Wd.shape[0], DOWN_COLS, 0, 2 * Wd.shape[0],
                                    0, 4 * DOWN_COLS,
                                    B_A, B_D, sl), mm, sl)
            S.add(lambda j, s: nc.store(j, D_EXT + c * DOWN_COLS * 4, B_D, 0,
                                        4 * DOWN_COLS, slot=s, **dep((mm, sl))))
            S.close()

    dump_h = os.path.join(outdir, "h2.bin")
    dump_a = os.path.join(outdir, "a.bin")
    dump_d = os.path.join(outdir, "d.bin")
    # the harness parses the dump length as hex
    S.lines.append("DUMP %x %x %s" % (H_EXT, 2 * K, dump_h))
    if full:
        S.lines.append("DUMP %x %x %s" % (A_EXT, 2 * N, dump_a))
        S.lines.append("DUMP %x %x %s" % (D_EXT, 4 * Wd.shape[1], dump_d))
    if args.debug:
        S.lines.append("DUMP %x %x %s" % (DBG_EXT, 4 * CHUNK_COLS,
                                          os.path.join(outdir, "g0.bin")))
        S.lines.append("DUMP %x %x %s" % (DBG_EXT + 0x400, 2 * CHUNK_COLS,
                                          os.path.join(outdir, "a0.bin")))

    script = os.path.join(outdir, "cmds.txt")
    open(script, "w").write("\n".join(S.lines) + "\n")
    print("%d commands, %d script lines" %
          (sum(1 for l in S.lines if l.startswith("CMD")), len(S.lines)))

    r = subprocess.run([os.path.abspath(args.bin), "--mem",
                        os.path.abspath(mem_bin), "--script",
                        os.path.abspath(script)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-4000:])
        print(r.stderr.strip())
        sys.exit("harness failed rc=%d" % r.returncode)
    out = r.stdout
    cpl = [l for l in out.splitlines() if l.startswith("CPL ")]
    errs = [l for l in cpl if l.split()[4] != "0000"]
    done = [l for l in out.splitlines() if l.startswith("DONE")]
    print(done[0] if done else out.strip().splitlines()[-1])
    res = {"cycles": int(done[0].split("=")[1]) if done else 0, "errs": len(errs)}
    if errs:
        print("%d of %d completions are not OK, first:" % (len(errs), len(cpl)))
        for l in errs[:10]:
            print("  " + l)

    # ------------------------------------------------------------------
    # compare
    # ------------------------------------------------------------------
    h2r = np.frombuffer(open(dump_h, "rb").read(), np.float16)[:K]
    h2r = h2r.astype(np.float32)
    print("\nRMSNorm (hardware fp16) vs the real h2")
    print(stat("h2", h2r, d["h2"][tok]))
    res["h2"] = metrics(h2r, d["h2"][tok])

    if args.debug:
        g0 = np.frombuffer(open(os.path.join(outdir, "g0.bin"), "rb").read(),
                           np.float32)
        a0 = np.frombuffer(open(os.path.join(outdir, "a0.bin"), "rb").read(),
                           np.float16).astype(np.float32)
        print("\nchunk 0 gate vs the real g")
        print(stat("g", g0, d["g"][tok][:CHUNK_COLS]))
        print(stat("a", a0, ref_a[:CHUNK_COLS]))
        # the same columns through an exact fp32 matmul over the *packed*
        # weights: what remains is the RTL, not the int4 requantization
        h64 = h2r.astype(np.float64)
        gi = (h64 @ rec_g[:, :CHUNK_COLS].astype(np.float64)).astype(np.float32)
        ui = (h64 @ rec_u[:, :CHUNK_COLS].astype(np.float64)).astype(np.float32)
        ai = ((gi / (1.0 + np.exp(-gi))) * ui)
        print("chunk 0 vs the exact matmul over the packed weights")
        print(stat("g", g0, gi))
        print(stat("a", a0, ai))

    if not full:
        print("\npartial run (chunks=%d): down_proj skipped" % nchunk)
        return res

    a_hw = np.frombuffer(open(dump_a, "rb").read(), np.float16)[:N].astype(np.float32)
    y = np.frombuffer(open(dump_d, "rb").read(), np.float32)[:Wd.shape[1]]

    # exact fp32 matmul over the packed weights, fed the hardware's own fp16
    # activation: what is left is the RTL, not the requantization
    h64 = h2r.astype(np.float64)
    g_i = h64 @ rec_g.astype(np.float64)
    u_i = h64 @ rec_u.astype(np.float64)
    a_i = ((g_i / (1.0 + np.exp(-g_i))) * u_i).astype(np.float32)
    y_pack = (a_hw.astype(np.float64) @ rec_d.astype(np.float64)).astype(np.float32)

    print("\nmlp_a vs the real model")
    print(stat("a", a_hw, ref_a))
    print("mlp_a vs the silu of the packed gate/up")
    print(stat("a", a_hw, a_i))

    print("\ndown_proj vs the real model")
    print(stat("rtl", y, ref_y))
    print(stat("pack", y_pack, ref_y))
    print("down_proj vs the exact matmul over packed weights")
    print(stat("rtl", y, y_pack))

    res["a_real"] = metrics(a_hw, ref_a)
    res["a_pack"] = metrics(a_hw, a_i)
    res["y_real"] = metrics(y, ref_y)
    res["y_pack"] = metrics(y, y_pack)
    return res


def fmt(m):
    """(rms, rms_ref, cos) -> '2.4e-05/1.000000'"""
    return "%.1e/%.6f" % (m[0], m[2]) if m else "-"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=qr.DEFAULT_MODEL)
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--token", type=int, default=-1)
    ap.add_argument("--layers", default=None,
                    help="comma separated layers: a sweep, one forward each")
    ap.add_argument("--tokens", default=None,
                    help="comma separated token positions")
    ap.add_argument("--bin", default="build/vsim/npu_top/Vnpu_top.exe")
    ap.add_argument("--outdir", default="build/real_mlp")
    ap.add_argument("--repack", action="store_true", help="ignore the pack cache")
    ap.add_argument("--chunks", type=int, default=0,
                    help="limit the gate/up chunks (debug; skips down_proj)")
    ap.add_argument("--debug", action="store_true",
                    help="store chunk 0's g and a for inspection")
    args = ap.parse_args()

    layers = [int(v) for v in args.layers.split(",")] if args.layers \
             else [args.layer]
    tokens = [int(v) for v in args.tokens.split(",")] if args.tokens \
             else [args.token]

    m = qr.Qwen3(args.model)
    ids = m.tok.encode(args.prompt)

    rows = []
    for layer in layers:
        for token in tokens:
            tag = os.path.join(args.outdir, "L%d_t%d" % (layer, token))
            try:
                r = run_block(m, args, ids, layer, token, tag)
            except SystemExit as e:
                print("!! block failed: %s" % e)
                r = {"cycles": 0, "errs": 1}
            rows.append((layer, token, r))
            if r["errs"]:
                print("!! %d completions with a non-zero status" % r["errs"])

    if len(rows) > 1:
        print("\nlayer tok      cycles   h2 cos  a/pack cos  a/real cos"
              "  y/pack cos  y/real cos")
        for layer, token, r in rows:
            print("%5d %3d %11d  %8s  %10s  %10s  %10s  %10s" % (
                layer, token, r["cycles"],
                fmt(r.get("h2")), fmt(r.get("a_pack")), fmt(r.get("a_real")),
                fmt(r.get("y_pack")), fmt(r.get("y_real"))))


if __name__ == "__main__":
    main()

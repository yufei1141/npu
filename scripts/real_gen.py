"""real_gen.py : Qwen3-0.6B prefill + decode, token by token, through the NPU.

The network keeps the 0.6B structure and is cut to `--layers` layers (3 by
default) so that an autoregressive run fits in simulation time.  What runs
where:

    embedding lookup                        host CPU (this script)
    every layer, the final RMSNorm,
    the lm_head GEMV, the k/v cache         the NPU

Each position is one command stream and one harness run:

    x_row[p]  <- CPU embedding of ids[p]          (host, into the image)
    for each layer i:
        x_row[p] <- layer i-1's x_out row          (in the NPU)
        h1, q of position p                        (only the new position)
        k, v of position p  -> KF/V16 row p        (appended to the cache)
        attention over positions 0..p              (reads the cache)
        o_proj, MLP, both residuals
    hf = RMSNorm(x_out, final)
    logits = hf @ lm_head                          (int4 weights, 151936 cols)

The first `T` positions are the prompt ("prefill"); after that the next token
is the argmax of the logits the NPU produced, so decode is self-driven.  The
k/v cache survives between runs because the image is carried forward: the
rows the NPU wrote are spliced back into this script's copy of it.

The reference is the checkpoint's own forward, re-run on every prefix with
the *packed* weights in place, so what is compared is the RTL against the
4-bit model it is supposed to implement.

    python scripts/real_gen.py --prompt "The capital of France is" --decode 3
"""
import argparse
import os
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qwen3_ref as qr
import hw_pack
import npu_cmd as nc
import real_layer as rl

HD, HX, HQ, HK = 128, 1024, 2048, 1024
NGRP, NTILES = rl.NGATE, rl.CHUNK_TILES
LM_COLS = 4 * NGRP * NTILES                  # output columns per lm_head chunk

# ----------------------------------------------------------------------
# memory map.  A layer owns LAYER_STRIDE bytes: 15 positions of k/v cache
# and 16 rows of activations below 1 MiB, its packed weights above.  The
# global area after the last layer holds the final norm and the lm_head.
# ----------------------------------------------------------------------
_LAY = {
    "X":   0x01000,   # fp32 x_in rows, 16 positions
    "H1":  0x11000,   # fp16 h1 rows
    "QP":  0x19000,   # fp32 q of the new position
    "KP":  0x1B000,   # fp32 k rows
    "VP":  0x2B000,   # fp32 v rows
    "QF":  0x3B000,   # fp16 q after norm + rope
    "KF":  0x3C000,   # fp16 k after norm + rope, the cache
    "V16": 0x44000,   # fp16 v, the cache
    "OIN": 0x4C000,   # fp16 attention output
    "OP":  0x4D000,   # fp32 o projection
    "XM":  0x4E000,   # fp32 x_mid
    "OUT": 0x4F000,   # fp32 x_out
    "G1":  0x50000,   # ln1 gamma
    "G2":  0x51000,   # ln2 gamma
    "QN":  0x52000,   # q_norm gamma
    "KN":  0x52200,   # k_norm gamma
    "COS": 0x52400,   # cos[hd/2] | sin[hd/2] per position
    "H2":  0x54400,   # fp16 h2
    "MA":  0x54C00,   # fp16 mlp activation
    "DW":  0x56400,   # fp32 down_proj
    "CL":  0x57400,   # silu clamp constant
    "W":   0x80000,   # packed weights
}
LAYER_STRIDE = 0xC00000          # 12 MiB per layer
GBASE = 3 * LAYER_STRIDE         # global area
FIN_EXT = GBASE                  # final norm gamma
HF_EXT = GBASE + 0x1000          # fp16 final hidden row
LOG_EXT = GBASE + 0x2000         # fp32 logits
LMW_EXT = GBASE + 0x100000       # lm_head weight tiles
IMG_SIZE = 0x7D00000             # 125 MiB
MAX_POS = 16                     # X and the k/v cache hold 16 rows each


def set_layers(n):
    """lay the global area out past the last layer; the whole network needs
    a 425 MiB image, the three-layer demo the 125 MiB the defaults give"""
    global GBASE, FIN_EXT, HF_EXT, LOG_EXT, LMW_EXT, IMG_SIZE
    GBASE = n * LAYER_STRIDE
    FIN_EXT = GBASE
    HF_EXT = GBASE + 0x1000
    LOG_EXT = GBASE + 0x2000
    LMW_EXT = GBASE + 0x100000
    IMG_SIZE = LMW_EXT + 0x5800000       # int4 lm_head (87.5 MB) + slack


def use(base):
    """point real_layer's stage emitters at one layer's region"""
    for k, v in _LAY.items():
        setattr(rl, k + "_EXT", base + v)


def cos_of(got, ref):
    rr, n, c = rl.metrics(np.asarray(got, np.float64),
                          np.asarray(ref, np.float64))
    return c, rr, n


# ----------------------------------------------------------------------
# lm_head: the tied embedding, [V, K] int4-coded one column of the GEMV at
# a time.  Sixteen blocks of output columns share one chunk, exactly like a
# layer's weight matrix, so the same tile packing applies.
# ----------------------------------------------------------------------
class LMat:
    def __init__(self, V, K, base, img, recon):
        self.K, self.N = K, V
        self.ngrp, self.ntiles = NGRP, NTILES
        self.cols = LM_COLS
        self.steps = (K + 15) // 16
        wb = self.ngrp * self.steps * 32
        tb = wb + self.ngrp * ((self.steps + 3) // 4) * 16
        self.nchunk = (V + self.cols - 1) // self.cols
        if img is None:                       # cache hit: addresses are arithmetic
            self.tiles = [[(base + (c * self.ntiles + t) * tb,
                            base + (c * self.ntiles + t) * tb + wb, tb)
                           for t in range(self.ntiles)]
                          for c in range(self.nchunk)]
            return
        off = base
        self.tiles = []
        for c in range(self.nchunk):
            row = []
            for t in range(self.ntiles):
                col0 = c * self.cols + t * 4 * self.ngrp
                w, s, r = hw_pack.pack_tile(
                    recon[col0:col0 + 4 * self.ngrp].T.astype(np.float32),
                    n_grp=self.ngrp)
                img[off:off + len(w)] = w
                img[off + len(w):off + len(w) + len(s)] = s
                row.append((off, off + len(w), len(w) + len(s)))
                off += len(w) + len(s)
            self.tiles.append(row)
        self.end = off


def stage_lm_head(img, embed, outdir, repack):
    """pack the embedding as lm_head weights; returns (LMat, recon fp16 [V,K])"""
    V, K = embed.shape
    cache = os.path.join(outdir, "lm_head.npz")
    if os.path.exists(cache) and not repack:
        z = np.load(cache)
        blob = z["blob"]
        img[LMW_EXT:LMW_EXT + len(blob)] = blob.tobytes()
        recon = z["recon"]                     # fp16 is exact for int4*2^e
        print("lm_head from cache: %.1f MB, %d chunks" %
              (len(blob) / 1e6, (V + LM_COLS - 1) // LM_COLS))
        return LMat(V, K, LMW_EXT, None, recon), recon

    # a tile holds 32 columns: pack_tile wants [K, 32]
    ng = NGRP
    nchunk = (V + LM_COLS - 1) // LM_COLS
    recon = np.zeros((V, K), np.float16)
    off = LMW_EXT
    tiles = []
    t0 = time.time()
    for c in range(nchunk):
        row = []
        for t in range(NTILES):
            col0 = c * LM_COLS + t * 4 * ng
            n = min(4 * ng, V - col0)
            w, s, r = hw_pack.pack_tile_fast(
                embed[col0:col0 + n].T.astype(np.float32), n_grp=ng)
            img[off:off + len(w)] = w
            img[off + len(w):off + len(w) + len(s)] = s
            row.append((off, off + len(w), len(w) + len(s)))
            recon[col0:col0 + n] = r.T[:n]
            off += len(w) + len(s)
        tiles.append(row)
        if c and c % 200 == 0:
            print("  lm_head %d/%d chunks, %.0f s" %
                  (c, nchunk, time.time() - t0))
    lm = LMat(V, K, LMW_EXT, None, recon)
    lm.tiles = tiles
    np.savez(cache, blob=np.frombuffer(bytes(img[LMW_EXT:off]), np.uint8),
             recon=recon)
    print("lm_head packed: %.1f MB, %d chunks, %.0f s" %
          ((off - LMW_EXT) / 1e6, nchunk, time.time() - t0))
    return lm, recon


def emit_lm_head(S, lm, hf_ext, log_ext, nchunk=None):
    """hf (fp16 row) through the packed embedding -> fp32 logits"""
    S.add(lambda j, s: nc.load(j, hf_ext, rl.B_H, row_bytes=2 * HX, slot=s))
    S.close()
    for c in range(lm.nchunk if nchunk is None else nchunk):
        mm, sl = S.njob(), S.nslot()
        for t, (wa, sa, nb) in enumerate(lm.tiles[c]):
            S.add(lambda j, s, t=t, wa=wa, sa=sa, nb=nb: nc.load_w(
                j, wa, sa, mm, t, 0, lm.ngrp, lm.steps, nb, slot=s))
        S.put(nc.matmul(mm, 1, lm.ngrp, 0, lm.ntiles, lm.steps, lm.K,
                        lm.cols, 0, 2 * lm.K, 0, 4 * lm.cols,
                        rl.B_H, rl.B_D, sl), mm, sl)
        S.add(lambda j, s, c=c: nc.store(j, log_ext + 4 * lm.cols * c,
                                         rl.B_D, 0, 4 * lm.cols, slot=s,
                                         **rl.dep((mm, sl))))
        S.close()


def fill_layer(img, base, L, cos, sin, npos):
    """gammas, per-position rope angles, the silu clamp"""
    img[base + _LAY["G1"]:base + _LAY["G1"] + 4 * HX] = \
        L["ln1"].astype(np.float32).tobytes()
    img[base + _LAY["G2"]:base + _LAY["G2"] + 4 * HX] = \
        L["ln2"].astype(np.float32).tobytes()
    img[base + _LAY["QN"]:base + _LAY["QN"] + 4 * HD] = \
        L["qn"].astype(np.float32).tobytes()
    img[base + _LAY["KN"]:base + _LAY["KN"] + 4 * HD] = \
        L["kn"].astype(np.float32).tobytes()
    for t in range(npos):
        cs = np.concatenate([cos[t][:HD // 2], sin[t][:HD // 2]])
        img[base + _LAY["COS"] + 4 * HD * t:
            base + _LAY["COS"] + 4 * HD * (t + 1)] = cs.astype(np.float32).tobytes()
    img[base + _LAY["CL"]:base + _LAY["CL"] + 4] = \
        np.float32(rl.SILU_CLAMP).tobytes()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=qr.DEFAULT_MODEL)
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--decode", type=int, default=3,
                    help="tokens generated after the prompt")
    ap.add_argument("--bin", default="build/vsim/qk/Vnpu_top.exe")
    ap.add_argument("--outdir", default="build/real_gen")
    ap.add_argument("--repack", action="store_true",
                    help="ignore the packed-weight caches")
    ap.add_argument("--dry", action="store_true",
                    help="build the images and streams, do not simulate")
    ap.add_argument("--fin-upto", type=int, default=2,
                    help="1: stop after the final norm, 2: lm_head too")
    ap.add_argument("--lm-chunks", type=int, default=0,
                    help="simulate only the first N lm_head chunks (timing)")
    ap.add_argument("--ddr-gbps", type=float, default=0.0,
                    help="DDR bandwidth the harness models (needs --fcore-mhz)")
    ap.add_argument("--fcore-mhz", type=float, default=0.0,
                    help="core clock the run is timed at")
    args = ap.parse_args()
    set_layers(args.layers)
    hargs = []
    if args.ddr_gbps > 0:
        hargs += ["--ddr-gbps", str(args.ddr_gbps)]
    if args.fcore_mhz > 0:
        hargs += ["--fcore-mhz", str(args.fcore_mhz)]

    m = qr.Qwen3(args.model)
    m.L = m.L[:args.layers]
    ids = m.tok.encode(args.prompt)
    T = len(ids)
    npos = T + args.decode
    if npos > MAX_POS:
        sys.exit("%d positions do not fit the %d-row cache" % (npos, MAX_POS))
    cos, sin = m.cos_sin(npos)
    rep = m.nh // m.nkv
    scale = m.hd ** -0.5
    V = m.embed.shape[0]
    print("%d layers, %d prompt tokens + %d decoded, cache %d rows, vocab %d"
          % (args.layers, T, args.decode, npos, V))
    if args.ddr_gbps > 0:
        print("DDR %.2f GB/s at %.0f MHz = %.1f B/cycle (one 128 bit beat is "
              "16 B, so the port caps at %.2f GB/s)"
              % (args.ddr_gbps, args.fcore_mhz,
                 args.ddr_gbps * 1e9 / (args.fcore_mhz * 1e6),
                 16.0 * args.fcore_mhz * 1e6 / 1e9))

    os.makedirs(args.outdir, exist_ok=True)
    img = bytearray(IMG_SIZE)
    mats_by = {}
    for i in range(args.layers):
        use(LAYER_STRIDE * i)
        mats_by[i] = rl.stage_weights(img, i, m.L[i], args.outdir, args.repack)
        fill_layer(img, LAYER_STRIDE * i, m.L[i], cos, sin, npos)

    lm, recon_lm = stage_lm_head(img, m.embed, args.outdir, args.repack)
    img[FIN_EXT:FIN_EXT + 4 * HX] = m.final.astype(np.float32).tobytes()

    # the network as the hardware's weights define it: same forward, packed
    # matrices everywhere, so the RTL is compared to a 4-bit model, not fp32
    embed32 = recon_lm.astype(np.float32)
    for i in range(args.layers):
        for k, W in mats_by[i].items():
            m.L[i][k] = W.recon
    m.embed = embed32

    mem_bin = os.path.join(args.outdir, "mem.bin")
    script = os.path.join(args.outdir, "cmds.txt")
    kf_p = [os.path.join(args.outdir, "kf_L%d.bin" % i)
            for i in range(args.layers)]
    v16_p = [os.path.join(args.outdir, "v16_L%d.bin" % i)
             for i in range(args.layers)]
    out_p = os.path.join(args.outdir, "x.bin")
    hf_p = os.path.join(args.outdir, "hf.bin")
    log_p = os.path.join(args.outdir, "logits.bin")

    total = 0.0
    worst, bad = 1.0, 0
    for p in range(npos):
        tok = ids[p] if p < len(ids) else ids[-1]   # dry runs never decode
        img[_LAY["X"] + 4 * HX * p:_LAY["X"] + 4 * HX * (p + 1)] = \
            embed32[tok].tobytes()
        do_lm = p >= T - 1

        use(0)
        S = rl.Script()
        for i in range(args.layers):
            b = LAYER_STRIDE * i
            use(b)
            if i:                       # hand the previous x_out to this layer
                src = LAYER_STRIDE * (i - 1) + _LAY["OUT"]
                dst = b + _LAY["X"] + 4 * HX * p
                ld = S.add(lambda j, s, src=src: nc.load(
                    j, src, rl.B_X, row_bytes=4 * HX, slot=s))
                S.add(lambda j, s, dst=dst: nc.store(
                    j, dst, rl.B_X, 0, 4 * HX, slot=s, **rl.dep(ld)))
                S.close()
            rl.emit_layer(S, mats_by[i], 7, p + 1, p, HD, m.nh, m.nkv, HX,
                          HQ, HK, rep, scale, t0=p)

        if do_lm:
            b = LAYER_STRIDE * (args.layers - 1)
            if args.fin_upto >= 1:
                rl.norm_block(S, b + _LAY["OUT"], FIN_EXT, HF_EXT, HX)
                S.close()
            if args.fin_upto >= 2:
                emit_lm_head(S, lm, HF_EXT, LOG_EXT, args.lm_chunks or None)

        for i in range(args.layers):
            b = LAYER_STRIDE * i
            n = 2 * HK * (p + 1)
            S.lines.append("DUMP %x %x %s" % (b + _LAY["KF"], n, kf_p[i]))
            S.lines.append("DUMP %x %x %s" % (b + _LAY["V16"], n, v16_p[i]))
        S.lines.append("DUMP %x %x %s" %
                       (LAYER_STRIDE * (args.layers - 1) + _LAY["OUT"],
                        4 * HX, out_p))
        nlm = min(args.lm_chunks or lm.nchunk, lm.nchunk)
        if do_lm:
            S.lines.append("DUMP %x %x %s" % (HF_EXT, 2 * HX, hf_p))
            if args.fin_upto >= 2:
                S.lines.append("DUMP %x %x %s" %
                               (LOG_EXT, min(4 * LM_COLS * nlm, 4 * V), log_p))

        open(mem_bin, "wb").write(bytes(img))
        open(script, "w").write("\n".join(S.lines) + "\n")
        ncmd = sum(1 for l in S.lines if l.startswith("CMD"))
        if args.dry:
            print("pos %d: %d commands (dry)" % (p, ncmd))
            continue

        # the harness writes one CPL line per command: several hours of
        # simulation per position, so the log is streamed to a file where
        # the progress can be counted instead of held in memory
        hlog = os.path.join(args.outdir, "harness_p%d.log" % p)
        t0 = time.time()
        with open(hlog, "w") as fo:
            r = subprocess.run([os.path.abspath(args.bin), "--mem",
                                os.path.abspath(mem_bin), "--script",
                                os.path.abspath(script)] + hargs,
                               stdout=fo, stderr=subprocess.PIPE, text=True)
        dt = time.time() - t0
        out = open(hlog, errors="replace").read()
        if r.returncode != 0:
            print(out[-4000:])
            print(r.stderr.strip())
            sys.exit("harness failed at position %d rc=%d" % (p, r.returncode))
        cpl = [l for l in out.splitlines() if l.startswith("CPL ")]
        errs = [l for l in cpl if l.split()[4] != "0000"]
        done = [l for l in out.splitlines() if l.startswith("DONE")]
        cyc, rd_b = 0, 0
        if done:
            kv = dict(t.split("=", 1) for t in done[0].split()[1:])
            cyc = int(kv.get("cycles", 0))
            rd_b = int(kv.get("rd_bytes", 0))
        total += cyc
        if errs:
            print("%d of %d completions not OK at position %d, first 5:" %
                  (len(errs), len(cpl), p))
            for l in errs[:5]:
                print("  " + l)
            sys.exit(1)

        # carry the cache the NPU wrote into the next run's image
        for i in range(args.layers):
            b = LAYER_STRIDE * i
            n = 2 * HK * (p + 1)
            img[b + _LAY["KF"]:b + _LAY["KF"] + n] = open(kf_p[i], "rb").read()
            img[b + _LAY["V16"]:b + _LAY["V16"] + n] = open(v16_p[i], "rb").read()

        x_hw = np.fromfile(out_p, np.float32)
        tag = "prefill" if p < T - 1 else ("last prompt" if p == T - 1
                                           else "decode")
        line = "pos %2d %-11s %-12r %6d cmds %8d cyc %5.0f s" % (
            p, tag, m.tok.decode([tok]), ncmd, cyc, dt)
        if cyc:                       # what the memory path actually moved
            line += "  ddr %.2f B/cyc (%.0f MB)" % (rd_b / cyc, rd_b / 1e6)

        # the network, restarted on this prefix with the packed weights; the
        # hidden row is comparable at every position, the logits only where
        # the lm_head actually ran
        # forward only fills its dump dict for the layer it is asked about;
        # the last layer's x is the stack output the NPU leaves at OUT
        logits_ref, dref = m.forward(ids[:p + 1], dump=args.layers - 1)
        cx = cos_of(x_hw, dref["x"][-1])
        line += "  x_out cos %.7f" % cx[0]
        if do_lm and args.fin_upto >= 2:
            lr = logits_ref[-1]
            lg = np.fromfile(log_p, np.float32)
            lrc = lr[:len(lg)]          # a partial lm_head compares to its slice
            cl = cos_of(lg, lrc)
            a_hw, a_ref = int(np.argmax(lg)), int(np.argmax(lrc))
            worst = min(worst, cl[0])
            bad += a_hw != a_ref
            line += "  logits cos %.7f  argmax %d/%d %s" % (
                cl[0], a_hw, a_ref, "same" if a_hw == a_ref else "DIFF")
            ids.append(a_hw)
            line += "  -> %r" % m.tok.decode([ids[-1]])
        print(line)

    if not args.dry:
        print("total %d cycles" % total)
        if args.fin_upto >= 2 and bad == 0 and worst > 0.999:
            print("PASS")
        else:
            print("worst logits cos %.7f, %d argmax mismatches" % (worst, bad))


if __name__ == "__main__":
    main()

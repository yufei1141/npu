"""real_lm.py : the whole lm_head on the NPU, from a finished demo position.

The autoregression demo simulates only the first few lm_head chunks (a full
vocabulary is 2374 chunks, ~52 minutes of Verilator on its own).  This picks
up the final normed hidden row the demo left in hf.bin, replays the entire
151936-wide GEMV in the accelerator, and compares every logit with the same
3-layer network in Python.  Nothing else is re-simulated.

    python scripts/real_lm.py --prompt "The capital" --token 2022
"""
import argparse
import os
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qwen3_ref as qr
import real_layer as rl
import real_gen as rg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=qr.DEFAULT_MODEL)
    ap.add_argument("--prompt", default="The capital")
    ap.add_argument("--token", type=int, default=2022,
                    help="the token the demo's partial lm_head chose")
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--outdir", default="build/real_gen")
    ap.add_argument("--bin", default="build/vsim/qk/Vnpu_top.exe")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()

    m = qr.Qwen3(a.model)
    m.L = m.L[:a.layers]
    for i in range(a.layers):
        z = np.load(os.path.join(a.outdir, "pack_L%d.npz" % i))
        for k in ("wq", "wk", "wv", "wo", "wg", "wu", "wd"):
            m.L[i][k] = z["r_" + k]
    recon = np.load(os.path.join(a.outdir, "lm_head.npz"))["recon"]
    m.embed = recon.astype(np.float32)
    lm = rg.LMat(m.embed.shape[0], rg.HX, rg.LMW_EXT, None, recon)

    # the demo's image already holds the packed lm_head; only the hidden row
    # the last position produced has to be put back (the harness wrote it)
    mem = os.path.join(a.outdir, "mem.bin")
    img = bytearray(open(mem, "rb").read())
    hf = np.fromfile(os.path.join(a.outdir, "hf.bin"), np.float16)
    img[rg.HF_EXT:rg.HF_EXT + 2 * rg.HX] = hf.tobytes()
    # keep the demo's own image intact: this run gets a copy
    mem = os.path.join(a.outdir, "mem_lm.bin")

    S = rl.Script()
    rg.emit_lm_head(S, lm, rg.HF_EXT, rg.LOG_EXT, None)
    S.close()
    logits_p = os.path.join(a.outdir, "logits_full.bin")
    S.lines.append("DUMP %x %x %s" % (rg.LOG_EXT, 4 * m.embed.shape[0],
                                      os.path.abspath(logits_p)))
    ncmd = sum(1 for l in S.lines if l.startswith("CMD"))
    print("%d chunks, %d commands" % (lm.nchunk, ncmd))
    if a.dry:
        return

    open(mem, "wb").write(bytes(img))
    script = os.path.join(a.outdir, "cmds_lm.txt")
    open(script, "w").write("\n".join(S.lines) + "\n")

    t0 = time.time()
    r = subprocess.run([os.path.abspath(a.bin), "--mem", os.path.abspath(mem),
                        "--script", os.path.abspath(script)],
                       capture_output=True, text=True)
    dt = time.time() - t0
    if r.returncode != 0:
        print(r.stdout[-4000:])
        print(r.stderr.strip())
        sys.exit("harness failed rc=%d" % r.returncode)
    cpl = [l for l in r.stdout.splitlines() if l.startswith("CPL ")]
    errs = [l for l in cpl if l.split()[4] != "0000"]
    if errs:
        print("%d of %d completions not OK, first 5:" % (len(errs), len(cpl)))
        for l in errs[:5]:
            print("  " + l)
        sys.exit(1)
    done = [l for l in r.stdout.splitlines() if l.startswith("DONE")]
    cyc = int(done[0].split("=")[1]) if done else 0

    lg = np.fromfile(logits_p, np.float32)
    ids = m.tok.encode(a.prompt) + [a.token]
    lr, _ = m.forward(ids)
    lr = lr[-1]
    rr, n, c = rl.metrics(lg.astype(np.float64), lr.astype(np.float64))
    a_hw, a_ref = int(np.argmax(lg)), int(np.argmax(lr))
    print("%d logits  %d cycles  %.0f s" % (len(lg), cyc, dt))
    print("logits cos %.7f  rel %.3g  argmax %d/%d %s  -> %r" %
          (c, rr, a_hw, a_ref, "same" if a_hw == a_ref else "DIFF",
           m.tok.decode([a_hw])))
    print("PASS" if a_hw == a_ref and c > 0.999 else "FAIL")


if __name__ == "__main__":
    main()

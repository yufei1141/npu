"""real_gemv.py : run one projection of a real Qwen3-0.6B layer in the RTL.

The activations are a row of the genuine network (whatever row feeds the
chosen projection), the weights are the genuine AWQ weights repacked to the
hardware's int4 + E8M0 format, and the reference is what the real network
computes -- not a hardware model.  The comparison is therefore a tolerance
check that isolates the hardware's own requantization error; it is not a
bit-exact claim.

    python scripts/real_gemv.py --cols 32 --token -1
    python scripts/real_gemv.py --proj gate --layer 13 --cols 32
    python scripts/real_gemv.py --proj down --layer 0  --cols 16
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qwen3_ref as qr
import hw_pack
import npu_cmd as nc

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

A_EXT = 0x1000
W_EXT = 0x4000
D_EXT = 0x10000        # the scale area follows the weights: LOAD_W is one
                       # contiguous external read (dma_engine reads P4 bytes)

A_BUF, D_BUF = 1, 0

# projection -> (weight key in m.L[layer], activation key in the dump,
#                reference key in the dump)
PROJ = {
    "q":    ("wq", "h1",    "q"),
    "k":    ("wk", "h1",    "k"),
    "v":    ("wv", "h1",    "v"),
    "o":    ("wo", "o_in",  "o"),
    "gate": ("wg", "h2",    "g"),
    "up":   ("wu", "h2",    "u"),
    "down": ("wd", "mlp_a", "down"),
}
JOB_LOAD, JOB_W, JOB_FENCE, JOB_MM, JOB_ST = 1, 2, 3, 4, 5
SLOT_FENCE, SLOT_MM, SLOT_ST, SLOT_LOAD, SLOT_W = 1, 2, 3, 4, 5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=qr.DEFAULT_MODEL)
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--proj", default="q", choices=sorted(PROJ),
                    help="which projection of the block to run")
    ap.add_argument("--cols", type=int, default=32, help="output columns (4*n_grp)")
    ap.add_argument("--token", type=int, default=-1)
    ap.add_argument("--bin", default="build/vsim/npu_top/Vnpu_top.exe")
    ap.add_argument("--outdir", default="build/real")
    args = ap.parse_args()

    m = qr.Qwen3(args.model)
    ids = m.tok.encode(args.prompt)
    _, d = m.forward(ids, dump=args.layer)
    wkey, akey, rkey = PROJ[args.proj]
    a = d[akey][args.token].astype(np.float32)        # real activation row
    n_grp = args.cols // 4
    W = m.L[args.layer][wkey][:, :args.cols]
    ref = d[rkey][args.token, :args.cols]             # real network output

    wbytes, sbytes, recon = hw_pack.pack_tile(W.astype(np.float32), n_grp=n_grp)
    K = W.shape[0]
    S = (K + 15) // 16

    os.makedirs(args.outdir, exist_ok=True)
    img = bytearray(1 << 21)

    a16 = a.astype(np.float16)
    img[A_EXT:A_EXT + 2 * K] = a16.tobytes()
    s_ext = W_EXT + len(wbytes)
    img[W_EXT:W_EXT + len(wbytes)] = wbytes
    img[s_ext:s_ext + len(sbytes)] = sbytes
    open(os.path.join(args.outdir, "mem.bin"), "wb").write(bytes(img))

    nbytes = len(wbytes) + len(sbytes)
    cmds = [
        nc.load(JOB_LOAD, A_EXT, A_BUF, row_bytes=2 * K, rows=1,
                slot=SLOT_LOAD),
        nc.load_w(JOB_W, W_EXT, s_ext, JOB_MM, 0, 0, n_grp, S, nbytes,
                  slot=SLOT_W),
        nc.fence(JOB_FENCE, SLOT_FENCE),
        nc.matmul(JOB_MM, 1, n_grp, 0, 1, S, K, args.cols,
                  0, 2 * K, 0, 4 * args.cols, A_BUF, D_BUF, SLOT_MM,
                  flags=nc.FLAG_DEP, dep_job=JOB_FENCE, dep_slot=SLOT_FENCE),
        nc.store(JOB_ST, D_EXT, D_BUF, 0, 4 * args.cols, rows=1, slot=SLOT_ST,
                 flags=nc.FLAG_DEP, dep_job=JOB_MM, dep_slot=SLOT_MM),
    ]
    lines = ["CMD " + nc.hex128(c) for c in cmds]
    lines += ["WAIT %d" % j for j in (JOB_LOAD, JOB_W, JOB_FENCE, JOB_MM, JOB_ST)]
    dump = os.path.join(args.outdir, "d.bin")
    lines.append("DUMP %x %d %s" % (D_EXT, 4 * args.cols, dump))
    script = os.path.join(args.outdir, "cmds.txt")
    open(script, "w").write("\n".join(lines) + "\n")

    r = subprocess.run([os.path.abspath(args.bin), "--mem",
                        os.path.abspath(os.path.join(args.outdir, "mem.bin")),
                        "--script", os.path.abspath(script)],
                       capture_output=True, text=True)
    print(r.stdout.strip())
    if r.returncode != 0:
        print(r.stderr.strip())
        sys.exit("harness failed rc=%d" % r.returncode)

    got = np.frombuffer(open(dump, "rb").read(), dtype=np.float32)
    got = got[:args.cols]

    # what the packed weights alone would produce in exact fp32 arithmetic
    ideal = a16.astype(np.float32) @ recon
    print("\nreference (real model) top columns:")
    print("  ref :", np.array2string(ref[:6], precision=4))
    print("  rtl :", np.array2string(got[:6], precision=4))
    print("  pack:", np.array2string(ideal[:6], precision=4))
    e_rtl = got - ref
    e_pack = ideal - ref
    print("\nRMS ref %.4f" % np.sqrt(np.mean(ref ** 2)))
    print("RMS err rtl  %.4f (%.1f%% of RMS)"
          % (np.sqrt(np.mean(e_rtl ** 2)),
             100 * np.sqrt(np.mean(e_rtl ** 2)) / np.sqrt(np.mean(ref ** 2))))
    print("RMS err pack %.4f (%.1f%% of RMS)"
          % (np.sqrt(np.mean(e_pack ** 2)),
             100 * np.sqrt(np.mean(e_pack ** 2)) / np.sqrt(np.mean(ref ** 2))))
    print("cos(rtl, ref) %.6f" % (got @ ref / np.linalg.norm(got) / np.linalg.norm(ref)))


if __name__ == "__main__":
    main()

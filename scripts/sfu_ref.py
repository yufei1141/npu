"""sfu_ref.py : drive the Verilator sfu_top harness against a float64 reference.

The reference is plain IEEE arithmetic evaluated in double precision and
rounded once to float32 (numpy).  It is not a model of the hardware: the
datapath is compared against the function it is supposed to compute.

    python scripts/sfu_ref.py --n 2000            # every op
    python scripts/sfu_ref.py --n 500 --ops 0,8

Requires numpy and a built harness:
    bash scripts/verilate.sh sfu_top tb/vsim/sfu_top_main.cpp
"""
import argparse
import os
import subprocess
import sys
import numpy as np

OPS = {0: "EXP2", 1: "LOG2", 2: "RCP", 3: "RSQRT", 4: "SQRT",
       5: "SIN", 6: "COS", 7: "TANH", 8: "SINCOS"}

# status bits, sfu_slice contract
ST_OVF, ST_UNF, ST_INV, ST_DIV0 = 1, 2, 4, 8


def ref(op, x):
    with np.errstate(all="ignore"):
        if op == 0:
            v = np.power(2.0, x)
        elif op == 1:
            v = np.log2(x)
        elif op == 2:
            v = 1.0 / x
        elif op == 3:
            v = 1.0 / np.sqrt(x)
        elif op == 4:
            v = np.sqrt(x)
        elif op == 5 or op == 8:
            v = np.sin(x)
        elif op == 6:
            v = np.cos(x)
        elif op == 7:
            v = np.tanh(x)
        else:
            raise ValueError(op)
    return v


def gen_inputs(op, n, rng):
    if op in (0,):
        return rng.uniform(-30.0, 30.0, n)
    if op == 1:
        return rng.uniform(1e-30, 1e30, n)
    if op == 2:
        return rng.uniform(-1e6, 1e6, n)
    if op in (3, 4):
        return rng.uniform(1e-30, 1e30, n)
    if op in (5, 6, 8):
        return rng.uniform(-100.0, 100.0, n)
    return rng.uniform(-8.0, 8.0, n)          # TANH


def ulp_diff(a, b):
    """integer distance between two finite same-sign float32 bit patterns"""
    ia = np.asarray(a, dtype=np.uint32).astype(np.int64)
    ib = np.asarray(b, dtype=np.uint32).astype(np.int64)
    sa = (ia >> 31) & 1
    sb = (ib >> 31) & 1
    ia = np.where(ia & 0x80000000, 0x80000000 - (ia & 0x7FFFFFFF), ia)
    ib = np.where(ib & 0x80000000, 0x80000000 - (ib & 0x7FFFFFFF), ib)
    return np.abs(ia - ib), sa == sb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", default="build/vsim/sfu_top/Vsfu_top.exe")
    ap.add_argument("--n", type=int, default=1000, help="vectors per op")
    ap.add_argument("--ops", default="0,1,2,3,4,5,6,7,8")
    ap.add_argument("--mode", type=int, default=0, choices=(0, 1))
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    ops = [int(t) for t in args.ops.split(",")]

    lines, meta = [], []
    for op in ops:
        xs = gen_inputs(op, args.n, rng).astype(np.float32)
        for i in range(0, len(xs), 8):
            chunk = xs[i:i + 8]
            lanes = list(chunk) + [np.float32(1.0)] * (8 - len(chunk))
            words = " ".join("%08x" % int(np.float32(v).view(np.uint32))
                             for v in lanes)
            # core format encoding: 0 FP32, 1 BF16, 2 FP16
            lines.append("%d 0 0 %d %s" % (op, args.mode, words))
            meta.append((op, chunk))

    out = subprocess.run([os.path.abspath(args.bin)], input="\n".join(lines) + "\n",
                         capture_output=True, text=True)
    if out.returncode != 0:
        print(out.stderr)
        sys.exit("harness failed rc=%d" % out.returncode)

    beats = {}
    for ln in out.stdout.splitlines():
        p = ln.split()
        if p[0] != "R":
            continue
        beats[(int(p[1]), int(p[2]))] = ([int(w, 16) for w in p[3:11]],
                                         int(p[11], 16))

    worst, checks, fails = {}, 0, 0
    for r, (op, chunk) in enumerate(meta):
        n = len(chunk)
        got_words, st = beats[(r, 0)]
        got = np.array(got_words, dtype=np.uint32)
        want64 = ref(op, chunk.astype(np.float64))
        # a lane whose exact value is outside the normal fp32 range is not
        # compared; the status bits carry the information instead
        normal = (np.abs(want64) >= 1.17549435e-38) & (np.abs(want64) <= 3.40282346e38)
        ok = normal
        if np.any(~normal):
            if np.any(np.abs(want64) > 3.40282346e38) and not (st & ST_OVF):
                fails += 1
                print("FAIL %s: overflow not flagged" % OPS[op])
        for lane in range(n):
            checks += 1
            if not ok[lane]:
                continue
            want = np.float32(want64[lane])
            w_bits = int(want.view(np.uint32))
            g_bits = int(got[lane])
            d, same = ulp_diff(g_bits, w_bits)
            if not same:
                fails += 1
                print("FAIL %s lane%d x=%08x got=%08x want=%08x sign"
                      % (OPS[op], lane,
                         int(np.float32(chunk[lane]).view(np.uint32)),
                         g_bits, w_bits))
                continue
            if d > worst.get(op, 0):
                worst[op] = int(d)
            # budget from sfu/arch/microarch.md: 2 ulp for EXP2/LOG2, 4 otherwise
            budget = 2 if op in (0, 1) else 4
            if d > budget:
                fails += 1
                print("FAIL %s lane%d x=%08x got=%08x want=%08x ulp=%d"
                      % (OPS[op], lane,
                         int(np.float32(chunk[lane]).view(np.uint32)),
                         g_bits, w_bits, d))

    for op in ops:
        print("  %-6s worst %d ulp" % (OPS[op], worst.get(op, 0)))
    print("%d checks, %d failures -> %s"
          % (checks, fails, "PASS" if fails == 0 else "FAIL"))
    sys.exit(0 if fails == 0 else 1)


if __name__ == "__main__":
    main()

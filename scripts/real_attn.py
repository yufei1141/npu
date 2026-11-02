"""real_attn.py : one head's whole attention block of Qwen3-0.6B in RTL.

The two halves of attention were validated separately (real_qk.py, real_pv.py)
through the MPU's activation-B path.  This script closes the block by putting
the softmax between them on the VPU/SFU:

    sc = q @ K^T                        MATMUL (B frame = packed K^T)
    m  = max(sc)                        REDUCE.MAX
    e  = exp2((sc - m) * log2e)         VECTOR.SUB/BCAST, MUL_SCALAR, SFU.EXP2
    l  = sum(e)                         REDUCE.SUM
    p  = e * (1/l)  -> fp16             SFU.RCP, VECTOR.MUL/BCAST
    o  = p @ V                          MATMUL (B frame = packed V)

Nothing is computed in Python except the reference.  The reference is the
model's own attention for the same head (scripts/qwen3_ref.py), and two
stricter ones: the softmax probabilities the hardware produced versus the fp16
of the mathematical softmax, and the output versus the float64 dot of the
hardware's own fp16 p and V -- which isolates RTL error from fp16 rounding.

    python scripts/real_attn.py --layer 0 --head 0
"""
import argparse
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qwen3_ref as qr
import npu_cmd as nc
from real_pv import frame_ref

# scratchpad logical buffers 0..7.  A command may not read and write the same
# logical buffer (access_decode rejects it; only VECTOR INPLACE is exempt), so
# the softmax ping-pongs between buffers and later windows reuse the ones whose
# contents are dead by then.
B_Q, B_K, B_V, B_KT, B_KF, B_SC, B_P = range(7)
B_EX = B_Q          # exp2 output, once q is dead
B_SS = B_KT         # scalars (max, sum), once K^T is dead
B_SS2 = B_KF        # 1/sum, once the K frame is dead
B_VF = B_K          # V's w16 frame; src1_buf != 0 marks the activation-B
                    # path, so the frame can never live in buffer 0, and bK
                    # is free once the transpose has copied K into B_KT
B_O = B_KT          # P*V output

Q_EXT, K_EXT, V_EXT, P_EXT, O_EXT = 0x1000, 0x2000, 0x3000, 0x4000, 0x5000

LOG2E = 1.4426950408889634
F32, F16 = nc.F_FP32, nc.F_FP16


def f32(x):
    return int(np.float32(x).view(np.uint32))


def metrics(got, ref):
    e = got.astype(np.float64) - ref.astype(np.float64)
    r = np.sqrt(np.mean(e * e))
    rr = np.sqrt(np.mean(ref.astype(np.float64) ** 2)) + 1e-30
    g = got.astype(np.float64)
    c = (g @ ref.astype(np.float64)) / (
        np.linalg.norm(g) * np.linalg.norm(ref.astype(np.float64)) + 1e-30)
    return r, rr, c


def stat(name, got, ref):
    r, rr, c = metrics(got, ref)
    print("  %-5s RMS err %.4e (%.1f%% of RMS)  cos %.6f" %
          (name, r, 100.0 * r / rr, c))


class Script:
    """command stream with job ids, slots and per-window release (real_mlp)"""

    def __init__(self, job0=100):
        self.lines = []
        self.job = job0
        self.slot = -1
        self.win = []

    def njob(self):
        self.job += 1
        return self.job

    def nslot(self):
        self.slot += 1
        if self.slot >= 16:
            raise RuntimeError("window needs more than 16 event slots")
        return self.slot

    def put(self, c, j, s):
        self.lines.append("CMD " + nc.hex128(c))
        self.win.append((j, s))

    def add(self, build):
        j, s = self.njob(), self.nslot()
        self.put(build(j, s), j, s)
        return (j, s)

    def close(self):
        if not self.win:
            return
        self.lines.append("WAIT %d" % self.win[-1][0])
        for j, s in self.win:
            self.lines.append("CMD " + nc.hex128(nc.release(self.njob(), s, j)))
        self.win = []
        self.slot = -1


def dep(prev, flags=nc.FLAG_DEP):
    return dict(flags=flags, dep_job=prev[0], dep_slot=prev[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=qr.DEFAULT_MODEL)
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--head", type=int, default=0)
    ap.add_argument("--token", type=int, default=-1)
    ap.add_argument("--keys", type=int, default=16,
                    help="key positions in the sweep (<= 32)")
    ap.add_argument("--bin", default="build/vsim/qk/Vnpu_top.exe")
    ap.add_argument("--outdir", default="build/real_attn")
    args = ap.parse_args()

    m = qr.Qwen3(args.model)
    ids = m.tok.encode(args.prompt)
    T = len(ids)
    _, d = m.forward(ids, dump=args.layer)
    L = m.L[args.layer]
    tok = args.token if args.token >= 0 else T - 1
    S = min(args.keys, T)
    hd = m.hd
    rep = m.nh // m.nkv
    kvh = args.head // rep

    h1 = d["h1"].astype(np.float32)
    scale = hd ** -0.5
    cos, sin = m.cos_sin(T)
    q = (h1 @ L["wq"])[tok].reshape(m.nh, hd)[args.head]
    q = qr.rmsnorm(q, L["qn"].astype(np.float32), m.eps) * scale
    q = qr.rope(q[None, None, :], cos[tok:tok + 1, None, :],
                sin[tok:tok + 1, None, :])[0, 0]
    k = (h1 @ L["wk"]).reshape(T, m.nkv, hd)[:S, kvh]
    k = qr.rmsnorm(k, L["kn"].astype(np.float32), m.eps)
    k = qr.rope(k[:, None, :], cos[:S, None, :], sin[:S, None, :])[:, 0]
    v = (h1 @ L["wv"]).reshape(T, m.nkv, hd)[:S, kvh]

    q16 = q.astype(np.float16)
    k16 = k.astype(np.float16)
    v16 = v.astype(np.float16)

    # reference softmax: the mathematical one over the fp16 operands, and the
    # model's own fp32 one for the end-to-end comparison
    sc64 = q16.astype(np.float64) @ k16.astype(np.float64).T
    p_ref = np.exp(sc64 - sc64.max())
    p_ref = (p_ref / p_ref.sum()).astype(np.float16)
    q32 = q.astype(np.float64)
    sc_m = q32 @ k.astype(np.float64).T
    p_model = np.exp(sc_m - sc_m.max())
    p_model /= p_model.sum()
    o_model = p_model @ v.astype(np.float64)

    ngrp = (S + 3) // 4
    ksteps = hd // 4

    ngrp_v = 8
    nblk = (hd + 4 * ngrp_v - 1) // (4 * ngrp_v)   # PACK_W16 caps at 8 groups
    vsteps = (S + 3) // 4

    img = bytearray(1 << 20)
    img[Q_EXT:Q_EXT + 2 * hd] = q16.tobytes()
    img[K_EXT:K_EXT + 2 * S * hd] = k16.tobytes()
    img[V_EXT:V_EXT + 2 * S * hd] = v16.tobytes()
    os.makedirs(args.outdir, exist_ok=True)
    mem_bin = os.path.join(args.outdir, "mem.bin")
    open(mem_bin, "wb").write(bytes(img))

    S_ = Script()

    # ------------------------------------------------------------------
    # window 1: q @ K^T
    # ------------------------------------------------------------------
    S_.add(lambda j, s: nc.load(j, Q_EXT, B_Q, row_bytes=2 * hd, slot=s))
    S_.add(lambda j, s: nc.load(j, K_EXT, B_K, row_bytes=2 * hd, rows=S,
                                ext_stride=2 * hd, local_stride=2 * hd,
                                slot=s))
    S_.add(lambda j, s: nc.load(j, V_EXT, B_V, row_bytes=2 * hd, rows=S,
                                ext_stride=2 * hd, local_stride=2 * hd,
                                slot=s))
    f0 = S_.add(lambda j, s: nc.fence(j, s))
    tr = S_.add(lambda j, s: nc.transpose2d(j, S, hd, B_K, 0, B_KT, 0,
                                            2 * hd, 2 * S, s, **dep(f0)))
    pk = S_.add(lambda j, s: nc.pack_w16(j, hd, S, B_KT, 0, B_KF, 0,
                                         2 * S, ngrp, s, **dep(tr)))
    S_.add(lambda j, s: nc.matmul(j, 1, ngrp, 1, 1, ksteps, hd, S,
                                  0, 2 * hd, 0, 4 * S, B_Q, B_SC, s,
                                  a_fmt=F16, b_buf=B_KF, b_base=0,
                                  b_stride=32 * ksteps, b_fmt=F16,
                                  **dep(pk)))
    S_.close()

    # ------------------------------------------------------------------
    # window 2: softmax, fp16 probabilities into B_P
    # ------------------------------------------------------------------
    mx = S_.add(lambda j, s: nc.reduce(j, nc.R_MAX, S, B_SC, 0, B_SS, 0, s))
    sb = S_.add(lambda j, s: nc.vector(j, nc.V_SUB | nc.V_BCAST_S, S,
                                       B_SC, 0, B_SC, 0, s, s1=B_SS, o1=0,
                                       flags=nc.FLAG_INPLACE | nc.FLAG_DEP,
                                       dep_job=mx[0], dep_slot=mx[1]))
    ml = S_.add(lambda j, s: nc.vector(j, nc.V_MUL_SCALAR, S, B_SC, 0,
                                       B_SC, 0, s, scalar=f32(LOG2E),
                                       flags=nc.FLAG_INPLACE | nc.FLAG_DEP,
                                       dep_job=sb[0], dep_slot=sb[1]))
    ex = S_.add(lambda j, s: nc.sfu(j, nc.S_EXP2, S, B_SC, 0, B_EX, 0, s,
                                    **dep(ml)))
    sm = S_.add(lambda j, s: nc.reduce(j, nc.R_SUM, S, B_EX, 0, B_SS, 64, s,
                                       **dep(ex)))
    rc = S_.add(lambda j, s: nc.sfu(j, nc.S_RCP, 1, B_SS, 64, B_SS2, 0, s,
                                    **dep(sm)))
    pr = S_.add(lambda j, s: nc.vector(j, nc.V_MUL | nc.V_BCAST_S, S,
                                       B_EX, 0, B_P, 0, s, s1=B_SS2, o1=0,
                                       fd=F16, **dep(rc)))
    S_.add(lambda j, s: nc.store(j, P_EXT, B_P, 0, 32, slot=s, **dep(pr)))
    S_.close()

    # ------------------------------------------------------------------
    # window 3: p @ V
    # ------------------------------------------------------------------
    # the blocks write disjoint regions, but the fence can only collect one
    # job, so they are chained (real_qk does the same)
    prev = None
    for i in range(nblk):
        prev = S_.add(lambda j, s, i=i, p=prev: nc.pack_w16(
            j, S, 4 * ngrp_v, B_V, 4 * ngrp_v * 2 * i, B_VF,
            ngrp_v * vsteps * 32 * i, 2 * hd, ngrp_v, s,
            **(dep(p) if p else {})))
    f1 = S_.add(lambda j, s: nc.fence(j, s))
    mm = S_.njob()
    sl = S_.nslot()
    S_.put(nc.matmul(mm, 1, ngrp_v, 1, nblk, vsteps, S, hd,
                     0, 32, 0, 4 * hd, B_P, B_O, sl, a_fmt=F16,
                     b_buf=B_VF, b_base=0, b_stride=32 * vsteps,
                     b_fmt=F16, **dep(f1)), mm, sl)
    S_.add(lambda j, s: nc.store(j, O_EXT, B_O, 0, 4 * hd, slot=s,
                                 **dep((mm, sl))))
    S_.close()

    out_p = os.path.join(args.outdir, "p.bin")
    out_o = os.path.join(args.outdir, "o.bin")
    S_.lines.append("DUMP %x %x %s" % (P_EXT, 32, out_p))
    S_.lines.append("DUMP %x %x %s" % (O_EXT, 4 * hd, out_o))

    script = os.path.join(args.outdir, "cmds.txt")
    open(script, "w").write("\n".join(S_.lines) + "\n")
    print("%d commands, head %d at token %d, %d keys, k frame %d groups, "
          "v frame %d blocks" % (sum(1 for l in S_.lines
                                     if l.startswith("CMD")),
                                 args.head, tok, S, ngrp, nblk))

    r = subprocess.run([os.path.abspath(args.bin), "--mem",
                        os.path.abspath(mem_bin), "--script",
                        os.path.abspath(script)], capture_output=True,
                       text=True)
    out = r.stdout
    if r.returncode != 0:
        print(out[-3000:])
        print(r.stderr.strip())
        sys.exit("harness failed rc=%d" % r.returncode)
    cpl = [l for l in out.splitlines() if l.startswith("CPL ")]
    errs = [l for l in cpl if l.split()[4] != "0000"]
    done = [l for l in out.splitlines() if l.startswith("DONE")]
    print(done[0] if done else out.strip().splitlines()[-1])
    if errs:
        print("%d of %d completions are not OK:" % (len(errs), len(cpl)))
        for l in errs[:10]:
            print("  " + l)
        sys.exit(1)

    p_hw = np.frombuffer(open(out_p, "rb").read(), np.float16)[:S]
    o_hw = np.frombuffer(open(out_o, "rb").read(), np.float32)[:hd]

    print("\nsoftmax")
    neq = int(np.count_nonzero(p_hw.view(np.uint16) != p_ref.view(np.uint16)))
    print("  %d of %d probabilities differ from fp16(softmax)" % (neq, S))
    print("  hw  %s" % np.array2string(p_hw.astype(np.float32), precision=5))
    print("  ref %s" % np.array2string(p_ref.astype(np.float32), precision=5))
    stat("prob", p_hw, p_ref)
    stat("p/mdl", p_hw, p_model)

    print("\nP*V against the float64 dot of the hardware's own fp16 p and V")
    o_exact = p_hw.astype(np.float64) @ v16.astype(np.float64)
    stat("pv", o_hw, o_exact)

    print("\nattention out vs the model's own fp32 attention")
    stat("attn", o_hw, o_model)
    print("  rtl  %s" % np.array2string(o_hw[:6], precision=5))
    print("  mdl  %s" % np.array2string(o_model[:6].astype(np.float32),
                                         precision=5))


if __name__ == "__main__":
    main()

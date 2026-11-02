"""real_qk.py : QK^T of one real Qwen3-0.6B head through the activation-B path.

This is the first user of the MPU's second B source.  The B operand of a
MATMUL is normally an int4/fp16 weight tile in a weight bank; when src1_buf is
non-zero it is instead a w16 frame the B frame reader streams out of a
scratchpad buffer.  Attention needs exactly that -- QK^T has K on the B side
and P*V has V -- so this test drives the whole intended sequence:

    LOAD q, K   fp16 from external memory
    TRANSPOSE2D K (T x hd) -> K^T (hd x T)          VPU
    PACK_W16    K^T -> B frame (group j, step s at j*S*32 + s*32)
    MATMUL      m_len=1, w16A16, src1_buf = the frame buffer
    STORE       the scores

The reference is the real checkpoint's own attention math (scripts/qwen3_ref
rmsnorm / rope), rounded to fp16 exactly as the hardware operands are; the
comparison is against a float64 dot product of those same fp16 values, so what
remains is the RTL.

    python scripts/real_qk.py --layer 0 --head 0
"""
import argparse
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qwen3_ref as qr
import npu_cmd as nc

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

B_K, B_KT, B_BF, B_Q, B_D = 0, 1, 2, 3, 4

K_EXT, Q_EXT, O_EXT, FR_EXT = 0x1000, 0x2000, 0x3000, 0x4000


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


def frame_ref(kt, g, n_valid, k_valid):
    """what LAYOUT.PACK_W16 writes: 32*(j*S+s) + 8*p + 2*i <- K^T[4s+i][4j+p]"""
    S = (k_valid + 3) // 4
    buf = np.zeros(g * S * 32, np.uint8)
    u16 = buf.view(np.uint16)
    for j in range(g):
        for s in range(S):
            base = 32 * (j * S + s)
            for p in range(4):
                for i in range(4):
                    kk, nn = 4 * s + i, 4 * j + p
                    val = np.float16(0.0)
                    if kk < k_valid and nn < n_valid:
                        val = np.float16(kt[kk, nn])
                    u16[(base + 8 * p + 2 * i) // 2] = \
                        np.array([val], np.float16).view(np.uint16)[0]
    return buf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=qr.DEFAULT_MODEL)
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--head", type=int, default=0)
    ap.add_argument("--token", type=int, default=-1)
    ap.add_argument("--keys", type=int, default=16,
                    help="key positions in the sweep (<= 32)")
    ap.add_argument("--bin", default="build/vsim/npu_top/Vnpu_top.exe")
    ap.add_argument("--outdir", default="build/real_qk")
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

    # the checkpoint's own q / k for this head, exactly as forward() builds them
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

    # the hardware operands are fp16, so that is the reference's input too
    q16 = q.astype(np.float16)
    k16 = k.astype(np.float16)
    ref = (q16.astype(np.float64) @ k16.astype(np.float64).T)

    kt = k16.astype(np.float32).T                       # [hd][S]
    ngrp = (S + 3) // 4
    ksteps = hd // 4
    fr = frame_ref(kt.astype(np.float16), ngrp, S, hd)

    # ------------------------------------------------------------------
    # memory image
    # ------------------------------------------------------------------
    img = bytearray(1 << 20)
    img[K_EXT:K_EXT + 2 * S * hd] = k16.tobytes()
    img[Q_EXT:Q_EXT + 2 * hd] = q16.tobytes()
    os.makedirs(args.outdir, exist_ok=True)
    mem_bin = os.path.join(args.outdir, "mem.bin")
    open(mem_bin, "wb").write(bytes(img))

    # ------------------------------------------------------------------
    # command stream
    # ------------------------------------------------------------------
    jid, slt = [100], [0]

    def njob():
        jid[0] += 1
        return jid[0]

    def nslot():
        slt[0] += 1
        return slt[0]

    lines, win = [], []

    def put(c, j, s):
        lines.append("CMD " + nc.hex128(c))
        win.append((j, s))

    def dep(prev):
        return dict(flags=nc.FLAG_DEP, dep_job=prev[0], dep_slot=prev[1])

    def add(build):
        j, s = njob(), nslot()
        put(build(j, s), j, s)
        return (j, s)

    add(lambda j, s: nc.load(j, K_EXT, B_K, row_bytes=2 * hd, rows=S,
                             ext_stride=2 * hd, local_stride=2 * hd, slot=s))
    add(lambda j, s: nc.load(j, Q_EXT, B_Q, row_bytes=2 * hd, slot=s))
    f0 = add(lambda j, s: nc.fence(j, s))

    tr = add(lambda j, s: nc.transpose2d(j, S, hd, B_K, 0, B_KT, 0,
                                         2 * hd, 2 * S, s, **dep(f0)))
    pk = add(lambda j, s: nc.pack_w16(j, hd, S, B_KT, 0, B_BF, 0,
                                      2 * S, ngrp, s, **dep(tr)))
    f1 = add(lambda j, s: nc.fence(j, s))

    mm = njob()
    sl = nslot()
    put(nc.matmul(mm, 1, ngrp, 1, 1, ksteps, hd, S,
                  0, 2 * hd, 0, 4 * S, B_Q, B_D, sl, a_fmt=nc.F_FP16,
                  b_buf=B_BF, b_base=0, b_stride=32 * ksteps,
                  b_fmt=nc.F_FP16, **dep(f1)), mm, sl)

    st_o = add(lambda j, s: nc.store(j, O_EXT, B_D, 0, 4 * S, slot=s,
                                     **dep((mm, sl))))
    st_f = add(lambda j, s: nc.store(j, FR_EXT, B_BF, 0, len(fr), slot=s,
                                     **dep((mm, sl))))

    # the DUMPs read the harness memory image, so they must wait for the
    # stores that fill it -- a WAIT on the MATMUL alone returns while the
    # frame copy is still in flight
    lines.append("WAIT %d" % st_o[0])
    lines.append("WAIT %d" % st_f[0])
    for j, s in win:
        lines.append("CMD " + nc.hex128(nc.release(njob(), s, j)))

    out_o = os.path.join(args.outdir, "o.bin")
    out_f = os.path.join(args.outdir, "frame.bin")
    lines.append("DUMP %x %x %s" % (O_EXT, 4 * S, out_o))
    lines.append("DUMP %x %x %s" % (FR_EXT, len(fr), out_f))

    script = os.path.join(args.outdir, "cmds.txt")
    open(script, "w").write("\n".join(lines) + "\n")
    print("%d commands, q head %d at token %d, %d keys, %d groups" %
          (sum(1 for l in lines if l.startswith("CMD")), args.head, tok, S,
           ngrp))

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

    got = np.frombuffer(open(out_o, "rb").read(), np.float32)[:S]
    frr = np.frombuffer(open(out_f, "rb").read(), np.uint8)[:len(fr)]

    print("\nB frame vs LAYOUT.PACK_W16's definition")
    bad = int(np.count_nonzero(frr != fr))
    print("  %d of %d bytes differ" % (bad, len(fr)))

    print("\nQK^T vs the float64 dot of the same fp16 q and K")
    stat("qk", got, ref)
    print("  rtl  %s" % np.array2string(got[:8], precision=4))
    print("  ref  %s" % np.array2string(ref[:8].astype(np.float32),
                                        precision=4))


if __name__ == "__main__":
    main()

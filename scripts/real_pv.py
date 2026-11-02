"""real_pv.py : P*V of one real Qwen3-0.6B head through the activation-B path.

QK^T (scripts/real_qk.py) produces one row of scores; softmax turns it into
one row of probabilities P.  The second half of attention is then a plain
matmul with P as the A operand and V as the B operand, and V is an activation
exactly like K: it goes through LAYOUT.PACK_W16 into a scratchpad frame and
is streamed by the MPU's B frame reader.  No weight bank is involved, so the
same MATMUL that serves every quantized projection serves attention too.

The softmax is done in Python for now -- the SFU can do exp, but the row
reduction and the scaling have not been wired into a kernel yet, and this test
is about the P*V datapath, not about that kernel.

    python scripts/real_pv.py --layer 0 --head 0

One query row, so m_len = 1.  V is [S][hd]; the B operand of a w16 MATMUL is
a PACK_W16 frame over (k = keys, n = head dim), which is exactly V's own
orientation -- no transpose, unlike K in QK^T.

PACK_W16 caps a single command at 8 column groups (32 columns), and the head
dim is 128 columns, so the frame is built by four PACK_W16s placed back to
back; the MATMUL then sees one flat 32-group frame with tiles = 4 and
n_grp = 8, tile t taking groups t*8 .. t*8+7.
"""
import argparse
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qwen3_ref as qr
import npu_cmd as nc

B_V, B_P, B_F, B_O = 0, 1, 2, 3

V_EXT, P_EXT, O_EXT, FR_EXT = 0x1000, 0x2000, 0x3000, 0x4000


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


def frame_ref(v, g, n_cols, k_rows, ksteps):
    """LAYOUT.PACK_W16 blocks laid end to end: 32*(j*s_steps+s)+8p+2i <- v[4s+i][4j+p]

    Block m holds groups 8m..8m+7 and starts at m*8*s_steps*32 bytes, which is
    where the MATMUL's flat frame (group stride s_steps*32) expects it.
    """
    S = (k_rows + 3) // 4
    assert S == ksteps
    buf = np.zeros(g * S * 32, np.uint8)
    u16 = buf.view(np.uint16)
    for j in range(g):
        for s in range(S):
            base = 32 * (j * S + s)
            for p in range(4):
                for i in range(4):
                    kk, nn = 4 * s + i, 4 * j + p
                    val = np.float16(0.0)
                    if kk < k_rows and nn < n_cols:
                        val = np.float16(v[kk, nn])
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
    ap.add_argument("--outdir", default="build/real_pv")
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

    # q and k exactly as forward() builds them, then the row softmax
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

    # the hardware operands are fp16; the probabilities the kernel will hand
    # to P*V are fp16 too, so the reference starts from those same values
    q16 = q.astype(np.float16)
    k16 = k.astype(np.float16)
    sc = q16.astype(np.float64) @ k16.astype(np.float64).T
    p16 = np.exp(sc - sc.max()).astype(np.float16)
    p16 = (p16 / np.float16(p16.astype(np.float64).sum()))
    p16 = p16.astype(np.float16)
    v16 = v.astype(np.float16)

    ref = p16.astype(np.float64) @ v16.astype(np.float64)

    ngrp = 8                                  # 8 column groups = 32 columns
    nblk = (hd + 4 * ngrp - 1) // (4 * ngrp)  # PACK_W16 commands per frame
    tiles = nblk
    ksteps = (S + 3) // 4
    # the blocks are one buffer, each holding columns 32*m .. 32*m+31
    fr = np.concatenate([frame_ref(v16[:, 4 * ngrp * i:4 * ngrp * (i + 1)],
                                   ngrp, 4 * ngrp, S, ksteps)
                         for i in range(nblk)])

    # ------------------------------------------------------------------
    # memory image
    # ------------------------------------------------------------------
    img = bytearray(1 << 20)
    img[V_EXT:V_EXT + 2 * S * hd] = v16.tobytes()
    img[P_EXT:P_EXT + 2 * S] = p16.tobytes()
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

    # P is 2*S bytes; the A packer reads whole 32 B words, so the row is
    # padded to one scratchpad word
    add(lambda j, s: nc.load(j, V_EXT, B_V, row_bytes=2 * hd, rows=S,
                             ext_stride=2 * hd, local_stride=2 * hd, slot=s))
    add(lambda j, s: nc.load(j, P_EXT, B_P, row_bytes=32, slot=s))
    f0 = add(lambda j, s: nc.fence(j, s))

    prev = f0
    for i in range(nblk):
        prev = add(lambda j, s, i=i: nc.pack_w16(
            j, S, 4 * ngrp, B_V, 4 * ngrp * 2 * i, B_F,
            ngrp * ksteps * 32 * i, 2 * hd, ngrp, s, **dep(prev)))
    f1 = add(lambda j, s: nc.fence(j, s))

    mm = njob()
    sl = nslot()
    put(nc.matmul(mm, 1, ngrp, 1, tiles, ksteps, S, hd,
                  0, 32, 0, 4 * hd, B_P, B_O, sl, a_fmt=nc.F_FP16,
                  b_buf=B_F, b_base=0, b_stride=32 * ksteps,
                  b_fmt=nc.F_FP16, **dep(f1)), mm, sl)

    st_o = add(lambda j, s: nc.store(j, O_EXT, B_O, 0, 4 * hd, slot=s,
                                     **dep((mm, sl))))
    st_f = add(lambda j, s: nc.store(j, FR_EXT, B_F, 0, len(fr), slot=s,
                                     **dep((mm, sl))))

    # see real_qk.py: a WAIT on the MATMUL returns before the frame store
    lines.append("WAIT %d" % st_o[0])
    lines.append("WAIT %d" % st_f[0])
    for j, s in win:
        lines.append("CMD " + nc.hex128(nc.release(njob(), s, j)))

    out_o = os.path.join(args.outdir, "o.bin")
    out_f = os.path.join(args.outdir, "frame.bin")
    lines.append("DUMP %x %x %s" % (O_EXT, 4 * hd, out_o))
    lines.append("DUMP %x %x %s" % (FR_EXT, len(fr), out_f))

    script = os.path.join(args.outdir, "cmds.txt")
    open(script, "w").write("\n".join(lines) + "\n")
    print("%d commands, head %d at token %d, %d keys, %d blocks of %d groups" %
          (sum(1 for l in lines if l.startswith("CMD")), args.head, tok, S,
           nblk, ngrp))

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

    got = np.frombuffer(open(out_o, "rb").read(), np.float32)[:hd]
    frr = np.frombuffer(open(out_f, "rb").read(), np.uint8)[:len(fr)]

    print("\nB frame vs LAYOUT.PACK_W16's definition")
    bad = int(np.count_nonzero(frr != fr))
    print("  %d of %d bytes differ" % (bad, len(fr)))

    print("\nP*V vs the float64 dot of the same fp16 P and V")
    stat("pv", got, ref)
    print("  rtl  %s" % np.array2string(got[:8], precision=5))
    print("  ref  %s" % np.array2string(ref[:8].astype(np.float32),
                                        precision=5))


if __name__ == "__main__":
    main()

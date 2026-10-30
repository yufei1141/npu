"""npu_cmd.py : build 512-bit command words for npu_top (docs 02).

    c[7:0] version, [15:8] opcode, [23:16] subop, [31:24] flags,
    [47:32] job_id, [63:48] dep_job_id, [71:64] src0_buf, [79:72] src1_buf,
    [87:80] dst_buf, [95:88] formats, [103:96] event_slot,
    [111:104] dep_slot, [127:112] reserved, P0..P11 from bit 128.
"""
FLAG_DEP = 0x01
FLAG_INPLACE = 0x02
FLAG_IRQ = 0x04

OP_LOAD, OP_STORE, OP_MATMUL = 0x01, 0x02, 0x10
OP_VECTOR, OP_REDUCE, OP_SFU, OP_LAYOUT = 0x20, 0x21, 0x22, 0x23
OP_FENCE, OP_RELEASE = 0x30, 0x31

F_FP16, F_BF16, F_FP32, F_INT4 = 0, 1, 2, 3


def cmd(opcode, job, subop=0, flags=0, dep_job=0, src0=0, src1=0, dst=0,
        fmt=0, slot=0, dep_slot=0, p=None):
    c = 0
    c |= 0x01
    c |= opcode << 8
    c |= (subop & 0xFF) << 16
    c |= (flags & 0xFF) << 24
    c |= (job & 0xFFFF) << 32
    c |= (dep_job & 0xFFFF) << 48
    c |= (src0 & 0xFF) << 64
    c |= (src1 & 0xFF) << 72
    c |= (dst & 0xFF) << 80
    c |= (fmt & 0xFF) << 88
    c |= (slot & 0xFF) << 96
    c |= (dep_slot & 0xFF) << 104
    for i, v in enumerate((p or []) + [0] * (12 - len(p or []))):
        c |= (v & 0xFFFFFFFF) << (128 + 32 * i)
    return c


def formats(src0, src1=0, dst=0, acc=0):
    return src0 | (src1 << 2) | (dst << 4) | (acc << 6)


def hex128(c):
    return "%0128x" % c


def load(job, ext, buf, off=0, row_bytes=0, rows=1, ext_stride=0,
         local_stride=0, slot=0, flags=0, dep_job=0, dep_slot=0, subop=0):
    return cmd(OP_LOAD, job, subop=subop, flags=flags, dep_job=dep_job,
               dst=buf, slot=slot, dep_slot=dep_slot,
               p=[ext & 0xFFFFFFFF, ext >> 32, off, row_bytes, rows,
                  ext_stride, local_stride])


def load_w(job, waddr, saddr, consumer, tile, mode, n_grp, k_steps, nbytes,
           slot=0, flags=0, dep_job=0, dep_slot=0):
    return cmd(OP_LOAD, job, subop=1, flags=flags, dep_job=dep_job,
               slot=slot, dep_slot=dep_slot,
               p=[waddr & 0xFFFFFFFF, saddr & 0xFFFFFFFF,
                  (tile << 16) | consumer,
                  (k_steps << 16) | (n_grp << 8) | mode, nbytes])


def matmul(job, m_len, n_grp, mode, tiles, k_steps, k_valid, n_valid,
           a_base, a_stride, d_base, d_stride, a_buf, d_buf, slot,
           flags=0, dep_job=0, dep_slot=0, a_fmt=F_FP16):
    return cmd(OP_MATMUL, job, flags=flags, dep_job=dep_job,
               src0=a_buf, dst=d_buf,
               fmt=formats(a_fmt, F_INT4, F_FP32, F_FP32),
               slot=slot, dep_slot=dep_slot,
               p=[m_len | (n_grp << 8) | (mode << 16),
                  tiles | (k_steps << 16), k_valid, n_valid,
                  a_base, a_stride, 0, d_base, d_stride])


def store(job, ext, buf, off, row_bytes, rows=1, ext_stride=0, local_stride=0,
          slot=0, flags=0, dep_job=0, dep_slot=0):
    return cmd(OP_STORE, job, flags=flags, dep_job=dep_job, src0=buf,
               slot=slot, dep_slot=dep_slot,
               p=[ext & 0xFFFFFFFF, ext >> 32, off, row_bytes, rows,
                  ext_stride, local_stride])


def fence(job, slot):
    return cmd(OP_FENCE, job, slot=slot)


def release(job, slot, target_job):
    return cmd(OP_RELEASE, job, slot=0xFF,
               p=[(target_job << 8) | slot])

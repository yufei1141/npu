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

# VECTOR subop[5:0]; bit 7 = scalar broadcast from src1[0], bit 6 = pair
# broadcast from src1[idx/2]
(V_ADD, V_SUB, V_MUL, V_MAX, V_SELECT, V_CONVERT, V_MUL_SCALAR,
 V_ADD_SCALAR, V_COMPARE_GT, V_ROTATE_PAIR) = range(10)
V_BCAST_S, V_BCAST_P = 0x80, 0x40

# REDUCE subop: bit 0 selects max, bit 7 enables the mask buffer
R_SUM, R_MAX, R_MAX_MASK = 0x00, 0x01, 0x81

# SFU subop
(S_EXP2, S_RCP, S_RSQRT, S_SINCOS, S_LOG2, S_SQRT, S_SIN, S_COS,
 S_TANH) = range(9)


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
           flags=0, dep_job=0, dep_slot=0, a_fmt=F_FP16,
           b_buf=0, b_base=0, b_stride=0, b_fmt=None):
    """b_buf != 0: B is a w16 activation frame in a scratchpad buffer
    (P6 base, P9 group stride) instead of a weight tile -- mode must be w16.
    The frame is what LAYOUT.PACK_W16 writes: group j, step s at
    b_base + j*b_stride + 32*s."""
    bf = F_INT4 if b_fmt is None else b_fmt
    return cmd(OP_MATMUL, job, flags=flags, dep_job=dep_job,
               src0=a_buf, src1=b_buf, dst=d_buf,
               fmt=formats(a_fmt, bf, F_FP32, F_FP32),
               slot=slot, dep_slot=dep_slot,
               p=[m_len | (n_grp << 8) | (mode << 16),
                  tiles | (k_steps << 16), k_valid, n_valid,
                  a_base, a_stride, b_base, d_base, d_stride, b_stride])


# LAYOUT subops
(L_TRANSPOSE, L_PACK_W16) = (0, 1)


def transpose2d(job, rows, cols, s0, o0, d, od, sstride, dstride, slot,
                fs=F_FP16, flags=0, dep_job=0, dep_slot=0):
    """P0 rows, P1 cols, P2 src off, P3 dst off, P4 src stride, P5 dst stride
    (byte strides); dst[c*dstride + r*E] = src[r*sstride + c*E]."""
    return cmd(OP_LAYOUT, job, subop=L_TRANSPOSE, flags=flags,
               dep_job=dep_job, dep_slot=dep_slot, src0=s0, dst=d,
               fmt=formats(fs, 0, fs, F_FP16), slot=slot,
               p=[rows, cols, o0, od, sstride, dstride])


def pack_w16(job, kk, nn, s0, o0, d, od, sstride, g, slot, fs=F_FP16,
             flags=0, dep_job=0, dep_slot=0):
    """P0 K (source rows = reduction), P1 N (source cols = output columns),
    P2 src off, P3 dst off, P4 src stride (bytes), P5 g column groups.
    dst = 32*(j*S+s) + 8*p + 2*i with k = 4s+i, n = 4j+p, S = ceil(K/4)."""
    return cmd(OP_LAYOUT, job, subop=L_PACK_W16, flags=flags,
               dep_job=dep_job, dep_slot=dep_slot, src0=s0, dst=d,
               fmt=formats(fs, 0, fs, F_FP16), slot=slot,
               p=[kk, nn, o0, od, sstride, g])


def store(job, ext, buf, off, row_bytes, rows=1, ext_stride=0, local_stride=0,
          slot=0, flags=0, dep_job=0, dep_slot=0):
    return cmd(OP_STORE, job, flags=flags, dep_job=dep_job, src0=buf,
               slot=slot, dep_slot=dep_slot,
               p=[ext & 0xFFFFFFFF, ext >> 32, off, row_bytes, rows,
                  ext_stride, local_stride])


def vector(job, sub, cnt, s0, o0, d, od, slot, s1=0, o1=0, scalar=0,
           fs0=F_FP32, fs1=None, fd=F_FP32, flags=0, dep_job=0, dep_slot=0,
           buf_m=0, off_m=0):
    """VECTOR: P0 count, P1 src0 off, P2 src1 off, P3 dst off, P4 scalar,
    P5/P6 mask buffer.  MUL_SCALAR/ADD_SCALAR/CONVERT take the scalar in P4
    and must leave src1 and the mask at zero, which the front end proves by
    requiring the (unused) src1 format to be FP16."""
    if fs1 is None:
        fs1 = F_FP16 if (sub & 0x3F) in (V_CONVERT, V_MUL_SCALAR,
                                         V_ADD_SCALAR, V_ROTATE_PAIR) \
              else F_FP32
    return cmd(OP_VECTOR, job, subop=sub, flags=flags, dep_job=dep_job,
               dep_slot=dep_slot, src0=s0, src1=s1, dst=d,
               fmt=formats(fs0, fs1, fd, F_FP32), slot=slot,
               p=[cnt, o0, o1, od, scalar, buf_m, off_m])


def reduce(job, sub, cnt, s0, o0, d, od, slot, fs0=F_FP32, buf_m=0, off_m=0,
           flags=0, dep_job=0, dep_slot=0):
    """REDUCE: one FP32 scalar into dst; the destination format is fixed."""
    return cmd(OP_REDUCE, job, subop=sub, flags=flags, dep_job=dep_job,
               dep_slot=dep_slot, src0=s0, dst=d,
               fmt=formats(fs0, 0, F_FP32, F_FP32), slot=slot,
               p=[cnt, o0, 0, od, 0, buf_m, off_m])


def sfu(job, sub, cnt, s0, o0, d, od, slot, coff=0, mode=0,
        fs=F_FP32, fd=F_FP32, flags=0, dep_job=0, dep_slot=0):
    """SFU: P0 count, P1 src off, P2 cos off (SINCOS only), P3 dst off,
    P4[1:0] mode (0 fast, 1 accurate); P5..P11 must be zero."""
    return cmd(OP_SFU, job, subop=sub, flags=flags, dep_job=dep_job,
               dep_slot=dep_slot, src0=s0, dst=d,
               fmt=formats(fs, 0, fd), slot=slot,
               p=[cnt, o0, coff, od, mode])


def fence(job, slot):
    return cmd(OP_FENCE, job, slot=slot)


def release(job, slot, target_job):
    return cmd(OP_RELEASE, job, slot=0xFF,
               p=[(target_job << 8) | slot])

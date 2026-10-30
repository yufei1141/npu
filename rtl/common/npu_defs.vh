//=============================================================================
// npu_defs.vh : shared command / event / error encodings
//
//   This header is the single source of field positions for the 512-bit
//   command word, the 128-bit completion/event word and the status codes.
//   It follows docs/npu_integration/01_common_protocol.md and 02_instruction_set.md
//   (v0.1 proposal; not yet a frozen software ABI).
//
//   Verilog-2001 only: macros and a few helper functions, no packages.
//
//   Field macros are written as  (x[hi:lo])  rather than  ((x)[hi:lo]):
//   Verilog-2001 forbids a bit-select on a parenthesized expression, so the
//   argument must be a simple identifier or a named wire, not an expression.
//=============================================================================
`ifndef NPU_DEFS_VH
`define NPU_DEFS_VH

// ---------------------------------------------------------------------------
// Command header (512 bit), little-endian field numbering
// ---------------------------------------------------------------------------
`define NPU_CMD_VERSION(c)   (c[7:0])
`define NPU_CMD_OPCODE(c)    (c[15:8])
`define NPU_CMD_SUBOP(c)     (c[23:16])
`define NPU_CMD_FLAGS(c)     (c[31:24])
`define NPU_CMD_JOB(c)       (c[47:32])
`define NPU_CMD_DEPJOB(c)    (c[63:48])
`define NPU_CMD_SRC0BUF(c)   (c[71:64])
`define NPU_CMD_SRC1BUF(c)   (c[79:72])
`define NPU_CMD_DSTBUF(c)    (c[87:80])
`define NPU_CMD_FORMATS(c)   (c[95:88])
`define NPU_CMD_SLOT(c)      (c[103:96])
`define NPU_CMD_DEPSLOT(c)   (c[111:104])
`define NPU_CMD_RSVD(c)      (c[127:112])

// payload word i at 128 + 32*i
`define NPU_CMD_P(c,i)       ((c)[128+32*(i) +: 32])

// flags
`define NPU_FLAG_DEP_EN        0
`define NPU_FLAG_INPLACE       1
`define NPU_FLAG_IRQ           2

// ---------------------------------------------------------------------------
// Opcodes
// ---------------------------------------------------------------------------
`define NPU_OP_LOAD           8'h01
`define NPU_OP_STORE          8'h02
`define NPU_OP_MATMUL         8'h10
`define NPU_OP_VECTOR         8'h20
`define NPU_OP_REDUCE         8'h21
`define NPU_OP_SFU            8'h22
`define NPU_OP_LAYOUT         8'h23
`define NPU_OP_FENCE          8'h30
`define NPU_OP_RELEASE        8'h31

// ---------------------------------------------------------------------------
// Format codes: formats[1:0] src0, [3:2] src1, [5:4] dst, [7:6] acc
// ---------------------------------------------------------------------------
`define NPU_FMT_FP16          2'd0
`define NPU_FMT_BF16          2'd1
`define NPU_FMT_FP32          2'd2
`define NPU_FMT_INT4          2'd3

// VECTOR subop (low 6 bits)
`define NPU_VEC_ADD           6'd0
`define NPU_VEC_SUB           6'd1
`define NPU_VEC_MUL           6'd2
`define NPU_VEC_MAX           6'd3
`define NPU_VEC_SELECT        6'd4
`define NPU_VEC_CONVERT       6'd5
`define NPU_VEC_MUL_SCALAR    6'd6
`define NPU_VEC_ADD_SCALAR    6'd7
`define NPU_VEC_COMPARE_GT    6'd8
`define NPU_VEC_ROTATE_PAIR   6'd9

// LAYOUT subop
`define NPU_LAYOUT_TRANSPOSE  8'd0
`define NPU_LAYOUT_PACK_W16   8'd1

// ---------------------------------------------------------------------------
// Event / completion word (128 bit)
// ---------------------------------------------------------------------------
`define NPU_EV_JOB(e)        (e[15:0])
`define NPU_EV_SLOT(e)       (e[23:16])
`define NPU_EV_OPCODE(e)     (e[31:24])
`define NPU_EV_STATUS(e)     (e[47:32])
`define NPU_EV_DETAIL(e)     (e[63:48])
`define NPU_EV_UNITS(e)      (e[95:64])
`define NPU_EV_AUX(e)        (e[127:96])

`define NPU_SLOT_NONE         8'hFF

// ---------------------------------------------------------------------------
// Status codes
// ---------------------------------------------------------------------------
`define NPU_ST_OK             16'h0000
`define NPU_ST_BAD_VERSION    16'h0001
`define NPU_ST_BAD_OPCODE     16'h0002
`define NPU_ST_BAD_FIELD      16'h0003
`define NPU_ST_RANGE          16'h0004
`define NPU_ST_ALIGN          16'h0005
`define NPU_ST_UNSUPPORTED    16'h0006
`define NPU_ST_DEP_FAILED     16'h0007
`define NPU_ST_DEP_MISSING    16'h0008
`define NPU_ST_DMA_BUS        16'h0009
`define NPU_ST_FRAME_LENGTH   16'h000A
`define NPU_ST_MEM_ACCESS     16'h000B
`define NPU_ST_NUMERIC        16'h000C
`define NPU_ST_INTERNAL       16'h000D
`define NPU_ST_RESET_REQUIRED 16'h000E
`define NPU_ST_EVENT_BUSY     16'h000F

// ---------------------------------------------------------------------------
// Event-table lookup states
// ---------------------------------------------------------------------------
`define NPU_LOOKUP_FREE       3'd0
`define NPU_LOOKUP_RUNNING    3'd1
`define NPU_LOOKUP_OK         3'd2
`define NPU_LOOKUP_FAILED     3'd3
`define NPU_LOOKUP_MISMATCH   3'd4

// ---------------------------------------------------------------------------
// Logical buffers
// ---------------------------------------------------------------------------
`define NPU_BUF_ACT0         8'd0
`define NPU_BUF_ACT1         8'd1
`define NPU_BUF_TMP0         8'd2
`define NPU_BUF_TMP1         8'd3
`define NPU_BUF_KV0          8'd4
`define NPU_BUF_KV1          8'd5
`define NPU_BUF_ATTN         8'd6
`define NPU_BUF_PARAM        8'd7

// ---------------------------------------------------------------------------
// ADMIN action codes
// ---------------------------------------------------------------------------
`define NPU_ADMIN_NONE       2'd0
`define NPU_ADMIN_RELEASE    2'd1

// Element width in bytes for a memory format code (INT4 is not a storage fmt)
`define NPU_ELEM_BYTES(f)    (((f) == `NPU_FMT_FP32) ? 32'd4 : 32'd2)

// ---------------------------------------------------------------------------
// memory_subsystem request / response word layout
//
//   mem_req_t = {tag[8], wstrb[B], wdata[MEM_DATA_W], byte_offset[32],
//                buf_id[8], write[1]}
//   mem_rsp_t = {reason[8], error[1], tag[8], rdata[MEM_DATA_W]}
//
//   DW is the data width (MEM_DATA_W) and B its byte count; both are constant
//   expressions when invoked with module parameters.
// ---------------------------------------------------------------------------
`define NPU_REQ_W(DW,BW)     (1 + 8 + 32 + (DW) + (BW) + 8)
`define NPU_RSP_W(DW)        ((DW) + 8 + 1 + 8)

`define NPU_REQ_WRITE(r)             ((r)[0])
`define NPU_REQ_BUF(r)               ((r)[8:1])
`define NPU_REQ_OFF(r)               ((r)[40:9])
`define NPU_REQ_DATA(r,DW)           ((r)[41+(DW)-1:41])
`define NPU_REQ_STRB(r,DW)           ((r)[41+(DW)+(DW)/8-1:41+(DW)])
`define NPU_REQ_TAG(r,DW)            ((r)[41+(DW)+(DW)/8+7:41+(DW)+(DW)/8])

`define NPU_RSP_DATA(r,DW)           ((r)[(DW)-1:0])
`define NPU_RSP_TAG(r,DW)            ((r)[(DW)+7:(DW)])
`define NPU_RSP_ERR(r,DW)            ((r)[(DW)+8])
`define NPU_RSP_REASON(r,DW)         ((r)[(DW)+16:(DW)+9])

// mem access fault reasons
`define NPU_MEMR_OK        8'd0
`define NPU_MEMR_BUF       8'd1
`define NPU_MEMR_RANGE     8'd2
`define NPU_MEMR_ALIGN     8'd3

// ---------------------------------------------------------------------------
// wtile_desc_t (127 bit), docs/npu_integration/07_weight_buffer.md section 2
//
//   {bank_id[1], consumer_job_id[16], tile_index[16], mode[2], n_grp[4],
//    k_steps[16], weight_bytes[32], scale_bytes[32], layout_version[8]}
//
//   Little-endian field numbering: layout_version occupies bits [7:0].
//   These macros take the descriptor as their argument.
// ---------------------------------------------------------------------------
`define NPU_WTILE_W            127
`define NPU_WTILE_LAYOUT(d)      (d[7:0])
`define NPU_WTILE_SCALE_BYTES(d) (d[39:8])
`define NPU_WTILE_W_BYTES(d)     (d[71:40])
`define NPU_WTILE_KSTEPS(d)      (d[87:72])
`define NPU_WTILE_NGRP(d)        (d[91:88])
`define NPU_WTILE_MODE(d)        (d[93:92])
`define NPU_WTILE_INDEX(d)       (d[109:94])
`define NPU_WTILE_JOB(d)         (d[125:110])
`define NPU_WTILE_BANK(d)        (d[126])

`define NPU_WTILE_LAYOUT_V1     8'h01
`define NPU_E8M0_ONE            32'h7F7F_7F7F

// weight bank release descriptor: {bank_id[1], job[16], tile[16]}
`define NPU_WREL_W              33
`define NPU_WREL_BANK(d)         (d[32])
`define NPU_WREL_JOB(d)          (d[31:16])
`define NPU_WREL_TILE(d)         (d[15:0])

`endif

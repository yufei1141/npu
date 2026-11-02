//=============================================================================
// vector_engine : VECTOR / REDUCE tasks on an FP32 lane datapath
//
//   Follows docs/npu_integration/10_vector_engine.md and the numeric contract
//   of 12_format_and_numeric.md.
//
//   Implementation shape (v0.1 functional baseline)
//   -------------------------------------------------
//   Elements are processed one at a time in ascending index order instead of
//   V_LANES at a time.  The visible semantics are unchanged (index order, tail
//   handled purely by count); the cost is throughput: one scratchpad read per
//   source element, one strobed write per destination element, and a single
//   shared format converter on the destination path.  This is a correctness
//   baseline, not the vector throughput target.
//
//   - 16-bit storage is widened to FP32 on the fly; only the destination is
//     narrowed, through format_converter (RNE, FTZ, flags).
//   - SELECT and masked REDUCE read the mask byte first and then fetch only
//     the candidate they actually use, so an unselected NaN candidate cannot
//     raise a numeric error (doc 10 section 4).
//   - SUM accumulates in index order from +0 with one rounding per add; MAX
//     initialises from the first valid element.  All-masked SUM yields +0,
//     all-masked MAX reports NUMERIC and writes nothing.
//   - ROTATE_PAIR is a raw sign-flip permutation, not a multiply by -1.
//   - COMPARE_GT writes one byte per element (0/1) without the converter.
//   - LAYOUT (opcode 0x23) is executed by tensor_layout, instantiated here.
//     The selection is latched when the task is accepted and holds until the
//     layout event is taken, so request, response and event are muxed on a
//     latched bit and the two datapaths never drive memory in the same task.
//
//   Element addresses are base + index*E with E = 2 or 4 bytes.  Base and
//   stride are element aligned, so an element never straddles a memory word
//   and every write is a single word with byte strobes.
//
//   Numeric flags are sticky for the task and cleared when it starts.  Invalid
//   or overflow ends the task with NUMERIC (elements already written counted);
//   underflow/inexact are recorded and the task still completes OK.  A memory
//   error response ends the task with MEM_ACCESS without fatal; a response
//   that does not match the outstanding request raises fatal.
//=============================================================================
//=============================================================================
// 中文说明
//
//   VECTOR / REDUCE 任务，跑在 FP32 lane 数据通路上。实现依据
//   docs/npu_integration/10_vector_engine.md 和 12_format_and_numeric.md 的
//   数值契约。
//
//   实现形态（v0.1 功能基线）：
//     元素是**一个一个按索引升序处理**的，而不是每拍处理 V_LANES 个。对外
//     可见的语义完全一样（索引顺序、尾部元素只由 count 决定），代价是吞吐：
//     每个源元素一次 scratchpad 读、每个目的元素一次带 strobe 的写、目的
//     通路上只共享一个格式转换器。这是正确性基线，不是向量吞吐的目标形态。
//
//   各项语义：
//     - 16 位存储按需加宽成 FP32；只有**目的地**才收窄，且必经
//       format_converter（RNE、FTZ、带 flag）；
//     - SELECT 与带掩码的 REDUCE **先读 mask 字节，再只取真正会用到的那一个
//       候选值**。这样未被选中的 NaN 候选不会触发数值错误——如果先无条件读
//       数据再做选择，就会出现"没被选中的 NaN 把整条任务打挂"（doc 10 第 4 节）；
//     - SUM 从 +0 开始按索引顺序累加、每次加法舍入一次；MAX 用第一个有效
//       元素初始化。全部被掩码时：SUM 结果为 +0，MAX 报 NUMERIC 且不写任何
//       数据；
//     - ROTATE_PAIR 是**符号位翻转**的置换，不是乘 -1：它只动符号位，完全
//       不经过数值通路，因此不产生舍入、不置任何 flag，也不受 FTZ 影响；
//     - COMPARE_GT 每个元素写 1 个字节（0/1），不经转换器；
//     - LAYOUT（opcode 0x23）由本文件内实例化的 tensor_layout 执行。选择位在
//       任务被接受时锁存、保持到 layout 事件被取走，因此请求、响应和事件都
//       由同一个锁存位来复用，两条数据通路在同一条任务里不会同时驱动存储器。
//
//   地址：base + index*E，E 为 2 或 4 字节。base 和 stride 都是元素对齐的，
//   所以一个元素绝不会横跨存储字，每次写都是单字 + 字节 strobe。
//
//   数值 flag 在整条任务内粘滞，任务开始时清零。invalid 或 overflow 以
//   NUMERIC 结束任务（已经写出的元素仍然计数）；underflow/inexact 只记录，
//   任务照常 OK 完成。存储器错误响应以 MEM_ACCESS 结束且不置 fatal；响应与
//   未完成请求不匹配则置 fatal。
//=============================================================================
`timescale 1ns/1ps
`default_nettype none

`include "npu_defs.vh"

module vector_engine #(
    parameter MEM_DATA_W  = 256,
    parameter MAX_K_STEPS = 256
)(
    input  wire        clk_core,
    input  wire        rst_core_n,

    // scheduler task channel
    input  wire        task_valid,
    output wire        task_ready,
    input  wire [511:0] task_cmd,

    // completion event, held until accepted
    output wire        event_valid,
    input  wire        event_ready,
    output wire [127:0] event_data,

    // scratchpad client VPU
    output wire        mem_req_valid,
    input  wire        mem_req_ready,
    output wire [`NPU_REQ_W(MEM_DATA_W, MEM_DATA_W/8)-1:0] mem_req,
    input  wire        mem_rsp_valid,
    output wire        mem_rsp_ready,
    input  wire [`NPU_RSP_W(MEM_DATA_W)-1:0] mem_rsp,

    output wire [7:0]  fp_exception,
    output wire        fatal
);

    localparam DW    = MEM_DATA_W;
    localparam B     = MEM_DATA_W / 8;
    localparam REQ_W = `NPU_REQ_W(MEM_DATA_W, MEM_DATA_W/8);
    localparam RSP_W = `NPU_RSP_W(MEM_DATA_W);

    localparam [7:0] TAG = 8'h5A;

    localparam S_IDLE  = 4'd0;
    localparam S_CHECK = 4'd1;
    localparam S_RD    = 4'd2;   // pick the next read step (or skip it)
    localparam S_RQ    = 4'd3;   // read request offered
    localparam S_RW    = 4'd4;   // read response
    localparam S_ALU   = 4'd5;
    localparam S_CV0   = 4'd6;   // destination conversion, input side
    localparam S_CV1   = 4'd7;   // destination conversion, output side
    localparam S_WQ    = 4'd8;   // write offer
    localparam S_WW    = 4'd9;   // write ack
    localparam S_FIN   = 4'd10;  // REDUCE scalar write
    localparam S_DONE  = 4'd11;
    localparam S_FAULT = 4'd12;

    localparam RK_NONE = 3'd0;
    localparam RK_S0   = 3'd1;
    localparam RK_S1   = 3'd2;
    localparam RK_S0N  = 3'd3;   // src0[idx+1], the second element of a pair
    localparam RK_MASK = 3'd4;

    localparam A_ADD = 3'd0;
    localparam A_SUB = 3'd1;
    localparam A_MUL = 3'd2;
    localparam A_MAX = 3'd3;
    localparam A_GT  = 3'd4;

    localparam V_ADD  = `NPU_VEC_ADD;
    localparam V_SUB  = `NPU_VEC_SUB;
    localparam V_MUL  = `NPU_VEC_MUL;
    localparam V_MAX  = `NPU_VEC_MAX;
    localparam V_SEL  = `NPU_VEC_SELECT;
    localparam V_CVT  = `NPU_VEC_CONVERT;
    localparam V_MULS = `NPU_VEC_MUL_SCALAR;
    localparam V_ADDS = `NPU_VEC_ADD_SCALAR;
    localparam V_GT   = `NPU_VEC_COMPARE_GT;
    localparam V_ROT  = `NPU_VEC_ROTATE_PAIR;

    // ------------------------------------------------------------------
    // latched task
    // ------------------------------------------------------------------
    reg [3:0]  state;
    reg [7:0]  op_r, sub_r, slot_r, flags_r;
    reg [15:0] job_r;
    reg [1:0]  fs0_r, fs1_r, fd_r;
    reg [7:0]  buf_s0_r, buf_s1_r, buf_d_r, buf_m_r;
    reg [31:0] cnt_r, off_s0_r, off_s1_r, off_d_r, off_m_r, scalar_r;
    reg [127:0] rsv_r;         // P7..P11, must be zero

    reg [31:0] idx;            // iteration base index (pair base for ROTATE)
    reg [1:0]  rd_i;           // read step within the iteration
    reg [1:0]  wr_step;        // write step within the iteration
    reg [1:0]  nwr;            // writes in this iteration

    reg [31:0] s0_raw, s0_f32, s0n_raw, s0n_f32;
    reg [31:0] s1_f32;
    reg        s1_hold;
    reg        mask_bit;
    reg [31:0] acc_r;
    reg        has_val;
    reg [31:0] res0_r;
    reg [31:0] written;

    reg [7:0]  fp_flags;
    reg [15:0] status_r;

    // read transaction bookkeeping
    reg [7:0]  rd_buf_q;
    reg [31:0] rd_off_q;
    reg [2:0]  rd_eb_q;
    reg [2:0]  rd_kind_q;

    // write transaction bookkeeping
    reg [7:0]    wq_buf;
    reg [31:0]   wq_off;
    reg [DW-1:0] wq_data;
    reg [B-1:0]  wq_strb;

    reg         ev_valid;
    reg [127:0] event_r;
    reg         fatal_r;

    // ------------------------------------------------------------------
    // LAYOUT routing
    //
    //   The selection is latched when a task is accepted and held until the
    //   layout event has been taken by the scheduler, so request, response and
    //   event are all muxed on a latched bit rather than on the live opcode.
    // ------------------------------------------------------------------
    reg         sel_layout;
    wire        lay_task_valid;
    wire        lay_task_ready;
    wire        lay_ev_valid;
    wire [127:0] lay_ev_data;
    wire        lay_req_valid, lay_req_ready, lay_rsp_valid, lay_rsp_ready;
    wire        lay_fatal;

    wire [REQ_W-1:0] lay_req;

    wire accept_layout = task_valid && (task_cmd[15:8] == `NPU_OP_LAYOUT);

    assign task_ready   = (state == S_IDLE) && !sel_layout;
    assign lay_task_valid = task_valid && (state == S_IDLE) && !sel_layout &&
                            accept_layout;

    assign event_valid  = sel_layout ? lay_ev_valid : ev_valid;
    assign event_data   = sel_layout ? lay_ev_data  : event_r;
    assign fp_exception = fp_flags;
    assign fatal        = fatal_r | lay_fatal;

    // ------------------------------------------------------------------
    // conversion helpers
    // ------------------------------------------------------------------
    function [2:0] elem_bytes;
        input [1:0] f;
        begin
            elem_bytes = (f == `NPU_FMT_FP32) ? 3'd4 : 3'd2;
        end
    endfunction

    // widen a storage element to FP32; {flags, bits}
    function [39:0] widen_f32;
        input [31:0] raw;
        input [1:0]  fmt;
        reg   [15:0] h;
        reg          s;
        reg   [7:0]  e;
        reg   [9:0]  f10;
        reg   [6:0]  f7;
        begin
            h = raw[15:0];
            s = h[15];
            if (fmt == `NPU_FMT_FP32) begin
                widen_f32 = {8'h00, raw};
            end else if (fmt == `NPU_FMT_FP16) begin
                e   = {3'd0, h[14:10]};
                f10 = h[9:0];
                if (e == 8'd0) begin
                    if (f10 == 10'd0) widen_f32 = {8'h00, {s, 31'd0}};
                    else              widen_f32 = {8'h0C, {s, 31'd0}};
                end else if (e == 8'h1F) begin
                    widen_f32 = {8'h01, (f10 != 10'd0) ? 32'h7FC0_0000
                                                      : {s, 8'hFF, 23'd0}};
                end else begin
                    widen_f32 = {8'h00, {s, e + 8'd112, f10, 13'd0}};
                end
            end else begin                      // BF16
                e  = h[14:7];
                f7 = h[6:0];
                if (e == 8'd0) begin
                    if (f7 == 7'd0) widen_f32 = {8'h00, {s, 31'd0}};
                    else            widen_f32 = {8'h0C, {s, 31'd0}};
                end else if (e == 8'hFF) begin
                    widen_f32 = {8'h01, (f7 != 7'd0) ? 32'h7FC0_0000
                                                     : {s, 8'hFF, 23'd0}};
                end else begin
                    widen_f32 = {8'h00, {s, e, f7, 16'd0}};
                end
            end
        end
    endfunction

    // extract an E-byte element at a byte position inside a memory word
    function [31:0] word_slice;
        input [DW-1:0] w;
        input [4:0]    pos;
        input [2:0]    eb;
        reg   [31:0]   v;
        begin
            v = 32'd0;
            case (eb)
            3'd1: v[7:0]   = w[{pos, 3'b0} +: 8];
            3'd2: v[15:0]  = w[{pos, 3'b0} +: 16];
            default: v     = w[{pos, 3'b0} +: 32];
            endcase
            word_slice = v;
        end
    endfunction

    // ------------------------------------------------------------------
    // operation decode
    // ------------------------------------------------------------------
    wire [5:0] vop = sub_r[5:0];

    wire is_vec = (op_r == `NPU_OP_VECTOR);
    wire is_red = (op_r == `NPU_OP_REDUCE);

    wire bcast_s = sub_r[7];
    wire bcast_p = sub_r[6];
    wire red_max = sub_r[0];
    wire mask_en = is_red && sub_r[7];
    wire sel_mask = is_vec && (vop == V_SEL);

    wire uses_src1 = is_vec && ((vop == V_ADD) || (vop == V_SUB) || (vop == V_MUL) ||
                                (vop == V_MAX) || (vop == V_GT));

    wire [2:0] es0_w = elem_bytes(fs0_r);
    wire [2:0] es1_w = elem_bytes(fs1_r);
    wire [2:0] ed_w  = (is_vec && (vop == V_GT)) ? 3'd1 :
                       is_red ? 3'd4 : elem_bytes(fd_r);

    wire [31:0] s1_idx = bcast_s ? 32'd0 : (bcast_p ? {1'b0, idx[31:1]} : idx);

    // ------------------------------------------------------------------
    // read step selection
    // ------------------------------------------------------------------
    reg [2:0] rd_kind;
    reg       rd_skip;

    always @* begin
        rd_kind = RK_NONE;
        rd_skip = 1'b1;
        if (is_red) begin
            case (rd_i)
            2'd0: begin
                rd_skip = 1'b0;
                rd_kind = mask_en ? RK_MASK : RK_S0;
            end
            2'd1: if (!mask_en || !mask_bit) begin
                rd_skip = 1'b1;                 // no mask, or masked out
            end else begin
                rd_skip = 1'b0;
                rd_kind = RK_S0;
            end
            default: rd_skip = 1'b1;
            endcase
        end else if (sel_mask) begin
            case (rd_i)
            2'd0: begin rd_skip = 1'b0; rd_kind = RK_MASK; end
            2'd1: if (mask_bit) begin
                      rd_skip = 1'b0; rd_kind = RK_S0;
                  end else if (bcast_s) begin
                      rd_skip = 1'b1;           // false candidate is immediate
                  end else begin
                      rd_skip = 1'b0; rd_kind = RK_S1;
                  end
            default: rd_skip = 1'b1;
            endcase
        end else begin
            case (rd_i)
            2'd0: begin rd_skip = 1'b0; rd_kind = RK_S0; end
            2'd1: if (vop == V_ROT) begin
                      rd_skip = 1'b0; rd_kind = RK_S0N;
                  end else if (!uses_src1) begin
                      rd_skip = 1'b1;
                  end else if (bcast_s && s1_hold) begin
                      rd_skip = 1'b1;           // scalar read once per task
                  end else begin
                      rd_skip = 1'b0; rd_kind = RK_S1;
                  end
            default: rd_skip = 1'b1;
            endcase
        end
    end

    reg [7:0]  rd_buf;
    reg [31:0] rd_off;
    reg [2:0]  rd_eb;

    always @* begin
        rd_buf = buf_s0_r;
        rd_eb  = es0_w;
        rd_off = off_s0_r;
        case (rd_kind)
        RK_S0: begin
            rd_buf = buf_s0_r;
            rd_eb  = es0_w;
            rd_off = off_s0_r + (idx * es0_w);
        end
        RK_S0N: begin
            rd_buf = buf_s0_r;
            rd_eb  = es0_w;
            rd_off = off_s0_r + ((idx + 32'd1) * es0_w);
        end
        RK_S1: begin
            rd_buf = buf_s1_r;
            rd_eb  = es1_w;
            rd_off = off_s1_r + (s1_idx * es1_w);
        end
        RK_MASK: begin
            rd_buf = buf_m_r;
            rd_eb  = 3'd1;
            rd_off = off_m_r + idx;
        end
        default: begin
        end
        endcase
    end

    // ------------------------------------------------------------------
    // response decode
    // ------------------------------------------------------------------
    wire        rsp_err = mem_rsp[DW + 8];
    wire [7:0]  rsp_tag = mem_rsp[DW + 7:DW];
    wire        rsp_fire = mem_rsp_valid && mem_rsp_ready;

    wire [31:0] rsp_slice = word_slice(mem_rsp[DW-1:0], rd_off_q[4:0], rd_eb_q);
    wire [39:0] rsp_widen = widen_f32(rsp_slice, (rd_kind_q == RK_S1) ? fs1_r : fs0_r);
    wire        rsp_bad_num = rsp_widen[32] || rsp_widen[33];

    // ------------------------------------------------------------------
    // ALU
    // ------------------------------------------------------------------
    wire [2:0] vec_op = (vop == V_SUB) ? A_SUB :
                        ((vop == V_MUL) || (vop == V_MULS)) ? A_MUL :
                        (vop == V_MAX) ? A_MAX :
                        (vop == V_GT)  ? A_GT : A_ADD;

    wire [2:0] alu_op = is_red ? (red_max ? A_MAX : A_ADD) : vec_op;
    wire [31:0] alu_a = is_red ? acc_r : s0_f32;
    wire [31:0] alu_b = is_red ? s0_f32 :
                        ((vop == V_MULS) || (vop == V_ADDS)) ? scalar_r : s1_f32;

    wire [31:0] alu_y;
    wire        alu_pred;
    wire [7:0]  alu_flags;

    fp32_alu u_alu (
        .a(alu_a), .b(alu_b), .op(alu_op),
        .y(alu_y), .pred(alu_pred), .flags(alu_flags)
    );

    wire [31:0] sel_y = mask_bit ? s0_f32 : (bcast_s ? scalar_r : s1_f32);
    wire [31:0] vec_y = (vop == V_SEL) ? sel_y :
                        (vop == V_CVT) ? s0_f32 : alu_y;

    // ------------------------------------------------------------------
    // destination conversion
    // ------------------------------------------------------------------
    wire        cvt_in_ready, cvt_out_valid;
    wire [31:0] cvt_bits;
    wire [7:0]  cvt_flags;

    format_converter u_cvt (
        .clk(clk_core), .rst_n(rst_core_n),
        .in_valid(state == S_CV0), .in_ready(cvt_in_ready),
        .in_bits(res0_r),
        .src_fmt(`NPU_FMT_FP32), .dst_fmt(fd_r), .in_tag(8'h00),
        .out_valid(cvt_out_valid), .out_ready(state == S_CV1),
        .out_bits(cvt_bits), .out_tag(), .fp_flags(cvt_flags)
    );

    // ------------------------------------------------------------------
    // memory ports
    // ------------------------------------------------------------------
    wire        vec_req_valid = (state == S_RQ) || (state == S_WQ);
    wire [REQ_W-1:0] vec_req = (state == S_WQ)
                   ? {TAG, wq_strb, wq_data, wq_off, wq_buf, 1'b1}
                   : {TAG, {B{1'b0}}, {DW{1'b0}}, {rd_off_q[31:5], 5'd0},
                      rd_buf_q, 1'b0};
    wire        vec_rsp_ready = (state == S_RW) || (state == S_WW);

    assign mem_req_valid = sel_layout ? lay_req_valid : vec_req_valid;
    assign mem_req       = sel_layout ? lay_req       : vec_req;
    assign mem_rsp_ready = sel_layout ? lay_rsp_ready : vec_rsp_ready;

    // only the selected datapath can have a request outstanding

    // ------------------------------------------------------------------
    // iteration geometry
    // ------------------------------------------------------------------
    wire [1:0]  el_step  = (vop == V_ROT) ? 2'd2 : 2'd1;
    wire [31:0] next_idx = idx + {30'd0, el_step};
    wire        idx_last = (next_idx >= cnt_r);

    wire [31:0] wa      = off_d_r + ((idx + {30'd0, wr_step}) * ed_w);
    wire [31:0] wa_next = off_d_r + ((idx + {30'd0, wr_step} + 32'd1) * ed_w);

    // ------------------------------------------------------------------
    // task validation
    // ------------------------------------------------------------------
    reg [15:0] chk_status;

    wire a0_bad = (es0_w == 3'd4) ? |off_s0_r[1:0] : (es0_w == 3'd2) ? off_s0_r[0] : 1'b0;
    wire a1_bad = (es1_w == 3'd4) ? |off_s1_r[1:0] : (es1_w == 3'd2) ? off_s1_r[0] : 1'b0;
    wire ad_bad = (ed_w  == 3'd4) ? |off_d_r[1:0]  : (ed_w  == 3'd2) ? off_d_r[0]  : 1'b0;

    wire [31:0] s1_cnt = bcast_s ? 32'd1 :
                         (bcast_p ? ((cnt_r + 32'd1) >> 1) : cnt_r);

    wire [63:0] s0_end = {32'd0, off_s0_r} + ({32'd0, cnt_r} * {61'd0, es0_w});
    wire [63:0] s1_end = {32'd0, off_s1_r} + ({32'd0, s1_cnt} * {61'd0, es1_w});
    wire [63:0] m_end  = {32'd0, off_m_r}  + {32'd0, cnt_r};
    wire [63:0] d_end  = {32'd0, off_d_r}  + ({32'd0, cnt_r} * {61'd0, ed_w});

    wire s1_rng_used = uses_src1 || sel_mask || (vop == V_ROT);
    wire m_rng_used  = sel_mask || mask_en;

    wire rng_bad = |s0_end[63:33] ||
                   (s1_rng_used && |s1_end[63:33]) ||
                   (m_rng_used  && |m_end[63:33])  ||
                   |d_end[63:33];

    wire inplace = flags_r[`NPU_FLAG_INPLACE];
    wire inplace_ok = is_vec && (vop == V_ADD || vop == V_SUB || vop == V_MUL ||
                                 vop == V_MAX || vop == V_MULS ||
                                 vop == V_ADDS || vop == V_CVT) &&
                      (buf_d_r == buf_s0_r) && (off_d_r == off_s0_r) &&
                      (ed_w == es0_w);

    always @* begin
        chk_status = `NPU_ST_OK;
        if (cnt_r == 32'd0) begin
            chk_status = `NPU_ST_BAD_FIELD;
        end else if (op_r == `NPU_OP_LAYOUT) begin
            chk_status = `NPU_ST_UNSUPPORTED;
        end else if (!is_vec && !is_red) begin
            chk_status = `NPU_ST_BAD_OPCODE;
        end else if (rsv_r != 128'd0) begin
            chk_status = `NPU_ST_BAD_FIELD;          // P7..P11 must be zero
        end else if (fs0_r == `NPU_FMT_INT4) begin
            chk_status = `NPU_ST_UNSUPPORTED;
        end else if (is_red) begin
            if (|sub_r[6:1])                     chk_status = `NPU_ST_BAD_FIELD;
            else if (fd_r != `NPU_FMT_FP32)      chk_status = `NPU_ST_BAD_FIELD;
            else if (buf_s1_r != 8'd0 || off_s1_r != 32'd0)
                                                 chk_status = `NPU_ST_BAD_FIELD;
            else if (!mask_en && (buf_m_r != 8'd0 || off_m_r != 32'd0))
                                                 chk_status = `NPU_ST_BAD_FIELD;
            else if (inplace)                    chk_status = `NPU_ST_BAD_FIELD;
            else if (a0_bad || ad_bad)           chk_status = `NPU_ST_ALIGN;
            else if (rng_bad)                    chk_status = `NPU_ST_RANGE;
        end else if (vop == V_GT) begin
            if (sub_r[7:6] != 2'b00)             chk_status = `NPU_ST_BAD_FIELD;
            else if (fd_r != 2'd0)               chk_status = `NPU_ST_BAD_FIELD;
            else if (a0_bad)                     chk_status = `NPU_ST_ALIGN;
            else if (rng_bad)                    chk_status = `NPU_ST_RANGE;
        end else if (vop == V_SEL) begin
            if (bcast_p)                          chk_status = `NPU_ST_BAD_FIELD;
            else if (bcast_s && (buf_s1_r != 8'd0 || off_s1_r != 32'd0))
                                                  chk_status = `NPU_ST_BAD_FIELD;
            else if (inplace)                     chk_status = `NPU_ST_BAD_FIELD;
            else if (a0_bad || ad_bad)            chk_status = `NPU_ST_ALIGN;
            else if (rng_bad)                     chk_status = `NPU_ST_RANGE;
        end else if (vop == V_CVT || vop == V_MULS || vop == V_ADDS ||
                     vop == V_ROT) begin
            if (sub_r[7:6] != 2'b00)              chk_status = `NPU_ST_BAD_FIELD;
            else if (buf_s1_r != 8'd0 || off_s1_r != 32'd0)
                                                  chk_status = `NPU_ST_BAD_FIELD;
            else if (buf_m_r != 8'd0 || off_m_r != 32'd0)
                                                  chk_status = `NPU_ST_BAD_FIELD;
            else if (vop == V_ROT && cnt_r[0])    chk_status = `NPU_ST_BAD_FIELD;
            else if (vop == V_ROT && fd_r != fs0_r)
                                                  chk_status = `NPU_ST_BAD_FIELD;
            else if (vop == V_ROT && inplace)     chk_status = `NPU_ST_BAD_FIELD;
            else if (vop != V_ROT && inplace && !inplace_ok)
                                                  chk_status = `NPU_ST_BAD_FIELD;
            else if (a0_bad || ad_bad)            chk_status = `NPU_ST_ALIGN;
            else if (rng_bad)                     chk_status = `NPU_ST_RANGE;
        end else begin                            // ADD / SUB / MUL / MAX
            if (bcast_s && bcast_p)               chk_status = `NPU_ST_BAD_FIELD;
            else if (bcast_p && cnt_r[0])         chk_status = `NPU_ST_BAD_FIELD;
            else if ((bcast_s || bcast_p) && buf_d_r == buf_s1_r)
                                                  chk_status = `NPU_ST_BAD_FIELD;
            else if (buf_m_r != 8'd0 || off_m_r != 32'd0)
                                                  chk_status = `NPU_ST_BAD_FIELD;
            else if (inplace && !inplace_ok)      chk_status = `NPU_ST_BAD_FIELD;
            else if (a0_bad || a1_bad || ad_bad)  chk_status = `NPU_ST_ALIGN;
            else if (rng_bad)                     chk_status = `NPU_ST_RANGE;
        end
    end

    // ------------------------------------------------------------------
    // FSM
    // ------------------------------------------------------------------
    always @(posedge clk_core or negedge rst_core_n) begin
        if (!rst_core_n) begin
            state     <= S_IDLE;
            op_r      <= 8'd0;
            sub_r     <= 8'd0;
            slot_r    <= 8'd0;
            flags_r   <= 8'd0;
            job_r     <= 16'd0;
            fs0_r     <= 2'd0;
            fs1_r     <= 2'd0;
            fd_r      <= 2'd0;
            buf_s0_r  <= 8'd0;
            buf_s1_r  <= 8'd0;
            buf_d_r   <= 8'd0;
            buf_m_r   <= 8'd0;
            cnt_r     <= 32'd0;
            off_s0_r  <= 32'd0;
            off_s1_r  <= 32'd0;
            off_d_r   <= 32'd0;
            off_m_r   <= 32'd0;
            scalar_r  <= 32'd0;
            rsv_r     <= 128'd0;
            idx       <= 32'd0;
            rd_i      <= 2'd0;
            wr_step   <= 2'd0;
            nwr       <= 2'd1;
            s0_raw    <= 32'd0;
            s0_f32    <= 32'd0;
            s0n_raw   <= 32'd0;
            s0n_f32   <= 32'd0;
            s1_f32    <= 32'd0;
            s1_hold   <= 1'b0;
            mask_bit  <= 1'b0;
            acc_r     <= 32'd0;
            has_val   <= 1'b0;
            res0_r    <= 32'd0;
            written   <= 32'd0;
            fp_flags  <= 8'd0;
            status_r  <= `NPU_ST_OK;
            rd_buf_q  <= 8'd0;
            rd_off_q  <= 32'd0;
            rd_eb_q   <= 3'd2;
            rd_kind_q <= RK_NONE;
            wq_buf    <= 8'd0;
            wq_off    <= 32'd0;
            wq_data   <= {DW{1'b0}};
            wq_strb   <= {B{1'b0}};
            ev_valid  <= 1'b0;
            event_r   <= 128'd0;
            fatal_r   <= 1'b0;
            sel_layout <= 1'b0;
        end else begin
            case (state)
            // ------------------------------------------------------
            S_IDLE: begin
                if (sel_layout) begin
                    // the layout datapath owns the task; wait for its event
                    if (lay_ev_valid && event_ready) sel_layout <= 1'b0;
                end else if (task_valid) begin
                    if (accept_layout) begin
                        sel_layout <= 1'b1;
                        ev_valid   <= 1'b0;
                    end else begin
                    op_r     <= task_cmd[15:8];
                    sub_r    <= task_cmd[23:16];
                    flags_r  <= task_cmd[31:24];
                    job_r    <= task_cmd[47:32];
                    buf_s0_r <= task_cmd[71:64];
                    buf_s1_r <= task_cmd[79:72];
                    buf_d_r  <= task_cmd[87:80];
                    fs0_r    <= task_cmd[89:88];
                    fs1_r    <= task_cmd[91:90];
                    fd_r     <= task_cmd[93:92];
                    slot_r   <= task_cmd[103:96];
                    cnt_r    <= task_cmd[159:128];
                    off_s0_r <= task_cmd[191:160];
                    off_s1_r <= task_cmd[223:192];
                    off_d_r  <= task_cmd[255:224];
                    scalar_r <= task_cmd[287:256];
                    buf_m_r  <= task_cmd[295:288];
                    off_m_r  <= task_cmd[351:320];
                    rsv_r    <= task_cmd[511:352];
                    ev_valid <= 1'b0;
                    state    <= S_CHECK;
                    end
                end
            end
            // ------------------------------------------------------
            S_CHECK: begin
                idx      <= 32'd0;
                rd_i     <= 2'd0;
                wr_step  <= 2'd0;
                written  <= 32'd0;
                fp_flags <= 8'd0;
                acc_r    <= 32'd0;
                has_val  <= 1'b0;
                s1_hold  <= 1'b0;
                mask_bit <= 1'b0;
                nwr      <= (sub_r[5:0] == V_ROT) ? 2'd2 : 2'd1;
                status_r <= chk_status;
                if (chk_status != `NPU_ST_OK) state <= S_DONE;
                else                          state <= S_RD;
            end
            // ------------------------------------------------------
            S_RD: begin
                if (rd_skip) begin
                    if (rd_i == 2'd2) begin
                        rd_i  <= 2'd0;
                        state <= S_ALU;
                    end else begin
                        rd_i <= rd_i + 2'd1;
                    end
                end else begin
                    rd_buf_q  <= rd_buf;
                    rd_off_q  <= rd_off;
                    rd_eb_q   <= rd_eb;
                    rd_kind_q <= rd_kind;
                    state     <= S_RQ;
                end
            end
            // ------------------------------------------------------
            S_RQ: begin
                if (mem_req_ready) state <= S_RW;
            end
            // ------------------------------------------------------
            S_RW: begin
                if (rsp_fire) begin
                    if (rsp_tag != TAG) begin
                        fatal_r  <= 1'b1;
                        status_r <= `NPU_ST_INTERNAL;
                        state    <= S_FAULT;
                    end else if (rsp_err) begin
                        status_r <= `NPU_ST_MEM_ACCESS;
                        state    <= S_DONE;
                    end else begin
                        case (rd_kind_q)
                        RK_S0: begin
                            s0_raw   <= rsp_slice;
                            s0_f32   <= rsp_widen[31:0];
                            fp_flags <= fp_flags | rsp_widen[39:32];
                        end
                        RK_S0N: begin
                            s0n_raw  <= rsp_slice;
                            s0n_f32  <= rsp_widen[31:0];
                            fp_flags <= fp_flags | rsp_widen[39:32];
                        end
                        RK_S1: begin
                            s1_f32   <= rsp_widen[31:0];
                            fp_flags <= fp_flags | rsp_widen[39:32];
                            if (bcast_s) s1_hold <= 1'b1;
                        end
                        default: begin
                            mask_bit <= rsp_slice[0];
                        end
                        endcase

                        if (rsp_bad_num && rd_kind_q != RK_MASK) begin
                            status_r <= `NPU_ST_NUMERIC;
                            state    <= S_DONE;
                        end else begin
                            rd_i  <= rd_i + 2'd1;
                            state <= S_RD;
                        end
                    end
                end
            end
            // ------------------------------------------------------
            S_ALU: begin
                if (is_red) begin
                    if (mask_en && !mask_bit) begin
                        // masked element does not participate
                        if (idx_last) begin
                            state <= S_FIN;
                        end else begin
                            idx   <= next_idx;
                            rd_i  <= 2'd0;
                            state <= S_RD;
                        end
                    end else if (red_max && !has_val) begin
                        acc_r   <= s0_f32;
                        has_val <= 1'b1;
                        if (idx_last) begin
                            state <= S_FIN;
                        end else begin
                            idx   <= next_idx;
                            rd_i  <= 2'd0;
                            state <= S_RD;
                        end
                    end else begin
                        fp_flags <= fp_flags | alu_flags;
                        if (alu_flags[0] || alu_flags[1]) begin
                            status_r <= `NPU_ST_NUMERIC;
                            state    <= S_DONE;
                        end else if (idx_last) begin
                            acc_r <= alu_y;
                            state <= S_FIN;
                        end else begin
                            acc_r <= alu_y;
                            idx   <= next_idx;
                            rd_i  <= 2'd0;
                            state <= S_RD;
                        end
                    end
                end else if (vop == V_GT) begin
                    fp_flags <= fp_flags | alu_flags;
                    if (alu_flags[0] || alu_flags[1]) begin
                        status_r <= `NPU_ST_NUMERIC;
                        state    <= S_DONE;
                    end else begin
                        res0_r  <= {31'd0, alu_pred};
                        wq_buf  <= buf_d_r;
                        wq_off  <= {wa[31:5], 5'd0};
                        wq_data <= {{(DW-32){1'b0}}, 32'd0}
                                   | ({{(DW-1){1'b0}}, alu_pred} << {wa[4:0], 3'd0});
                        wq_strb <= 32'd1 << wa[4:0];
                        wr_step <= 2'd0;
                        state   <= S_WQ;
                    end
                end else if (vop == V_ROT) begin
                    // out[2j] = -x[2j+1], out[2j+1] = x[2j]; raw sign flip
                    res0_r  <= s0n_raw ^ ((es0_w == 3'd4) ? 32'h8000_0000
                                                          : 32'h0000_8000);
                    wq_buf  <= buf_d_r;
                    wq_off  <= {wa[31:5], 5'd0};
                    wq_data <= {{(DW-32){1'b0}},
                                s0n_raw ^ ((es0_w == 3'd4) ? 32'h8000_0000
                                                           : 32'h0000_8000)}
                               << {wa[4:0], 3'd0};
                    wq_strb <= ((32'd1 << ed_w) - 32'd1) << wa[4:0];
                    wr_step <= 2'd0;
                    state   <= S_WQ;
                end else begin
                    fp_flags <= fp_flags | alu_flags;
                    if (alu_flags[0] || alu_flags[1]) begin
                        status_r <= `NPU_ST_NUMERIC;
                        state    <= S_DONE;
                    end else begin
                        res0_r  <= vec_y;
                        wr_step <= 2'd0;
                        state   <= S_CV0;
                    end
                end
            end
            // ------------------------------------------------------
            S_CV0: begin
                if (cvt_in_ready) state <= S_CV1;
            end
            // ------------------------------------------------------
            S_CV1: begin
                if (cvt_out_valid) begin
                    fp_flags <= fp_flags | cvt_flags;
                    if (cvt_flags[0] || cvt_flags[1]) begin
                        status_r <= `NPU_ST_NUMERIC;
                        state    <= S_DONE;
                    end else begin
                        wq_buf  <= buf_d_r;
                        wq_off  <= {wa[31:5], 5'd0};
                        wq_data <= {{(DW-32){1'b0}}, cvt_bits} << {wa[4:0], 3'd0};
                        wq_strb <= ((32'd1 << ed_w) - 32'd1) << wa[4:0];
                        state   <= S_WQ;
                    end
                end
            end
            // ------------------------------------------------------
            S_WQ: begin
                if (mem_req_ready) state <= S_WW;
            end
            // ------------------------------------------------------
            S_WW: begin
                if (rsp_fire) begin
                    if (rsp_tag != TAG) begin
                        fatal_r  <= 1'b1;
                        status_r <= `NPU_ST_INTERNAL;
                        state    <= S_FAULT;
                    end else if (rsp_err) begin
                        status_r <= `NPU_ST_MEM_ACCESS;
                        state    <= S_DONE;
                    end else if ((wr_step + 2'd1) < nwr) begin
                        // second write of a ROTATE_PAIR iteration
                        wr_step <= wr_step + 2'd1;
                        wq_buf  <= buf_d_r;
                        wq_off  <= {wa_next[31:5], 5'd0};
                        wq_data <= {{(DW-32){1'b0}}, s0_raw} << {wa_next[4:0], 3'd0};
                        wq_strb <= ((32'd1 << ed_w) - 32'd1) << wa_next[4:0];
                        state   <= S_WQ;
                    end else begin
                        written <= written + {30'd0, nwr};
                        wr_step <= 2'd0;
                        if (idx_last) begin
                            state <= S_DONE;
                        end else begin
                            idx   <= next_idx;
                            rd_i  <= 2'd0;
                            state <= S_RD;
                        end
                    end
                end
            end
            // ------------------------------------------------------
            S_FIN: begin
                if (red_max && !has_val) begin
                    // every element was masked: no scalar to write
                    status_r <= `NPU_ST_NUMERIC;
                    state    <= S_DONE;
                end else begin
                    wq_buf  <= buf_d_r;
                    wq_off  <= {off_d_r[31:5], 5'd0};
                    wq_data <= {{(DW-32){1'b0}}, acc_r} << {off_d_r[4:0], 3'd0};
                    wq_strb <= 32'h0000_000F << off_d_r[4:0];
                    state   <= S_WQ;
                end
            end
            // ------------------------------------------------------
            S_DONE: begin
                if (!ev_valid) begin
                    event_r  <= {32'd0, written, 16'd0, status_r,
                                 op_r, slot_r, job_r};
                    ev_valid <= 1'b1;
                end else if (event_ready) begin
                    ev_valid <= 1'b0;
                    state    <= S_IDLE;
                end
            end
            // ------------------------------------------------------
            S_FAULT: begin
                fatal_r <= 1'b1;
                if (!ev_valid) begin
                    event_r  <= {32'd0, written, 16'd0, status_r,
                                 op_r, slot_r, job_r};
                    ev_valid <= 1'b1;
                end else if (event_ready) begin
                    ev_valid <= 1'b0;
                end
                // sticky: recovery requires the platform reset
            end
            // ------------------------------------------------------
            default: state <= S_FAULT;
            endcase
        end
    end

    // ------------------------------------------------------------------
    // LAYOUT datapath (shares client 1 and the VPU event source)
    // ------------------------------------------------------------------
    tensor_layout #(
        .MEM_DATA_W(MEM_DATA_W),
        .MAX_K_STEPS(MAX_K_STEPS)
    ) u_layout (
        .clk_core(clk_core),
        .rst_core_n(rst_core_n),
        .task_valid(lay_task_valid),
        .task_ready(lay_task_ready),
        .task_cmd(task_cmd),
        .event_valid(lay_ev_valid),
        .event_ready(event_ready),
        .event_data(lay_ev_data),
        .mem_req_valid(lay_req_valid),
        .mem_req_ready(mem_req_ready),
        .mem_req(lay_req),
        .mem_rsp_valid(mem_rsp_valid),
        .mem_rsp_ready(lay_rsp_ready),
        .mem_rsp(mem_rsp),
        .active(),
        .element_index(),
        .fatal(lay_fatal)
    );

endmodule

`default_nettype wire

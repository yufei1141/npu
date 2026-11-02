//=============================================================================
// sfu_adapter : SFU task adapter around the existing sfu_top core
//
//   Follows docs/npu_integration/11_sfu_adapter.md.  The core is used as-is:
//   the adapter pads a lane vector to the core width, maps the logical subop
//   to the core opcode and converts storage formats to and from the FP32 lane
//   profile the core is built for.  It never writes the core datapath.
//
//   Subops and their core opcodes (mapped explicitly, never connected raw):
//     0 EXP2 -> 0    1 RCP -> 2       2 RSQRT -> 3    3 SINCOS -> 8
//     4 LOG2 -> 1    5 SQRT -> 4      6 SIN -> 5      7 COS -> 6
//     8 TANH -> 7
//   0..3 keep the encoding they had before the core grew the remaining
//   functions, so older commands keep their meaning.
//
//   P4[1:0] selects the core mode (0 fast, 1 accurate); it is latched with the
//   task and held for the whole batch like op/fmt.  Other values are rejected.
//
//   capability_ops / capability_formats are build-time verification results,
//   not run-time configuration.  A function or storage format whose bit is
//   clear is rejected with UNSUPPORTED instead of being run on a datapath that
//   has not been accepted against a numeric reference.
//
//   Batching
//   --------
//   L = min(8, count - index) elements per core call.  Unused lanes are filled
//   with a finite safe value (0 for EXP2/SINCOS/SIN/COS/TANH, 1.0 for
//   RCP/RSQRT/LOG2/SQRT) so a
//   tail lane cannot raise a false div0 through the lane-aggregated status,
//   and they are never written back.  One batch is in flight at a time: op,
//   fmt, mode and din are held from ISSUE until the last output of the batch
//   is acknowledged, because the core has no tag and no frame counter.
//
//   SINCOS uses the two-frame profile: one issue, then sin into P3 on frame 0
//   and cos into P2 on frame 1, strictly in that order.  Both planes live in
//   dst_buf and must not overlap.
//
//   Errors
//   ------
//   A non-zero core status, or a narrowing overflow on the way out, ends the
//   task with NUMERIC; a scratchpad error response ends it with MEM_ACCESS.
//   The remaining frames already in flight are drained (no new batch starts)
//   and only then is the event raised, so the core is never left with an
//   unread result.  A response that does not match the outstanding request is
//   fatal.  Storage-format widening that meets NaN/Inf ends the task with
//   NUMERIC as well.
//=============================================================================
//=============================================================================
// 中文说明
//
//   SFU 任务适配器，包在现成的 sfu_top 内核外面。实现依据
//   docs/npu_integration/11_sfu_adapter.md。内核原样使用：适配层负责把一条
//   lane 向量补齐到内核宽度、把逻辑 subop 映射成内核 opcode、并在存储器
//   格式与内核的 FP32 lane 规格之间做收发转换。适配层**从不改写内核的数据
//   通路**。
//
//   subop 到内核 opcode 的映射（显式映射，绝不是把字段直连）：
//     0 EXP2 -> 0   1 RCP -> 2   2 RSQRT -> 3   3 SINCOS -> 8
//     4 LOG2 -> 1   5 SQRT -> 4  6 SIN -> 5     7 COS -> 6     8 TANH -> 7
//   0..3 沿用内核扩展功能之前的编码，所以老命令的语义不会因为新增函数而改变。
//
//   P4[1:0] 选择内核模式（0 快速、1 精确），与 op/fmt 一样在任务开始时锁存，
//   整批保持；其它取值一律拒绝。
//
//   capability_ops / capability_formats 是**编译期验证结果**，不是运行期配置。
//   某一位为 0 表示该函数或该存储格式还没通过数值参考比对，此时直接以
//   UNSUPPORTED 拒绝，而不是硬跑——这是防止"能跑但结果不对"的护栏。
//
//   分批（batching）：
//     - 每次内核调用处理 L = min(8, count - index) 个元素；
//     - 用不到的 lane 填入"安全的有限值"（EXP2/SINCOS/SIN/COS/TANH 填 0，
//       RCP/RSQRT/LOG2/SQRT 填 1.0），避免尾巴 lane 通过按 lane 聚合的状态位
//       制造假的除零；这些值永远不会被写回；
//     - 同一时刻只有一批在途：op、fmt、mode、din 从 ISSUE 一直保持到本批最后
//       一个输出被接收，因为内核既没有 tag 也没有帧计数器，无法区分迟到数据
//       属于哪一批。
//
//   SINCOS 走两帧格式：一次 issue，第 0 帧把 sin 写入 P3、第 1 帧把 cos 写入
//   P2，顺序严格如此。两个平面都在 dst_buf 内且不得重叠。
//
//   错误处理：
//     - 内核状态非零、或输出时收窄上溢，以 NUMERIC 结束；
//     - scratchpad 返回错误响应，以 MEM_ACCESS 结束；
//     - 出错后**先把已经在途的帧排空**（不再启动新批），然后才上报事件，
//       这样内核不会留下没人读的结果；
//     - 响应与未完成请求不匹配属于 fatal；
//     - 存储格式加宽时遇到 NaN/Inf 同样以 NUMERIC 结束。
//=============================================================================
`timescale 1ns/1ps
`default_nettype none

`include "npu_defs.vh"

module sfu_adapter #(
    parameter MEM_DATA_W    = 256,
    parameter CORE_LANES    = 8,
    // core format encoding: 0 FP32, 1 BF16, 2 FP16 (not the storage encoding
    // used by NPU_FMT_*); the core lane profile is FP32
    parameter [1:0] CORE_FMT_IN  = 2'd0,
    parameter [1:0] CORE_FMT_OUT = 2'd0,
    parameter SINCOS_TWO_FRAME   = 1
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

    // scratchpad client SFU
    output wire        mem_req_valid,
    input  wire        mem_req_ready,
    output wire [`NPU_REQ_W(MEM_DATA_W, MEM_DATA_W/8)-1:0] mem_req,
    input  wire        mem_rsp_valid,
    output wire        mem_rsp_ready,
    input  wire [`NPU_RSP_W(MEM_DATA_W)-1:0] mem_rsp,

    // build-time capability bits, one per subop 0..8
    input  wire [8:0]  capability_ops,
    input  wire [2:0]  capability_formats,

    // sfu_top
    output wire        core_in_valid,
    input  wire        core_in_ready,
    output wire [3:0]  core_op,
    output wire [1:0]  core_fmt_in,
    output wire [1:0]  core_fmt_out,
    output wire [1:0]  core_mode,
    output wire [CORE_LANES*32-1:0] core_din,
    input  wire        core_out_valid,
    output wire        core_out_ready,
    input  wire [CORE_LANES*32-1:0] core_dout,
    input  wire [7:0]  core_status,
    output wire        core_flush,

    output reg         fatal
);

    localparam DW = MEM_DATA_W;
    localparam B  = MEM_DATA_W / 8;

    localparam [7:0] TAG = 8'h6F;

    localparam S_IDLE     = 5'd0;
    localparam S_CHECK    = 5'd1;
    localparam S_LD       = 5'd2;   // next load lane, or fill, or ISSUE
    localparam S_LRQ      = 5'd3;
    localparam S_LRW      = 5'd4;
    localparam S_ISSUE    = 5'd5;
    localparam S_WAIT     = 5'd6;
    localparam S_ST       = 5'd7;   // next store lane
    localparam S_CV0      = 5'd8;   // destination conversion, input side
    localparam S_CV1      = 5'd9;   // destination conversion, output side
    localparam S_WQ       = 5'd10;
    localparam S_WW       = 5'd11;
    localparam S_NEXTB    = 5'd12;
    localparam S_DRAIN    = 5'd13;
    localparam S_COMPLETE = 5'd14;
    localparam S_FAULT    = 5'd15;

    localparam [3:0] CORE_OP_EXP2   = 4'd0;
    localparam [3:0] CORE_OP_LOG2   = 4'd1;
    localparam [3:0] CORE_OP_RCP    = 4'd2;
    localparam [3:0] CORE_OP_RSQRT  = 4'd3;
    localparam [3:0] CORE_OP_SQRT   = 4'd4;
    localparam [3:0] CORE_OP_SIN    = 4'd5;
    localparam [3:0] CORE_OP_COS    = 4'd6;
    localparam [3:0] CORE_OP_TANH   = 4'd7;
    localparam [3:0] CORE_OP_SINCOS = 4'd8;

    // ------------------------------------------------------------------
    // latched task
    // ------------------------------------------------------------------
    reg [4:0]   state;
    reg [7:0]   op_r, sub_r, slot_r, flags_r;
    reg [15:0]  job_r;
    reg [1:0]   mode_r;           // core mode from P4[1:0], held per batch
    reg [1:0]   fs0_r, fd_r;
    reg [7:0]   buf_s_r, buf_d_r;
    reg [31:0]  cnt_r, off_s_r, off_d_r, off_c_r;

    reg [31:0]  idx;
    reg [3:0]   lid;              // lane counter, 0..8
    reg [1:0]   phase;            // SINCOS output plane
    reg [1:0]   frames_left;      // frames of this batch not yet consumed
    reg [31:0]  written;

    reg [CORE_LANES*32-1:0] lane_r;
    reg [CORE_LANES*32-1:0] res_r;

    // read transaction bookkeeping
    reg [7:0]  rd_buf_q;
    reg [31:0] rd_off_q;
    reg [2:0]  rd_eb_q;

    // write transaction bookkeeping
    reg [7:0]    wq_buf;
    reg [31:0]   wq_off;
    reg [DW-1:0] wq_data;
    reg [B-1:0]  wq_strb;

    reg [15:0]  status_r;
    reg         ev_valid;
    reg [127:0] event_r;

    assign task_ready  = (state == S_IDLE);
    assign event_valid = ev_valid;
    assign event_data  = event_r;

    // ------------------------------------------------------------------
    // conversion helpers (same contract as vector_engine)
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
    // task validation
    // ------------------------------------------------------------------
    wire [7:0]  d_opcode = task_cmd[15:8];
    wire [7:0]  d_sub    = task_cmd[23:16];
    wire [7:0]  d_flags  = task_cmd[31:24];
    wire [15:0] d_job    = task_cmd[47:32];
    wire [7:0]  d_s0b    = task_cmd[71:64];
    wire [7:0]  d_s1b    = task_cmd[79:72];
    wire [7:0]  d_db     = task_cmd[87:80];
    wire [7:0]  d_fmt    = task_cmd[95:88];
    wire [7:0]  d_slot   = task_cmd[103:96];
    wire [15:0] d_rsvd   = task_cmd[127:112];

    // `NPU_CMD_P cannot take task_cmd: a bit-select on a parenthesised
    // expression is not Verilog-2001, so the payload words are sliced here
    wire [31:0]  d_cnt  = task_cmd[159:128];
    wire [31:0]  d_soff = task_cmd[191:160];
    wire [31:0]  d_coff = task_cmd[223:192];
    wire [31:0]  d_doff = task_cmd[255:224];
    wire [31:0]  d_p4   = task_cmd[287:256];
    wire [223:0] d_p5_11 = task_cmd[511:288];

    wire [1:0] d_fs0  = d_fmt[1:0];
    wire [1:0] d_fs1  = d_fmt[3:2];
    wire [1:0] d_fd   = d_fmt[5:4];
    wire [1:0] d_facc = d_fmt[7:6];

    wire d_opcode_ok = (d_opcode == `NPU_OP_SFU);
    wire d_sub_ok  = (d_sub <= 8'd8);
    wire d_op_ok   = d_opcode_ok && d_sub_ok;
    wire d_sincos  = (d_sub == 8'd3);
    wire d_prof_ok = !d_sincos || (SINCOS_TWO_FRAME != 0);
    wire d_cap     = d_op_ok && capability_ops[d_sub[3:0]];
    wire d_mode_ok = d_p4[1:0] <= 2'd1;

    wire d_fmt_ok  = (d_fs0 != `NPU_FMT_INT4) && (d_fd != `NPU_FMT_INT4) &&
                     (d_fs1 == 2'd0) && (d_facc == 2'd0) &&
                     capability_formats[d_fs0] && capability_formats[d_fd];

    // P2 is only a second output plane for SINCOS
    wire d_p2_ok   = d_sincos || (d_coff == 32'd0);

    wire d_body_ok = (d_s1b == 8'd0) &&
                     (d_flags[`NPU_FLAG_INPLACE] == 1'b0) &&
                     (d_rsvd == 16'd0) && (d_p5_11 == 224'd0) &&
                     (d_p4[31:2] == 30'd0) && d_p2_ok && d_mode_ok;

    wire [2:0] d_es = elem_bytes(d_fs0);
    wire [2:0] d_ed = elem_bytes(d_fd);

    wire d_a_src = (d_es == 3'd4) ? (|d_soff[1:0]) : d_soff[0];
    wire d_a_dst = (d_ed == 3'd4) ? (|d_doff[1:0]) : d_doff[0];
    wire d_a_cos = (d_ed == 3'd4) ? (|d_coff[1:0]) : d_coff[0];

    // 64-bit ends so a wrapped range cannot pass as in-range
    wire [63:0] d_send = {32'd0, d_soff} + ({32'd0, d_cnt} * d_es);
    wire [63:0] d_dend = {32'd0, d_doff} + ({32'd0, d_cnt} * d_ed);
    wire [63:0] d_cend = {32'd0, d_coff} + ({32'd0, d_cnt} * d_ed);

    wire d_ovf = (d_send[63:32] != 32'd0) || (d_dend[63:32] != 32'd0) ||
                 (d_sincos && (d_cend[63:32] != 32'd0));

    // SINCOS planes must be disjoint, and units = 2*count must fit
    wire d_ovl = d_sincos &&
                 ({32'd0, d_doff} < d_cend) && ({32'd0, d_coff} < d_dend);

    wire d_ok = d_op_ok && d_prof_ok && d_cap && d_fmt_ok && d_body_ok &&
                (d_cnt != 32'd0) &&
                (!d_sincos || (d_cnt[31] == 1'b0)) &&
                !d_a_src && !d_a_dst && (!d_sincos || !d_a_cos) &&
                !d_ovf && !d_ovl;

    wire [15:0] d_status =
        !d_opcode_ok ? `NPU_ST_BAD_OPCODE :
        !d_sub_ok   ? `NPU_ST_BAD_FIELD :
        !d_prof_ok  ? `NPU_ST_UNSUPPORTED :
        !d_cap      ? `NPU_ST_UNSUPPORTED :
        !d_fmt_ok   ? `NPU_ST_UNSUPPORTED :
        !d_body_ok  ? `NPU_ST_BAD_FIELD  :
        (d_cnt == 32'd0) ? `NPU_ST_BAD_FIELD :
        (d_sincos && d_cnt[31]) ? `NPU_ST_RANGE :
        (d_a_src || d_a_dst || (d_sincos && d_a_cos)) ? `NPU_ST_ALIGN :
        d_ovf       ? `NPU_ST_RANGE      :
        d_ovl       ? `NPU_ST_BAD_FIELD  : `NPU_ST_OK;

    // ------------------------------------------------------------------
    // batch geometry and core mapping
    // ------------------------------------------------------------------
    wire [31:0] remain  = cnt_r - idx;
    wire [3:0]  lbatch  = (remain >= 32'd8) ? 4'd8 : remain[3:0];
    wire [31:0] next_idx = idx + {28'd0, lbatch};

    wire [2:0] es_r = elem_bytes(fs0_r);
    wire [2:0] ed_r = elem_bytes(fd_r);

    wire [31:0] ld_addr = off_s_r + ((idx + {28'd0, lid[2:0]}) * es_r);

    wire [31:0] plane_off = (phase == 2'd0) ? off_d_r : off_c_r;
    wire [31:0] st_addr   = plane_off + ((idx + {28'd0, lid[2:0]}) * ed_r);

    // padding lanes must not raise a false flag through the lane-aggregated
    // status: RCP/RSQRT(1)=1, LOG2(4): log2(1)=0, SQRT(5): sqrt(1)=1
    wire [31:0] safe_val = ((sub_r == 8'd1) || (sub_r == 8'd2) ||
                            (sub_r == 8'd4) || (sub_r == 8'd5))
                           ? 32'h3F80_0000 : 32'h0000_0000;

    wire [1:0]  nframes = (sub_r == 8'd3) ? 2'd2 : 2'd1;

    reg  [3:0]  core_op_w;
    always @* begin
        case (sub_r[3:0])
        4'd0:    core_op_w = CORE_OP_EXP2;
        4'd1:    core_op_w = CORE_OP_RCP;
        4'd2:    core_op_w = CORE_OP_RSQRT;
        4'd3:    core_op_w = CORE_OP_SINCOS;
        4'd4:    core_op_w = CORE_OP_LOG2;
        4'd5:    core_op_w = CORE_OP_SQRT;
        4'd6:    core_op_w = CORE_OP_SIN;
        4'd7:    core_op_w = CORE_OP_COS;
        4'd8:    core_op_w = CORE_OP_TANH;
        default: core_op_w = CORE_OP_EXP2;
        endcase
    end

    assign core_in_valid  = (state == S_ISSUE);
    assign core_op        = core_op_w;
    assign core_fmt_in    = CORE_FMT_IN;
    assign core_fmt_out   = CORE_FMT_OUT;
    assign core_mode      = mode_r;
    assign core_din       = lane_r;
    assign core_out_ready = (state == S_WAIT) || (state == S_DRAIN);
    assign core_flush     = 1'b0;

    // ------------------------------------------------------------------
    // destination conversion
    // ------------------------------------------------------------------
    wire        cvt_in_ready, cvt_out_valid;
    wire [31:0] cvt_bits;
    wire [7:0]  cvt_flags;

    format_converter u_cvt (
        .clk(clk_core), .rst_n(rst_core_n),
        .in_valid(state == S_CV0), .in_ready(cvt_in_ready),
        .in_bits(res_r[{lid[2:0], 5'd0} +: 32]),
        .src_fmt(`NPU_FMT_FP32), .dst_fmt(fd_r), .in_tag(8'h00),
        .out_valid(cvt_out_valid), .out_ready(state == S_CV1),
        .out_bits(cvt_bits), .out_tag(), .fp_flags(cvt_flags)
    );

    // ------------------------------------------------------------------
    // memory ports
    // ------------------------------------------------------------------
    assign mem_req_valid = (state == S_LRQ) || (state == S_WQ);
    assign mem_req = (state == S_WQ)
                   ? {TAG, wq_strb, wq_data, wq_off, wq_buf, 1'b1}
                   : {TAG, {B{1'b0}}, {DW{1'b0}}, {rd_off_q[31:5], 5'd0},
                      rd_buf_q, 1'b0};
    assign mem_rsp_ready = (state == S_LRW) || (state == S_WW);

    wire        rsp_err  = mem_rsp[DW + 8];
    wire [7:0]  rsp_tag  = mem_rsp[DW + 7:DW];
    wire        rsp_fire = mem_rsp_valid && mem_rsp_ready;

    wire [31:0] rsp_slice = word_slice(mem_rsp[DW-1:0], rd_off_q[4:0], rd_eb_q);
    wire [39:0] rsp_widen = widen_f32(rsp_slice, fs0_r);
    wire        rsp_bad_num = rsp_widen[32] || rsp_widen[33];

    // ------------------------------------------------------------------
    // adapter FSM
    // ------------------------------------------------------------------
    always @(posedge clk_core or negedge rst_core_n) begin
        if (!rst_core_n) begin
            state      <= S_IDLE;
            op_r       <= 8'd0;
            sub_r      <= 8'd0;
            slot_r     <= 8'd0;
            flags_r    <= 8'd0;
            job_r      <= 16'd0;
            mode_r     <= 2'd0;
            fs0_r      <= 2'd0;
            fd_r       <= 2'd0;
            buf_s_r    <= 8'd0;
            buf_d_r    <= 8'd0;
            cnt_r      <= 32'd0;
            off_s_r    <= 32'd0;
            off_d_r    <= 32'd0;
            off_c_r    <= 32'd0;
            idx        <= 32'd0;
            lid        <= 4'd0;
            phase      <= 2'd0;
            frames_left<= 2'd0;
            written    <= 32'd0;
            lane_r     <= {(CORE_LANES*32){1'b0}};
            res_r      <= {(CORE_LANES*32){1'b0}};
            rd_buf_q   <= 8'd0;
            rd_off_q   <= 32'd0;
            rd_eb_q    <= 3'd0;
            wq_buf     <= 8'd0;
            wq_off     <= 32'd0;
            wq_data    <= {DW{1'b0}};
            wq_strb    <= {B{1'b0}};
            status_r   <= `NPU_ST_OK;
            ev_valid   <= 1'b0;
            event_r    <= 128'd0;
            fatal      <= 1'b0;
        end else begin
            case (state)
            // ----------------------------------------------------------
            S_IDLE: begin
                if (task_valid) begin
                    op_r    <= d_opcode;
                    sub_r   <= d_sub;
                    slot_r  <= d_slot;
                    flags_r <= d_flags;
                    job_r   <= d_job;
                    mode_r  <= d_p4[1:0];
                    fs0_r   <= d_fs0;
                    fd_r    <= d_fd;
                    buf_s_r <= d_s0b;
                    buf_d_r <= d_db;
                    off_s_r <= d_soff;
                    off_d_r <= d_doff;
                    off_c_r <= d_coff;
                    idx     <= 32'd0;
                    lid     <= 4'd0;
                    phase   <= 2'd0;
                    written <= 32'd0;
                    status_r<= `NPU_ST_OK;
                    cnt_r   <= d_ok ? d_cnt : 32'd0;
                    if (d_ok) begin
                        state <= S_CHECK;
                    end else begin
                        status_r <= d_status;
                        cnt_r    <= 32'd0;
                        state    <= S_COMPLETE;
                    end
                end
            end
            // ----------------------------------------------------------
            S_CHECK: begin
                // the rejected path never reaches here; CHECK is kept as the
                // single place where a task's dependent state is armed
                state <= S_LD;
            end
            // ----------------------------------------------------------
            S_LD: begin
                if (lid == 4'd8) begin
                    lid   <= 4'd0;
                    state <= S_ISSUE;
                end else if (lid >= lbatch) begin
                    // unused tail lane: finite safe value, never written back
                    lane_r[{lid[2:0], 5'd0} +: 32] <= safe_val;
                    lid   <= lid + 4'd1;
                    state <= S_LD;
                end else begin
                    rd_buf_q <= buf_s_r;
                    rd_off_q <= ld_addr;
                    rd_eb_q  <= es_r;
                    state    <= S_LRQ;
                end
            end
            // ----------------------------------------------------------
            S_LRQ: begin
                if (mem_req_ready) state <= S_LRW;
            end
            // ----------------------------------------------------------
            S_LRW: begin
                if (rsp_fire) begin
                    if (rsp_tag != TAG) begin
                        fatal    <= 1'b1;
                        status_r <= `NPU_ST_INTERNAL;
                        state    <= S_FAULT;
                    end else if (rsp_err) begin
                        status_r <= `NPU_ST_MEM_ACCESS;
                        state    <= S_DRAIN;
                    end else if (rsp_bad_num) begin
                        // NaN/Inf in a 16-bit source cannot enter the lanes
                        status_r <= `NPU_ST_NUMERIC;
                        state    <= S_DRAIN;
                    end else begin
                        lane_r[{lid[2:0], 5'd0} +: 32] <= rsp_widen[31:0];
                        lid   <= lid + 4'd1;
                        state <= S_LD;
                    end
                end
            end
            // ----------------------------------------------------------
            S_ISSUE: begin
                if (core_in_ready) begin
                    frames_left <= nframes;
                    phase       <= 2'd0;
                    state       <= S_WAIT;
                end
            end
            // ----------------------------------------------------------
            S_WAIT: begin
                if (core_out_valid) begin
                    res_r       <= core_dout;
                    frames_left <= frames_left - 2'd1;
                    if (core_status != 8'd0) begin
                        // profile-defined aggregate status: any bit set is a
                        // numeric failure (doc 11 section 6)
                        status_r <= `NPU_ST_NUMERIC;
                        state    <= S_DRAIN;
                    end else begin
                        lid   <= 4'd0;
                        state <= S_ST;
                    end
                end
            end
            // ----------------------------------------------------------
            S_ST: begin
                if (lid == 4'd8) begin
                    lid <= 4'd0;
                    if (frames_left != 2'd0) begin
                        phase <= phase + 2'd1;      // second SINCOS plane
                        state <= S_WAIT;
                    end else begin
                        state <= S_NEXTB;
                    end
                end else if (lid >= lbatch) begin
                    lid   <= lid + 4'd1;            // tail lane, not written
                    state <= S_ST;
                end else begin
                    state <= S_CV0;
                end
            end
            // ----------------------------------------------------------
            S_CV0: begin
                if (cvt_in_ready) state <= S_CV1;
            end
            // ----------------------------------------------------------
            S_CV1: begin
                if (cvt_out_valid) begin
                    wq_buf  <= buf_d_r;
                    wq_off  <= {st_addr[31:5], 5'd0};
                    wq_data <= {{(DW-32){1'b0}}, cvt_bits} << {st_addr[4:0], 3'b0};
                    wq_strb <= ((32'd1 << ed_r) - 32'd1) << st_addr[4:0];
                    if (cvt_flags[0] || cvt_flags[1]) begin
                        status_r <= `NPU_ST_NUMERIC;
                        state    <= S_DRAIN;
                    end else begin
                        state <= S_WQ;
                    end
                end
            end
            // ----------------------------------------------------------
            S_WQ: begin
                if (mem_req_ready) state <= S_WW;
            end
            // ----------------------------------------------------------
            S_WW: begin
                if (rsp_fire) begin
                    if (rsp_tag != TAG) begin
                        fatal    <= 1'b1;
                        status_r <= `NPU_ST_INTERNAL;
                        state    <= S_FAULT;
                    end else if (rsp_err) begin
                        status_r <= `NPU_ST_MEM_ACCESS;
                        state    <= S_DRAIN;
                    end else begin
                        written <= written + 32'd1;
                        lid     <= lid + 4'd1;
                        state   <= S_ST;
                    end
                end
            end
            // ----------------------------------------------------------
            S_NEXTB: begin
                idx <= next_idx;
                if (next_idx >= cnt_r) begin
                    state <= S_COMPLETE;
                end else begin
                    state <= S_LD;
                end
            end
            // ----------------------------------------------------------
            S_DRAIN: begin
                if (frames_left == 2'd0) begin
                    state <= S_COMPLETE;
                end else if (core_out_valid) begin
                    frames_left <= frames_left - 2'd1;
                end
            end
            // ----------------------------------------------------------
            S_COMPLETE: begin
                if (!ev_valid) begin
                    event_r  <= {32'd0, written, 16'd0, status_r,
                                 op_r, slot_r, job_r};
                    ev_valid <= 1'b1;
                end else if (event_ready) begin
                    ev_valid <= 1'b0;
                    state    <= S_IDLE;
                end
            end
            // ----------------------------------------------------------
            S_FAULT: begin
                fatal <= 1'b1;
                if (!ev_valid) begin
                    event_r  <= {32'd0, written, 16'd0, status_r,
                                 op_r, slot_r, job_r};
                    ev_valid <= 1'b1;
                end else if (event_ready) begin
                    ev_valid <= 1'b0;
                end
                // sticky: recovery requires the platform reset
            end
            default: state <= S_IDLE;
            endcase
        end
    end

endmodule

`default_nettype wire

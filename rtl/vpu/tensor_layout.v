//=============================================================================
// tensor_layout : LAYOUT task (TRANSPOSE2D, PACK_W16)
//
//   Follows docs/npu_integration/16_tensor_layout_and_kv.md.  Instantiated by
//   vector_engine and sharing memory client 1 and the VPU completion source;
//   vector_engine selects one datapath per task and muxes request, response
//   and event on the latched selection.
//
//   Both subops move raw elements: no arithmetic, no format conversion, and the
//   source and destination formats must be identical.  Each element is one
//   aligned read (when the element is inside the logical source) followed by
//   one word write with byte strobes, so bytes outside the element are never
//   touched.  Elements are not merged even when they share a word -- the
//   minimal implementation of doc 16 section 5.
//
//   TRANSPOSE2D   src = src_base + r*src_stride + c*E
//                 dst = dst_base + c*dst_stride + r*E        for rows x cols
//   PACK_W16      j=0..g-1, s=0..S-1, p=0..3, i=0..3, k=4s+i, n=4j+p
//                 src = src_base + k*src_stride + 2*n
//                 dst = dst_base + 32*(j*S+s) + 8*p + 2*i
//                 k>=K or n>=N is padding: a +0 element is written, the source
//                 is not read, and it does not count as a completed unit.
//
//   completed_units is rows*cols (TRANSPOSE2D) or K*N (PACK_W16); a task only
//   completes successfully after every element and every padding word has been
//   write-acknowledged.  Buffer capacity is checked statically by
//   access_decode and again by the memory subsystem; this module checks shape,
//   alignment, limits and 32-bit address overflow.
//=============================================================================
//=============================================================================
// 中文说明
//
//   LAYOUT 任务的实现：TRANSPOSE2D 与 PACK_W16。实现依据
//   docs/npu_integration/16_tensor_layout_and_kv.md。
//
//   它与 vector_engine 共用 memory client 1 和 VPU 的完成源：vector_engine 在
//   每条任务里选一条数据通路，请求、响应和事件都由锁存的选择位来复用，因此
//   两条通路在同一条任务里不会同时驱动存储器。
//
//   共同性质：两个 subop 都只**搬原始元素**——不做算术、不做格式转换，源格式
//   与目的格式必须完全一致。每个元素是一次对齐读（当该元素落在逻辑源区间内）
//   加一次带字节 strobe 的字写，因此元素之外的字节永远不会被碰到。即使多个
//   元素共处一个字也**不合并写**，这是 doc 16 第 5 节规定的最小实现。
//
//   TRANSPOSE2D：src = src_base + r*src_stride + c*E
//                dst = dst_base + c*dst_stride + r*E        （rows x cols）
//
//   PACK_W16：j=0..g-1、s=0..S-1、p=0..3、i=0..3、k=4s+i、n=4j+p
//                src = src_base + k*src_stride + 2*n
//                dst = dst_base + 32*(j*S+s) + 8*p + 2*i
//             当 k>=K 或 n>=N 时是**填充**：写入一个 +0 元素，不读源，也不
//             计入完成的元素数。这个"写零而不读"的约定保证了 A/B 帧的尾部
//             是确定的零，而不是上一轮残留。
//
//   完成判据：completed_units = rows*cols（TRANSPOSE2D）或 K*N（PACK_W16）；
//   只有每个元素和每个填充字都被写确认，任务才算成功完成。缓冲区容量由
//   access_decode 静态检查一次、memory_subsystem 再检查一次；本模块负责检查
//   形状、对齐、上限以及 32 位地址溢出。
//=============================================================================
`timescale 1ns/1ps
`default_nettype none

`include "npu_defs.vh"

module tensor_layout #(
    parameter MEM_DATA_W  = 256,
    parameter MAX_K_STEPS = 256
)(
    input  wire        clk_core,
    input  wire        rst_core_n,

    input  wire        task_valid,
    output wire        task_ready,
    input  wire [511:0] task_cmd,

    output wire        event_valid,
    input  wire        event_ready,
    output wire [127:0] event_data,

    output wire        mem_req_valid,
    input  wire        mem_req_ready,
    output wire [`NPU_REQ_W(MEM_DATA_W, MEM_DATA_W/8)-1:0] mem_req,
    input  wire        mem_rsp_valid,
    output wire        mem_rsp_ready,
    input  wire [`NPU_RSP_W(MEM_DATA_W)-1:0] mem_rsp,

    output wire        active,
    output wire [31:0] element_index,
    output reg         fatal
);

    localparam DW = MEM_DATA_W;
    localparam B  = MEM_DATA_W / 8;

    localparam [7:0] TAG = 8'h7B;

    localparam S_IDLE  = 4'd0;
    localparam S_CHECK = 4'd1;
    localparam S_IDX   = 4'd2;   // pick the next element
    localparam S_RQ    = 4'd3;   // source read offered
    localparam S_RW    = 4'd4;   // source read response
    localparam S_PREP  = 4'd5;   // build the destination word and strobes
    localparam S_WQ    = 4'd6;   // destination write offered
    localparam S_WW    = 4'd7;   // destination write acknowledged
    localparam S_NEXT  = 4'd8;
    localparam S_DONE  = 4'd9;
    localparam S_FAULT = 4'd10;

    // ------------------------------------------------------------------
    // latched task
    // ------------------------------------------------------------------
    reg [3:0]  state;
    reg [7:0]  op_r, sub_r, slot_r, flags_r;
    reg [15:0] job_r;
    reg [1:0]  fs_r, fd_r, acc_r;
    reg [7:0]  buf_s_r, buf_d_r, buf_s1_r;
    reg [31:0] p0_r, p1_r, p2_r, p3_r, p4_r, p5_r;
    reg [31:0] p6_r;
    reg [159:0] rsv_r;          // command bits 511:352, P7..P11
    reg [15:0] rsv0_r;          // command bits 127:112, always zero

    // geometry
    reg [31:0] rows_r, cols_r, kd_r, nd_r, g_r, ssteps_r;
    reg [31:0] sstr_r, dstr_r, sbase_r, dbase_r;
    reg [3:0]  e_r;

    // walk
    reg [31:0] r_r, c_r, s_r, j_r, iter_r;
    reg [1:0]  p_r, i_r;
    reg [31:0] written;

    // memory
    reg [7:0]  rd_buf_q;
    reg [31:0] rd_off_q;
    reg [4:0]  lane_r;
    reg [31:0] el_hold;
    reg [7:0]  wq_buf;
    reg [31:0] wq_off;
    reg [DW-1:0] wq_data;
    reg [B-1:0]  wq_strb;

    reg [15:0] status_r;
    reg        ev_valid;
    reg [127:0] event_r;

    assign task_ready    = (state == S_IDLE);
    assign event_valid   = ev_valid;
    assign event_data    = event_r;
    assign active        = (state != S_IDLE);
    assign element_index = iter_r;

    // ------------------------------------------------------------------
    // memory ports
    // ------------------------------------------------------------------
    assign mem_req_valid = (state == S_RQ) || (state == S_WQ);
    assign mem_req = (state == S_WQ)
                   ? {TAG, wq_strb, wq_data, wq_off, wq_buf, 1'b1}
                   : {TAG, {B{1'b0}}, {DW{1'b0}}, rd_off_q, rd_buf_q, 1'b0};
    assign mem_rsp_ready = (state == S_RW) || (state == S_WW);

    wire [DW-1:0] rsp_data = mem_rsp[DW-1:0];
    wire          rsp_err  = mem_rsp[DW + 8];
    wire [7:0]    rsp_tag  = mem_rsp[DW + 7:DW];

    // ------------------------------------------------------------------
    // index and addresses
    // ------------------------------------------------------------------
    wire [63:0] e64 = {60'd0, e_r};
    wire [1:0]  amask = (e_r == 4'd4) ? 2'b11 : 2'b01;

    wire [63:0] td_sa = {32'd0, sbase_r} + ({32'd0, r_r} * {32'd0, sstr_r})
                                       + ({32'd0, c_r} * e64);
    wire [63:0] td_da = {32'd0, dbase_r} + ({32'd0, c_r} * {32'd0, dstr_r})
                                       + ({32'd0, r_r} * e64);

    wire [31:0] pk_k = ({30'd0, s_r} << 2) + {30'd0, i_r};
    wire [31:0] pk_n = ({30'd0, j_r} << 2) + {30'd0, p_r};
    wire [63:0] pk_sa = {32'd0, sbase_r} + ({32'd0, pk_k} * {32'd0, sstr_r})
                                        + ({32'd0, pk_n} * 64'd2);
    wire [63:0] pk_da = {32'd0, dbase_r}
                      + (({32'd0, j_r} * {32'd0, ssteps_r} + {32'd0, s_r}) * 64'd32)
                      + ({32'd0, p_r} * 64'd8) + ({32'd0, i_r} * 64'd2);

    wire td_sel  = (sub_r == `NPU_LAYOUT_TRANSPOSE);
    wire [63:0] sa_c = td_sel ? td_sa : pk_sa;
    wire [63:0] da_c = td_sel ? td_da : pk_da;
    wire        need_src = td_sel ? 1'b1 : ((pk_k < kd_r) && (pk_n < nd_r));

    // ------------------------------------------------------------------
    // element extraction from the read word and placement in the write word
    // ------------------------------------------------------------------
    integer bi;
    reg [31:0]   el_ext;
    reg [DW-1:0] wdata_c;
    reg [B-1:0]  wstrb_c;

    always @* begin
        el_ext = 32'd0;
        for (bi = 0; bi < B; bi = bi + 1) begin
            if ((bi >= lane_r) && (bi < lane_r + e_r))
                el_ext[8*(bi - lane_r) +: 8] = rsp_data[8*bi +: 8];
        end
    end

    always @* begin
        wdata_c = {DW{1'b0}};
        wstrb_c = {B{1'b0}};
        for (bi = 0; bi < B; bi = bi + 1) begin
            if ((bi >= da_c[4:0]) && (bi < da_c[4:0] + e_r)) begin
                wstrb_c[bi] = 1'b1;
                wdata_c[8*bi +: 8] = el_hold[8*(bi - da_c[4:0]) +: 8];
            end
        end
    end

    // ------------------------------------------------------------------
    // task validation
    // ------------------------------------------------------------------
    reg [15:0] chk_status;

    wire [63:0] td_src_end = {32'd0, p2_r} + (({32'd0, p0_r} - 64'd1) *
                             {32'd0, p4_r}) + ({32'd0, p1_r} * e64);
    wire [63:0] td_dst_end = {32'd0, p3_r} + (({32'd0, p1_r} - 64'd1) *
                             {32'd0, p5_r}) + ({32'd0, p0_r} * e64);

    wire [31:0] pack_s    = (p0_r + 32'd3) >> 2;
    wire [63:0] pk_src_end = {32'd0, p2_r} + (({32'd0, p0_r} - 64'd1) *
                             {32'd0, p4_r}) + ({32'd0, p1_r} * 64'd2);
    wire [63:0] pk_dst_end = {32'd0, p3_r} + ({32'd0, p5_r} * 32'd32 *
                             {32'd0, pack_s});
    wire [63:0] pk_units   = {32'd0, p0_r} * {32'd0, p1_r};
    wire [63:0] td_units   = {32'd0, p0_r} * {32'd0, p1_r};

    always @* begin
        chk_status = `NPU_ST_OK;
        if (op_r != `NPU_OP_LAYOUT) begin
            chk_status = `NPU_ST_BAD_OPCODE;
        end else if (sub_r != `NPU_LAYOUT_TRANSPOSE &&
                     sub_r != `NPU_LAYOUT_PACK_W16) begin
            chk_status = `NPU_ST_BAD_FIELD;
        end else if (flags_r[`NPU_FLAG_INPLACE] || buf_s1_r != 8'd0 ||
                     acc_r != 2'd0) begin
            chk_status = `NPU_ST_BAD_FIELD;
        end else if (p6_r != 32'd0 || |rsv_r || rsv0_r != 16'd0) begin
            chk_status = `NPU_ST_BAD_FIELD;
        end else if (fd_r != fs_r) begin
            chk_status = `NPU_ST_BAD_FIELD;
        end else if (fs_r == `NPU_FMT_INT4) begin
            chk_status = `NPU_ST_UNSUPPORTED;
        end else if (!td_sel && fs_r == `NPU_FMT_FP32) begin
            chk_status = `NPU_ST_BAD_FIELD;
        end else if (p0_r == 32'd0 || p1_r == 32'd0) begin
            chk_status = `NPU_ST_BAD_FIELD;
        end else if (td_sel) begin
            if (|(p2_r[1:0] & amask) || |(p3_r[1:0] & amask) ||
                |(p4_r[1:0] & amask) || |(p5_r[1:0] & amask)) begin
                chk_status = `NPU_ST_ALIGN;
            end else if ({32'd0, p4_r} < ({32'd0, p1_r} * e64) ||
                         {32'd0, p5_r} < ({32'd0, p0_r} * e64)) begin
                chk_status = `NPU_ST_BAD_FIELD;
            end else if (|td_units[63:32] || |td_src_end[63:33] ||
                         |td_dst_end[63:33]) begin
                chk_status = `NPU_ST_RANGE;
            end
        end else begin
            if (p5_r < 32'd1 || p5_r > 32'd8) begin
                chk_status = `NPU_ST_BAD_FIELD;
            end else if (p3_r[3:0] != 4'd0) begin
                chk_status = `NPU_ST_ALIGN;
            end else if (p2_r[0] != 1'b0 || p4_r[0] != 1'b0) begin
                chk_status = `NPU_ST_ALIGN;
            end else if (p4_r < (p1_r << 1)) begin
                chk_status = `NPU_ST_BAD_FIELD;
            end else if (p1_r > (p5_r << 2)) begin
                chk_status = `NPU_ST_BAD_FIELD;
            end else if (pack_s > MAX_K_STEPS) begin
                chk_status = `NPU_ST_RANGE;
            end else if (|pk_units[63:32] || |pk_src_end[63:33] ||
                         |pk_dst_end[63:33]) begin
                chk_status = `NPU_ST_RANGE;
            end
        end
    end

    // ------------------------------------------------------------------
    // walk termination
    // ------------------------------------------------------------------
    wire td_last = ((r_r + 32'd1) == rows_r) && ((c_r + 32'd1) == cols_r);
    wire pk_last = ((j_r + 32'd1) == g_r) && ((s_r + 32'd1) == ssteps_r) &&
                   (p_r == 2'd3) && (i_r == 2'd3);
    wire iter_last = td_sel ? td_last : pk_last;

    // ------------------------------------------------------------------
    // FSM
    // ------------------------------------------------------------------
    always @(posedge clk_core or negedge rst_core_n) begin
        if (!rst_core_n) begin
            state    <= S_IDLE;
            op_r     <= 8'd0;
            sub_r    <= 8'd0;
            slot_r   <= 8'd0;
            flags_r  <= 8'd0;
            job_r    <= 16'd0;
            fs_r     <= 2'd0;
            fd_r     <= 2'd0;
            acc_r    <= 2'd0;
            buf_s_r  <= 8'd0;
            buf_d_r  <= 8'd0;
            buf_s1_r <= 8'd0;
            p0_r     <= 32'd0;
            p1_r     <= 32'd0;
            p2_r     <= 32'd0;
            p3_r     <= 32'd0;
            p4_r     <= 32'd0;
            p5_r     <= 32'd0;
            p6_r     <= 32'd0;
            rsv_r    <= 160'd0;
            rsv0_r   <= 16'd0;
            rows_r   <= 32'd0;
            cols_r   <= 32'd0;
            kd_r     <= 32'd0;
            nd_r     <= 32'd0;
            g_r      <= 32'd0;
            ssteps_r <= 32'd0;
            sstr_r   <= 32'd0;
            dstr_r   <= 32'd0;
            sbase_r  <= 32'd0;
            dbase_r  <= 32'd0;
            e_r      <= 4'd2;
            r_r      <= 32'd0;
            c_r      <= 32'd0;
            s_r      <= 32'd0;
            j_r      <= 32'd0;
            p_r      <= 2'd0;
            i_r      <= 2'd0;
            iter_r   <= 32'd0;
            written  <= 32'd0;
            rd_buf_q <= 8'd0;
            rd_off_q <= 32'd0;
            lane_r   <= 5'd0;
            el_hold  <= 32'd0;
            wq_buf   <= 8'd0;
            wq_off   <= 32'd0;
            wq_data  <= {DW{1'b0}};
            wq_strb  <= {B{1'b0}};
            status_r <= `NPU_ST_OK;
            ev_valid <= 1'b0;
            event_r  <= 128'd0;
            fatal    <= 1'b0;
        end else begin
            case (state)
            // ------------------------------------------------------
            S_IDLE: begin
                if (task_valid) begin
                    op_r     <= task_cmd[15:8];
                    sub_r    <= task_cmd[23:16];
                    flags_r  <= task_cmd[31:24];
                    job_r    <= task_cmd[47:32];
                    buf_s_r  <= task_cmd[71:64];
                    buf_s1_r <= task_cmd[79:72];
                    buf_d_r  <= task_cmd[87:80];
                    fs_r     <= task_cmd[89:88];
                    fd_r     <= task_cmd[93:92];
                    acc_r    <= task_cmd[95:94];
                    slot_r   <= task_cmd[103:96];
                    e_r      <= (task_cmd[89:88] == `NPU_FMT_FP32) ? 4'd4 : 4'd2;
                    p0_r     <= task_cmd[159:128];
                    p1_r     <= task_cmd[191:160];
                    p2_r     <= task_cmd[223:192];
                    p3_r     <= task_cmd[255:224];
                    p4_r     <= task_cmd[287:256];
                    p5_r     <= task_cmd[319:288];
                    p6_r     <= task_cmd[351:320];
                    rsv_r    <= task_cmd[511:352];
                    rsv0_r   <= task_cmd[127:112];
                    ev_valid <= 1'b0;
                    state    <= S_CHECK;
                end
            end
            // ------------------------------------------------------
            S_CHECK: begin
                iter_r  <= 32'd0;
                written <= 32'd0;
                r_r     <= 32'd0;
                c_r     <= 32'd0;
                s_r     <= 32'd0;
                j_r     <= 32'd0;
                p_r     <= 2'd0;
                i_r     <= 2'd0;
                status_r <= chk_status;
                if (chk_status != `NPU_ST_OK) begin
                    state <= S_DONE;
                end else begin
                    rows_r   <= p0_r;
                    cols_r   <= p1_r;
                    kd_r     <= p0_r;
                    nd_r     <= p1_r;
                    g_r      <= p5_r;
                    ssteps_r <= (p0_r + 32'd3) >> 2;
                    sbase_r  <= p2_r;
                    dbase_r  <= p3_r;
                    sstr_r   <= p4_r;
                    dstr_r   <= p5_r;
                    state    <= S_IDX;
                end
            end
            // ------------------------------------------------------
            S_IDX: begin
                iter_r <= iter_r + 32'd1;
                if (need_src) begin
                    rd_buf_q <= buf_s_r;
                    rd_off_q <= {sa_c[31:5], 5'd0};
                    lane_r   <= sa_c[4:0];
                    state    <= S_RQ;
                end else begin
                    el_hold <= 32'd0;
                    state   <= S_PREP;
                end
            end
            // ------------------------------------------------------
            S_RQ: begin
                if (mem_req_ready) state <= S_RW;
            end
            // ------------------------------------------------------
            S_RW: begin
                if (mem_rsp_valid) begin
                    if (rsp_tag != TAG) begin
                        status_r <= `NPU_ST_INTERNAL;
                        state    <= S_FAULT;
                    end else if (rsp_err) begin
                        status_r <= `NPU_ST_MEM_ACCESS;
                        state    <= S_DONE;
                    end else begin
                        el_hold <= el_ext;
                        state   <= S_PREP;
                    end
                end
            end
            // ------------------------------------------------------
            S_PREP: begin
                wq_buf  <= buf_d_r;
                wq_off  <= {da_c[31:5], 5'd0};
                wq_data <= wdata_c;
                wq_strb <= wstrb_c;
                state   <= S_WQ;
            end
            // ------------------------------------------------------
            S_WQ: begin
                if (mem_req_ready) state <= S_WW;
            end
            // ------------------------------------------------------
            S_WW: begin
                if (mem_rsp_valid) begin
                    if (rsp_tag != TAG) begin
                        status_r <= `NPU_ST_INTERNAL;
                        state    <= S_FAULT;
                    end else if (rsp_err) begin
                        status_r <= `NPU_ST_MEM_ACCESS;
                        state    <= S_DONE;
                    end else begin
                        if (need_src) written <= written + 32'd1;
                        state <= S_NEXT;
                    end
                end
            end
            // ------------------------------------------------------
            S_NEXT: begin
                if (td_sel) begin
                    if ((c_r + 32'd1) == cols_r) begin
                        c_r <= 32'd0;
                        r_r <= r_r + 32'd1;
                    end else begin
                        c_r <= c_r + 32'd1;
                    end
                end else begin
                    if (i_r != 2'd3) begin
                        i_r <= i_r + 2'd1;
                    end else begin
                        i_r <= 2'd0;
                        if (p_r != 2'd3) begin
                            p_r <= p_r + 2'd1;
                        end else begin
                            p_r <= 2'd0;
                            if ((s_r + 32'd1) == ssteps_r) begin
                                s_r <= 32'd0;
                                j_r <= j_r + 32'd1;
                            end else begin
                                s_r <= s_r + 32'd1;
                            end
                        end
                    end
                end
                state <= iter_last ? S_DONE : S_IDX;
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
                fatal <= 1'b1;
                if (!ev_valid) begin
                    event_r  <= {32'd0, written, 16'd0, status_r,
                                 op_r, slot_r, job_r};
                    ev_valid <= 1'b1;
                end else if (event_ready) begin
                    ev_valid <= 1'b0;
                end
            end
            // ------------------------------------------------------
            default: state <= S_FAULT;
            endcase
        end
    end

endmodule

`default_nettype wire

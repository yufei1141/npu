//=============================================================================
// dma_engine : ordinary LOAD/STORE between external memory and the scratchpad
//
//   Follows docs/npu_integration/06_dma_engine.md.
//
//   One DMA command in flight, one external row transaction in flight.  The
//   module spans the core and io clock domains:
//
//     core domain                        io domain
//       task bridge                        command FSM (IDLE/W_ALLOC/ROW_REQ/
//       scratchpad data bridge             ROW_DATA/ROW_WAIT/ROW_ADV/FINISH)
//       row-write acknowledgement          external frame handling
//
//   Cross-domain paths:
//     cmd mailbox      core -> io   (512 bit, four-phase)
//     done mailbox     io   -> core (128 bit: units + status)
//     row ack mailbox  core -> io   (32 bit: {status[15:0], row_index[15:0]},
//                                    normal LOAD only)
//     load FIFO        io   -> core (depth 8, 146 bit beat)
//     store FIFO        core -> io  (depth 8, 146 bit beat)
//
//   Beat format in both FIFOs: {error, row_last, keep[15:0], data[127:0]}.
//
//   Every row always ends with exactly one beat carrying row_last, including
//   rows terminated by an error: once a row has failed the io side emits a
//   zero-length error beat for each remaining row instead of starting a new
//   external transaction.  That keeps the core data bridge free of abort
//   paths.  A frame whose boundary is not trustworthy (tag mismatch) raises
//   fatal; the remaining rows are still drained so the core does not hang,
//   but no further external request is issued.
//
//   Throughput note: the core bridge serialises each scratchpad word (issue
//   request, wait response) and the io side keeps one row in flight, matching
//   the v0.1 "one outstanding per client / per row" model.  This is a
//   correctness baseline, not the 18 GB/s target.
//
//   LOAD_W (subop 1) has no scratchpad data path: the io side hands the
//   command to weight_loader and streams the frame straight to its data port.
//   Weight bank release is owned by weight_reader, never by this module.
//=============================================================================
//=============================================================================
// 中文说明
//
//   普通 LOAD/STORE 通道：在片外存储（DDR）和片内 scratchpad 之间搬数据。
//   实现依据 docs/npu_integration/06_dma_engine.md。
//
//   并发度：一条 DMA 命令在途、一个外部行事务在途。这是刻意的保守设计——
//   它让"每个客户端一个未完成请求"的模型在整条链路上成立，正确性容易论证，
//   但也意味着吞吐是基线而非目标（见文末的吞吐说明）。
//
//   两个时钟域的分工：
//     core 域：task 桥、scratchpad 数据桥、行写确认；
//     io   域：命令状态机（IDLE / W_ALLOC / ROW_REQ / ROW_DATA / ROW_WAIT /
//              ROW_ADV / FINISH）、外部帧处理。
//
//   跨域通路（全部有明确的握手或 FIFO，没有裸信号跨域）：
//     cmd 邮箱      core -> io   512 位，四相握手
//     done 邮箱     io   -> core 128 位：units + status
//     row ack 邮箱  core -> io   32 位：{status[15:0], row_index[15:0]}，
//                                只用于普通 LOAD
//     load FIFO     io   -> core 深度 8，146 位一个 beat
//     store FIFO    core -> io   深度 8，146 位一个 beat
//   两个 FIFO 的 beat 格式一致：{error, row_last, keep[15:0], data[127:0]}。
//
//   错误处理的核心约定：**每一行都必须以恰好一个带 row_last 的 beat 结束**，
//   出错的行也一样。一旦某行失败，io 侧对剩余每一行只发一个"零长度的错误
//   beat"，而不再发起新的外部事务——这样 core 侧的数据桥就不需要任何 abort
//   路径，状态机只有一条主线。若帧边界不可信（tag 不匹配）则直接置 fatal，
//   剩余行仍然被排空以免 core 挂死，但不再发出外部请求。
//
//   吞吐说明：core 桥对每个 scratchpad 字都是"发请求、等响应"的串行节奏，
//   io 侧只保持一行在途，对应 v0.1 的"每客户端/每行一个未完成"模型。这是
//   正确性基线，不是 18 GB/s 的目标形态。
//
//   LOAD_W（subop 1）没有 scratchpad 数据通路：io 侧把命令交给 weight_loader，
//   帧数据直接流到它的数据口。权重 bank 的释放权归 weight_reader，绝不归本
//   模块——否则会出现两个模块同时管理同一组 bank 的引用计数。
//=============================================================================
`timescale 1ns/1ps
`default_nettype none

`include "npu_defs.vh"

module dma_engine #(
    parameter MEM_DATA_W = 256,
    parameter EXT_WR_W   = 128
)(
    input  wire        clk_core,
    input  wire        rst_core_n,
    input  wire        clk_io,
    input  wire        rst_io_n,

    // scheduler task channel (core)
    input  wire        task_valid,
    output wire        task_ready,
    input  wire [511:0] task_cmd,

    // completion event (core)
    output wire        event_valid,
    input  wire        event_ready,
    output wire [127:0] event_data,

    // scratchpad client DMA_CORE (core)
    output wire        mem_req_valid,
    input  wire        mem_req_ready,
    output wire [`NPU_REQ_W(MEM_DATA_W, MEM_DATA_W/8)-1:0] mem_req,
    input  wire        mem_rsp_valid,
    output wire        mem_rsp_ready,
    input  wire [`NPU_RSP_W(MEM_DATA_W)-1:0] mem_rsp,

    // external memory read (io)
    output wire        rd_req_valid,
    input  wire        rd_req_ready,
    output wire [63:0] rd_req_addr,
    output wire [31:0] rd_req_bytes,
    output wire [15:0] rd_req_tag,
    input  wire        rd_valid,
    output wire        rd_ready,
    input  wire [127:0] rd_data,
    input  wire [15:0] rd_keep,
    input  wire        rd_last,
    input  wire [15:0] rd_tag,
    input  wire        rd_error,

    // external memory write (io)
    output wire        wr_req_valid,
    input  wire        wr_req_ready,
    output wire [63:0] wr_req_addr,
    output wire [31:0] wr_req_bytes,
    output wire [15:0] wr_req_tag,
    output wire        wr_valid,
    input  wire        wr_ready,
    output wire [EXT_WR_W-1:0] wr_data,
    output wire [EXT_WR_W/8-1:0] wr_keep,
    output wire        wr_last,
    output wire [15:0] wr_tag,
    input  wire        wr_rsp_valid,
    output wire        wr_rsp_ready,
    input  wire [15:0] wr_rsp_tag,
    input  wire        wr_rsp_error,

    // LOAD_W handoff (io)
    output wire        wload_valid,
    input  wire        wload_ready,
    output wire [511:0] wload_cmd,
    output wire        wdata_valid,
    input  wire        wdata_ready,
    output wire [127:0] wdata,
    output wire        wdata_last,
    output wire        wdata_error,
    input  wire        wload_done_valid,
    output wire        wload_done_ready,
    input  wire [15:0] wload_status,

    output wire        fatal
);

    localparam DW    = MEM_DATA_W;
    localparam B     = MEM_DATA_W / 8;
    localparam BPW   = MEM_DATA_W / 128;    // 128-bit beats per scratchpad word
    localparam FIFO_D = 8;

    localparam F_ERR  = 145;
    localparam F_LAST = 144;
    localparam F_KEEP = 143;
    localparam F_DATA = 0;
    localparam FIFO_W = 146;

    // core states
    localparam C_IDLE  = 3'd0;
    localparam C_SEND  = 3'd1;
    localparam C_LOAD  = 3'd2;
    localparam C_STORE = 3'd3;
    localparam C_FIN   = 3'd4;
    localparam C_EVENT = 3'd5;

    localparam L_BEAT   = 2'd0;
    localparam L_WR     = 2'd1;
    localparam L_WR_RSP = 2'd2;
    localparam L_ROWACK = 2'd3;

    localparam S_READ = 2'd0;
    localparam S_RSP  = 2'd1;
    localparam S_PUSH = 2'd2;

    // io states
    localparam I_IDLE    = 3'd0;
    localparam I_WALLOC  = 3'd1;
    localparam I_ROWREQ  = 3'd2;
    localparam I_ROWDATA = 3'd3;
    localparam I_ROWWAIT = 3'd4;
    localparam I_ROWADV  = 3'd5;
    localparam I_FIN     = 3'd6;
    localparam I_FATAL   = 3'd7;

    localparam M_NONE = 2'd0;
    localparam M_NORM = 2'd1;   // ordinary LOAD or STORE
    localparam M_WGT  = 2'd2;   // LOAD_W

    function [5:0] pop16;
        input [15:0] v;
        integer k;
        begin
            pop16 = 6'd0;
            for (k = 0; k < 16; k = k + 1)
                pop16 = pop16 + {5'd0, v[k]};
        end
    endfunction

    // ------------------------------------------------------------------
    // cross-domain channels
    // ------------------------------------------------------------------
    wire         cmd_src_start, cmd_src_busy, cmd_src_done;
    reg  [511:0] cmd_core_data;
    wire         cmd_dst_valid, cmd_dst_ready;
    wire [511:0] cmd_dst_data;

    wire         done_src_start, done_src_busy, done_src_done;
    wire [127:0] done_payload;
    wire         done_dst_valid, done_dst_ready;
    wire [127:0] done_dst_data;

    wire         ack_src_start, ack_src_busy, ack_src_done;
    reg  [31:0]  ack_payload;
    wire         ack_dst_valid, ack_dst_ready;
    wire [31:0]  ack_dst_data;

    wire         ld_wr, ld_rd, ld_full, ld_empty;
    wire [FIFO_W-1:0] ld_din, ld_dout;
    wire         st_wr, st_rd, st_full, st_empty;
    wire [FIFO_W-1:0] st_din, st_dout;

    cdc_handshake #(.WIDTH(512)) u_cmd_mbx (
        .src_clk(clk_core), .src_rst_n(rst_core_n),
        .src_start(cmd_src_start), .src_data(cmd_core_data),
        .src_busy(cmd_src_busy), .src_done(cmd_src_done),
        .dst_clk(clk_io), .dst_rst_n(rst_io_n),
        .dst_valid(cmd_dst_valid), .dst_data(cmd_dst_data),
        .dst_ready(cmd_dst_ready)
    );

    cdc_handshake #(.WIDTH(128)) u_done_mbx (
        .src_clk(clk_io), .src_rst_n(rst_io_n),
        .src_start(done_src_start), .src_data(done_payload),
        .src_busy(done_src_busy), .src_done(done_src_done),
        .dst_clk(clk_core), .dst_rst_n(rst_core_n),
        .dst_valid(done_dst_valid), .dst_data(done_dst_data),
        .dst_ready(done_dst_ready)
    );

    cdc_handshake #(.WIDTH(32)) u_rowack_mbx (
        .src_clk(clk_core), .src_rst_n(rst_core_n),
        .src_start(ack_src_start), .src_data(ack_payload),
        .src_busy(ack_src_busy), .src_done(ack_src_done),
        .dst_clk(clk_io), .dst_rst_n(rst_io_n),
        .dst_valid(ack_dst_valid), .dst_data(ack_dst_data),
        .dst_ready(ack_dst_ready)
    );

    async_fifo #(.WIDTH(FIFO_W), .DEPTH(FIFO_D)) u_load_fifo (
        .wclk(clk_io), .wrst_n(rst_io_n), .wr(ld_wr), .din(ld_din), .full(ld_full),
        .rclk(clk_core), .rrst_n(rst_core_n), .rd(ld_rd), .dout(ld_dout), .empty(ld_empty)
    );

    async_fifo #(.WIDTH(FIFO_W), .DEPTH(FIFO_D)) u_store_fifo (
        .wclk(clk_core), .wrst_n(rst_core_n), .wr(st_wr), .din(st_din), .full(st_full),
        .rclk(clk_io), .rrst_n(rst_io_n), .rd(st_rd), .dout(st_dout), .empty(st_empty)
    );

    // ==================================================================
    // core domain
    // ==================================================================
    reg [2:0]   core_state;
    reg [1:0]   ld_state;
    reg [1:0]   st_state;

    reg [511:0] cmd_c;
    reg [7:0]   c_op, c_sub;
    reg [7:0]   c_dstbuf, c_srcbuf;
    reg [15:0]  c_job;
    reg [7:0]   c_slot;
    reg [31:0]  c_p2, c_p3, c_p5, c_p6;

    reg [15:0]  c_rows, c_row;
    reg [31:0]  c_row_bytes;
    reg [31:0]  c_woff;
    reg [31:0]  c_off;

    reg [31:0]  c_units;
    reg [15:0]  c_status;

    reg [1:0]   l_half;
    reg [DW-1:0] l_data;
    reg [B-1:0]  l_strb;
    reg          l_beat_last;
    reg          l_err;

    reg [DW-1:0] s_data;
    reg [31:0]   s_valid;
    reg [1:0]    s_beat;
    reg [1:0]    s_chunks;

    reg          cmd_start_r;
    reg          ack_start_r;

    reg [127:0]  ev_r;
    reg          ev_valid_r;

    assign task_ready  = (core_state == C_IDLE);
    assign event_valid = ev_valid_r;
    assign event_data  = ev_r;

    assign cmd_src_start  = cmd_start_r;
    assign ack_src_start  = ack_start_r;
    assign done_dst_ready = (core_state == C_FIN);

    wire rsp_error = mem_rsp[DW + 8];

    assign mem_req_valid = (core_state == C_LOAD  && ld_state == L_WR) ||
                           (core_state == C_STORE && st_state == S_READ);
    assign mem_req = (core_state == C_LOAD)
        ? {8'h00, l_strb, l_data, (c_off + c_woff), c_dstbuf, 1'b1}
        : {8'h00, {B{1'b0}}, {DW{1'b0}}, (c_off + c_woff), c_srcbuf, 1'b0};
    assign mem_rsp_ready = (core_state == C_LOAD  && ld_state == L_WR_RSP) ||
                           (core_state == C_STORE && st_state == S_RSP);

    assign ld_rd = (core_state == C_LOAD) && (ld_state == L_BEAT) && !ld_empty;

    // STORE split: keep mask and row-last flag for the current beat
    reg [31:0]  st_beat_off;
    reg [31:0]  st_rem;
    reg [15:0]  st_keep;
    reg [127:0] st_data;
    reg         st_last_flag;

    always @* begin
        st_beat_off = {30'd0, s_beat} << 4;
        st_rem      = c_row_bytes - c_woff;
        if (st_rem > B) st_rem = B;
        st_data     = s_data[128*s_beat +: 128];
        if (st_beat_off >= st_rem)
            st_keep = 16'h0000;
        else if (st_rem - st_beat_off >= 16)
            st_keep = 16'hFFFF;
        else
            st_keep = (16'h0001 << (st_rem - st_beat_off)) - 16'h0001;
        st_last_flag = (s_beat + 2'd1 == s_chunks) &&
                       (c_woff + st_rem >= c_row_bytes);
    end

    assign st_din = {1'b0, st_last_flag, st_keep, st_data};
    assign st_wr  = (core_state == C_STORE) && (st_state == S_PUSH) && !st_full;

    // ------------------------------------------------------------------
    // core FSM
    // ------------------------------------------------------------------
    always @(posedge clk_core or negedge rst_core_n) begin
        if (!rst_core_n) begin
            core_state <= C_IDLE;
            ld_state   <= L_BEAT;
            st_state   <= S_READ;
            cmd_c      <= 512'd0;
            c_op <= 8'd0; c_sub <= 8'd0;
            c_dstbuf <= 8'd0; c_srcbuf <= 8'd0;
            c_job <= 16'd0; c_slot <= 8'd0;
            c_p2 <= 0; c_p3 <= 0; c_p5 <= 0; c_p6 <= 0;
            c_rows <= 16'd0; c_row <= 16'd0;
            c_row_bytes <= 32'd0; c_woff <= 32'd0; c_off <= 32'd0;
            c_units <= 32'd0; c_status <= `NPU_ST_OK;
            l_half <= 2'd0; l_data <= {DW{1'b0}}; l_strb <= {B{1'b0}};
            l_beat_last <= 1'b0; l_err <= 1'b0;
            s_data <= {DW{1'b0}}; s_valid <= 32'd0;
            s_beat <= 2'd0; s_chunks <= 2'd0;
            cmd_start_r <= 1'b0; ack_start_r <= 1'b0;
            ack_payload <= 32'd0;
            ev_r <= 128'd0; ev_valid_r <= 1'b0;
        end else begin
            ev_valid_r  <= ev_valid_r & ~event_ready;
            cmd_start_r <= 1'b0;
            ack_start_r <= 1'b0;

            case (core_state)
            // ----------------------------------------------------------
            C_IDLE: begin
                if (task_valid && !cmd_src_busy) begin
                    cmd_c    <= task_cmd;
                    c_op     <= task_cmd[15:8];
                    c_sub    <= task_cmd[23:16];
                    c_dstbuf <= task_cmd[87:80];
                    c_srcbuf <= task_cmd[71:64];
                    c_job    <= task_cmd[47:32];
                    c_slot   <= task_cmd[103:96];
                    c_p2     <= task_cmd[223:192];
                    c_p3     <= task_cmd[255:224];
                    c_p5     <= task_cmd[319:288];
                    c_p6     <= task_cmd[351:320];
                    c_rows   <= task_cmd[271:256];
                    c_row    <= 16'd0;
                    c_row_bytes <= task_cmd[255:224];
                    c_woff   <= 32'd0;
                    c_off    <= task_cmd[223:192];
                    c_units  <= 32'd0;
                    c_status <= `NPU_ST_OK;
                    l_half <= 2'd0;
                    l_data <= {DW{1'b0}};
                    l_strb <= {B{1'b0}};
                    l_err  <= 1'b0;
                    l_beat_last <= 1'b0;
                    s_beat <= 2'd0;
                    cmd_core_data <= task_cmd;
                    cmd_start_r   <= 1'b1;
                    core_state    <= C_SEND;
                end
            end
            // ----------------------------------------------------------
            C_SEND: begin
                if (cmd_src_done) begin
                    if (c_op == `NPU_OP_LOAD && c_sub == 8'd0) begin
                        ld_state   <= L_BEAT;
                        core_state <= C_LOAD;
                    end else if (c_op == `NPU_OP_STORE) begin
                        st_state   <= S_READ;
                        core_state <= C_STORE;
                    end else begin
                        core_state <= C_FIN;
                    end
                end
            end
            // ----------------------------------------------------------
            C_LOAD: begin
                case (ld_state)
                L_BEAT: begin
                    if (!ld_empty) begin
                        l_data[l_half*128 +: 128] <= ld_dout[127:0];
                        l_strb[l_half*16 +: 16]   <= ld_dout[143:128];
                        l_err       <= l_err | ld_dout[F_ERR];
                        l_beat_last <= ld_dout[F_LAST];
                        if ((l_half == BPW-1) || ld_dout[F_LAST]) begin
                            l_half   <= 2'd0;
                            ld_state <= L_WR;
                        end else begin
                            l_half   <= l_half + 2'd1;
                        end
                    end
                end
                L_WR: begin
                    if (mem_req_ready) begin
                        c_woff   <= c_woff + B;
                        ld_state <= L_WR_RSP;
                    end
                end
                L_WR_RSP: begin
                    if (mem_rsp_valid) begin
                        if (rsp_error) l_err <= 1'b1;
                        l_data <= {DW{1'b0}};
                        l_strb <= {B{1'b0}};
                        if (l_beat_last) begin
                            ack_payload <= {l_err | rsp_error
                                            ? `NPU_ST_MEM_ACCESS : `NPU_ST_OK,
                                            c_row};
                            ack_start_r <= 1'b1;
                            l_err       <= 1'b0;
                            l_beat_last <= 1'b0;
                            ld_state    <= L_ROWACK;
                        end else begin
                            ld_state <= L_BEAT;
                        end
                    end
                end
                L_ROWACK: begin
                    if (ack_src_done) begin
                        c_row  <= c_row + 16'd1;
                        c_woff <= 32'd0;
                        c_off  <= c_p2 + (c_row + 16'd1) * c_p6;
                        if (c_row + 16'd1 == c_rows)
                            core_state <= C_FIN;
                        else
                            ld_state <= L_BEAT;
                    end
                end
                endcase
            end
            // ----------------------------------------------------------
            C_STORE: begin
                case (st_state)
                S_READ: begin
                    if (mem_req_ready)
                        st_state <= S_RSP;
                end
                S_RSP: begin
                    if (mem_rsp_valid) begin
                        s_data <= mem_rsp[DW-1:0];
                        if (rsp_error) c_status <= `NPU_ST_MEM_ACCESS;
                        s_valid  <= st_rem;
                        s_beat   <= 2'd0;
                        s_chunks <= (st_rem > 16) ? 2'd2 : 2'd1;
                        st_state <= S_PUSH;
                    end
                end
                S_PUSH: begin
                    if (!st_full) begin
                        if (s_beat + 2'd1 == s_chunks) begin
                            s_beat <= 2'd0;
                            if (c_woff + st_rem >= c_row_bytes) begin
                                c_row  <= c_row + 16'd1;
                                c_woff <= 32'd0;
                                c_off  <= c_p2 + (c_row + 16'd1) * c_p6;
                                if (c_row + 16'd1 == c_rows)
                                    core_state <= C_FIN;
                                else
                                    st_state <= S_READ;
                            end else begin
                                c_woff   <= c_woff + B;
                                st_state <= S_READ;
                            end
                        end else begin
                            s_beat <= s_beat + 2'd1;
                        end
                    end
                end
                endcase
            end
            // ----------------------------------------------------------
            C_FIN: begin
                if (done_dst_valid) begin
                    c_units  <= done_dst_data[31:0];
                    c_status <= done_dst_data[47:32];
                    ev_r <= {32'd0,
                             done_dst_data[31:0],
                             16'd0,
                             (c_status != `NPU_ST_OK) ? c_status
                                                      : done_dst_data[47:32],
                             c_op, c_slot, c_job};
                    ev_valid_r <= 1'b1;
                    core_state <= C_EVENT;
                end
            end
            // ----------------------------------------------------------
            C_EVENT: begin
                if (ev_valid_r && event_ready) begin
                    ev_valid_r <= 1'b0;
                    core_state <= C_IDLE;
                end
            end
            default: core_state <= C_IDLE;
            endcase
        end
    end

    // ==================================================================
    // io domain
    // ==================================================================
    reg [2:0]   io_state;
    reg [511:0] cmd_i;
    reg [7:0]   i_op, i_sub;
    reg [1:0]   i_mode;
    reg [63:0]  i_addr;
    reg [15:0]  i_rows, i_row;
    reg [31:0]  i_row_bytes;
    reg [31:0]  i_units;
    reg [15:0]  i_status;
    reg         i_err_mode;
    reg         i_synth;
    reg [127:0] done_payload_r;
    reg         done_start_r;
    reg         done_sent;
    reg         i_fatal;

    wire is_load  = (i_op == `NPU_OP_LOAD) && (i_sub == 8'd0);
    wire is_store = (i_op == `NPU_OP_STORE);
    wire is_wgt   = (i_op == `NPU_OP_LOAD) && (i_sub == 8'd1);

    assign cmd_dst_ready = (io_state == I_IDLE) && cmd_dst_valid;
    assign done_src_start = done_start_r;
    assign done_payload   = done_payload_r;
    assign ack_dst_ready  = (io_state == I_ROWWAIT) && ack_dst_valid;

    assign rd_req_valid = (io_state == I_ROWREQ) && (is_load || is_wgt) &&
                          !(is_load && i_err_mode);
    assign rd_req_addr  = i_addr;
    assign rd_req_bytes = i_row_bytes;
    assign rd_req_tag   = i_row;

    assign wr_req_valid = (io_state == I_ROWREQ) && is_store && !i_err_mode;
    assign wr_req_addr  = i_addr;
    assign wr_req_bytes = i_row_bytes;
    assign wr_req_tag   = i_row;

    assign wload_valid = (io_state == I_WALLOC);
    assign wload_cmd   = cmd_i;

    // LOAD / LOAD_W input path
    assign ld_din = i_synth ? {1'b1, 1'b1, 16'h0000, 128'd0}
                            : {rd_error, rd_last, rd_keep, rd_data};
    assign ld_wr  = (io_state == I_ROWDATA) && is_load && !ld_full &&
                    (i_synth || rd_valid);
    assign rd_ready = (io_state == I_ROWDATA) &&
                      (is_wgt ? wdata_ready :
                       (is_load ? (!ld_full && !i_synth) : 1'b0));

    assign wdata_valid = (io_state == I_ROWDATA) && is_wgt && rd_valid;
    assign wdata       = rd_data;
    assign wdata_last  = rd_last;
    assign wdata_error = rd_error;

    // STORE output path
    assign wr_valid = (io_state == I_ROWDATA) && is_store && !st_empty &&
                      !i_err_mode;
    assign wr_data  = st_dout[127:0];
    assign wr_keep  = st_dout[143:128];
    assign wr_last  = st_dout[F_LAST];
    assign wr_tag   = i_row;
    assign st_rd    = (io_state == I_ROWDATA) && is_store && !st_empty &&
                      (i_err_mode || wr_ready);

    assign wr_rsp_ready     = (io_state == I_ROWWAIT) && is_store;
    assign wload_done_ready = (io_state == I_ROWWAIT) && is_wgt;

    always @(posedge clk_io or negedge rst_io_n) begin
        if (!rst_io_n) begin
            io_state <= I_IDLE;
            cmd_i <= 512'd0;
            i_op <= 8'd0; i_sub <= 8'd0; i_mode <= M_NONE;
            i_addr <= 64'd0;
            i_rows <= 16'd0; i_row <= 16'd0; i_row_bytes <= 32'd0;
            i_units <= 32'd0; i_status <= `NPU_ST_OK;
            i_err_mode <= 1'b0; i_synth <= 1'b0;
            done_payload_r <= 128'd0; done_start_r <= 1'b0;
            done_sent <= 1'b0;
            i_fatal <= 1'b0;
        end else begin
            done_start_r <= 1'b0;

            case (io_state)
            // ----------------------------------------------------------
            I_IDLE: begin
                done_sent  <= 1'b0;
                i_err_mode <= 1'b0;
                i_synth    <= 1'b0;
                i_units    <= 32'd0;
                i_status   <= `NPU_ST_OK;
                if (cmd_dst_valid) begin
                    cmd_i <= cmd_dst_data;
                    i_op  <= cmd_dst_data[15:8];
                    i_sub <= cmd_dst_data[23:16];
                    // LOAD/STORE carry one 64-bit address split over P0/P1.
                    // LOAD_W instead carries two independent 32-bit starts:
                    // P0 = weight region, P1 = scale region, which is adjacent
                    // (P1 = P0 + W_BYTES).  The loader takes one contiguous
                    // stream of P4 bytes from P0 and splits it internally, so
                    // the bus address must be {32'd0, P0} -- not {P1, P0}.
                    i_addr <= ((cmd_dst_data[15:8] == `NPU_OP_LOAD) &&
                               (cmd_dst_data[23:16] == 8'd1))
                              ? {32'd0, cmd_dst_data[159:128]}
                              : {cmd_dst_data[191:160], cmd_dst_data[159:128]};
                    i_row  <= 16'd0;
                    if (cmd_dst_data[15:8] == `NPU_OP_LOAD &&
                        cmd_dst_data[23:16] == 8'd1) begin
                        i_mode      <= M_WGT;
                        i_rows      <= 16'd1;
                        i_row_bytes <= cmd_dst_data[287:256];
                        io_state    <= I_WALLOC;
                    end else if (cmd_dst_data[15:8] == `NPU_OP_LOAD ||
                                 cmd_dst_data[15:8] == `NPU_OP_STORE) begin
                        i_mode      <= M_NORM;
                        i_rows      <= cmd_dst_data[271:256];
                        i_row_bytes <= cmd_dst_data[255:224];
                        io_state    <= I_ROWREQ;
                    end else begin
                        i_status <= `NPU_ST_BAD_OPCODE;
                        io_state <= I_FIN;
                    end
                end
            end
            // ----------------------------------------------------------
            I_WALLOC: begin
                if (wload_ready)
                    io_state <= I_ROWREQ;
            end
            // ----------------------------------------------------------
            I_ROWREQ: begin
                if (i_err_mode) begin
                    if (is_load) i_synth <= 1'b1;
                    io_state <= I_ROWDATA;
                end else if (is_store) begin
                    if (wr_req_ready) io_state <= I_ROWDATA;
                end else begin
                    if (rd_req_ready) io_state <= I_ROWDATA;
                end
            end
            // ----------------------------------------------------------
            I_ROWDATA: begin
                if (is_wgt) begin
                    if (rd_valid && rd_ready) begin
                        i_units <= i_units + {26'd0, pop16(rd_keep)};
                        if (rd_tag != i_row) begin
                            i_status <= `NPU_ST_INTERNAL;
                            i_fatal  <= 1'b1;
                        end
                        if (rd_error) i_status <= `NPU_ST_DMA_BUS;
                        if (rd_last)  io_state <= I_ROWWAIT;
                    end
                end else if (is_load) begin
                    if (i_synth) begin
                        if (!ld_full) begin
                            i_synth  <= 1'b0;
                            io_state <= I_ROWWAIT;
                        end
                    end else if (rd_valid && rd_ready) begin
                        i_units <= i_units + {26'd0, pop16(rd_keep)};
                        if (rd_tag != i_row) begin
                            i_status <= `NPU_ST_INTERNAL;
                            i_fatal  <= 1'b1;
                        end
                        if (rd_error) begin
                            i_status   <= `NPU_ST_DMA_BUS;
                            i_err_mode <= 1'b1;
                        end
                        if (rd_last) io_state <= I_ROWWAIT;
                    end
                end else begin
                    if (st_rd) begin
                        if (!i_err_mode)
                            i_units <= i_units + {26'd0, pop16(st_dout[143:128])};
                        if (st_dout[F_LAST])
                            io_state <= I_ROWWAIT;
                    end
                end
            end
            // ----------------------------------------------------------
            I_ROWWAIT: begin
                if (is_wgt) begin
                    if (wload_done_valid) begin
                        if (wload_status != `NPU_ST_OK)
                            i_status <= wload_status;
                        io_state <= I_FIN;
                    end
                end else if (is_load) begin
                    if (ack_dst_valid) begin
                        if (ack_dst_data[15:0] != i_row)
                            i_status <= `NPU_ST_INTERNAL;
                        else if (ack_dst_data[31:16] != `NPU_ST_OK)
                            i_status <= ack_dst_data[31:16];
                        if (i_row + 16'd1 == i_rows)
                            io_state <= I_FIN;
                        else
                            io_state <= I_ROWADV;
                    end
                end else begin
                    if (i_err_mode) begin
                        if (i_row + 16'd1 == i_rows)
                            io_state <= I_FIN;
                        else
                            io_state <= I_ROWADV;
                    end else if (wr_rsp_valid) begin
                        if (wr_rsp_tag != i_row)
                            i_status <= `NPU_ST_INTERNAL;
                        if (wr_rsp_error) begin
                            i_status   <= `NPU_ST_DMA_BUS;
                            i_err_mode <= 1'b1;
                        end
                        if (i_row + 16'd1 == i_rows)
                            io_state <= I_FIN;
                        else
                            io_state <= I_ROWADV;
                    end
                end
            end
            // ----------------------------------------------------------
            I_ROWADV: begin
                i_addr   <= i_addr + {32'd0, cmd_i[319:288]};
                i_row    <= i_row + 16'd1;
                io_state <= I_ROWREQ;
            end
            // ----------------------------------------------------------
            I_FIN: begin
                if (!done_sent) begin
                    // layout matches the core-side decode in C_FIN:
                    // units[31:0] first, then status[15:0]
                    done_payload_r <= {80'd0, i_status, i_units};
                    done_start_r   <= 1'b1;
                    done_sent      <= 1'b1;
                end else if (done_src_done) begin
                    io_state <= i_fatal ? I_FATAL : I_IDLE;
                end
            end
            // ----------------------------------------------------------
            I_FATAL: io_state <= I_FATAL;
            default: io_state <= I_IDLE;
            endcase
        end
    end

    // fatal is exported in the core domain
    (* ASYNC_REG = "TRUE" *) reg [1:0] fatal_sync;
    always @(posedge clk_core or negedge rst_core_n) begin
        if (!rst_core_n) fatal_sync <= 2'b00;
        else             fatal_sync <= {fatal_sync[0], i_fatal};
    end
    assign fatal = fatal_sync[1];

endmodule

`default_nettype wire

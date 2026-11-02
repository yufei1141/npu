//=============================================================================
// b_frame_reader : scratchpad w16 B frame -> gemv_stream B beats
//
//   The weight path reaches the B port through a weight bank and the
//   io-domain reader, which only the loader can fill.  Attention needs a B
//   operand that is itself an activation (K for QK^T, V for P*V), so this
//   reader streams an already packed w16 frame straight out of a scratchpad
//   buffer:
//
//     group j, step s  ->  32 B at  B_base + j*B_stride + 32*s
//
//   which is exactly the frame LAYOUT.PACK_W16 writes (doc 16).
//
//   One beat carries n_grp groups of 256 bits, in the same (tile, step) order
//   the weight reader produces, so gemv_stream and the datapath need no
//   change.  w16 has no scales, so the top 256 bits of the beat stay zero and
//   the groups past n_grp are cleared rather than left stale.
//
//   Client rule: one read in flight, one beat held until accepted.
//=============================================================================
//=============================================================================
// 中文说明
//
//   把 scratchpad 里已打包好的 w16 B 帧读成 gemv_stream 的 B beat。
//
//   为什么需要这条路径：正常权重是从 weight bank 经 io 域 reader 送进 B 口的，
//   而那条通路只有 loader 能填。但 attention 里的 B 操作数本身就是激活值
//   （QK^T 的 K、P·V 的 V），它们不来自权重 bank，所以这里直接从 scratchpad
//   的普通 buffer 里把 w16 帧流式读出。
//
//   寻址：第 j 组、第 s 步取 B_base + j*B_stride + 32*s 开始的 32 字节。这
//   正是 LAYOUT.PACK_W16 写出的帧布局（见 doc 16），因此上游不需要为"权重
//   还是激活"准备两套格式。
//
//   beat 组织：一拍承载 n_grp 个 256 位组，顺序与 weight reader 产生的
//   (tile, step) 完全一致，所以 gemv_stream 和后面的数据通路一行都不用改。
//   w16 没有 scale，因此 beat 的最高 256 位恒为 0；n_grp 之外的组被显式清零，
//   而不是留着上一拍的残留值——残留会直接进累加器。
//
//   客户端约束：一个读请求在途，一个 beat 保持到被接收为止
//   （memory_subsystem 的每客户端单在途规则）。
//=============================================================================
`timescale 1ns/1ps
`default_nettype none

`include "npu_defs.vh"

module b_frame_reader #(
    parameter MEM_DATA_W = 256
)(
    input  wire        clk_core,
    input  wire        rst_core_n,

    // job configuration: the MATMUL command itself
    input  wire        cfg_valid,
    output wire        cfg_ready,
    input  wire [511:0] cfg_cmd,

    // scratchpad read channel (arbitrated with the A packer on client MPU_A)
    output wire        mem_req_valid,
    input  wire        mem_req_ready,
    output wire [`NPU_REQ_W(MEM_DATA_W, MEM_DATA_W/8)-1:0] mem_req,
    input  wire        mem_rsp_valid,
    output wire        mem_rsp_ready,
    input  wire [`NPU_RSP_W(MEM_DATA_W)-1:0] mem_rsp,

    // B beat stream
    output wire        b_valid,
    input  wire        b_ready,
    output wire [2047:0] b_data,

    output wire        b_done,
    output wire [15:0] error
);

    localparam DW    = MEM_DATA_W;
    localparam BW    = MEM_DATA_W / 8;
    localparam REQ_W = `NPU_REQ_W(MEM_DATA_W, MEM_DATA_W/8);
    localparam RSP_W = `NPU_RSP_W(MEM_DATA_W);

    localparam R_IDLE = 3'd0;
    localparam R_NEXT = 3'd1;
    localparam R_REQ  = 3'd2;
    localparam R_WAIT = 3'd3;
    localparam R_EMIT = 3'd4;
    localparam R_DONE = 3'd5;
    localparam R_ERR  = 3'd6;

    reg [2:0]  state;
    reg [7:0]  buf_r;
    reg [31:0] base_r, stride_r;
    reg [3:0]  ngrp_r;
    reg [15:0] ksteps_r, tiles_r;
    reg [15:0] t_r, s_r;
    reg [3:0]  g_r;
    reg [2047:0] b_buf;
    reg [15:0] error_r;
    reg        done_r;

    // Tiles are consecutive group blocks of one PACK_W16 frame, so tile t
    // starts at group t*n_grp; that is what makes `base + t*g*stride` the
    // next tile's frame and keeps a multi-tile B a single flat stream.
    wire [15:0] g_abs = ({12'd0, ngrp_r} * t_r) + {12'd0, g_r};
    wire [63:0] off64 = ({32'd0, stride_r} * {48'd0, g_abs}) +
                        ({43'd0, s_r} << 5);
    wire [63:0] addr64 = {32'd0, base_r} + off64;

    wire [DW-1:0] rsp_data   = mem_rsp[DW-1:0];
    wire          rsp_err    = mem_rsp[DW+8];
    wire [7:0]    rsp_reason = mem_rsp[DW+16:DW+9];
    wire          rsp_bad    = rsp_err || (rsp_reason != `NPU_MEMR_OK);

    assign mem_req = {8'h00, {BW{1'b0}}, {DW{1'b0}}, addr64[31:0],
                      buf_r, 1'b0};
    assign mem_req_valid = (state == R_REQ);
    assign mem_rsp_ready = (state == R_WAIT);

    assign b_valid = (state == R_EMIT);
    assign b_data  = b_buf;
    assign b_done  = done_r;
    assign error   = error_r;
    assign cfg_ready = (state == R_IDLE);

    always @(posedge clk_core or negedge rst_core_n) begin
        if (!rst_core_n) begin
            state    <= R_IDLE;
            buf_r    <= 8'd0;
            base_r   <= 32'd0;
            stride_r <= 32'd0;
            ngrp_r   <= 4'd0;
            ksteps_r <= 16'd0;
            tiles_r  <= 16'd0;
            t_r      <= 16'd0;
            s_r      <= 16'd0;
            g_r      <= 4'd0;
            b_buf    <= 2048'd0;
            error_r  <= `NPU_ST_OK;
            done_r   <= 1'b0;
        end else begin
            case (state)
            R_IDLE: begin
                done_r <= 1'b0;
                if (cfg_valid) begin
                    buf_r    <= cfg_cmd[79:72];
                    base_r   <= cfg_cmd[351:320];      // P6
                    stride_r <= cfg_cmd[447:416];      // P9
                    ngrp_r   <= cfg_cmd[139:136];
                    ksteps_r <= cfg_cmd[191:176];
                    tiles_r  <= cfg_cmd[175:160];
                    t_r      <= 16'd0;
                    s_r      <= 16'd0;
                    g_r      <= 4'd0;
                    b_buf    <= 2048'd0;
                    error_r  <= `NPU_ST_OK;
                    state    <= R_NEXT;
                end
            end
            R_NEXT: begin
                if (t_r == tiles_r) begin
                    done_r <= 1'b1;
                    state  <= R_DONE;
                end else if (g_r == ngrp_r) begin
                    state <= R_EMIT;
                end else if (addr64[63:32] != 32'd0) begin
                    error_r <= `NPU_ST_RANGE;
                    state   <= R_ERR;
                end else begin
                    state <= R_REQ;
                end
            end
            R_REQ: begin
                if (mem_req_ready) state <= R_WAIT;
            end
            R_WAIT: begin
                if (mem_rsp_valid) begin
                    if (rsp_bad) begin
                        error_r <= `NPU_ST_MEM_ACCESS;
                        state   <= R_ERR;
                    end else begin
                        b_buf[256*g_r +: 256] <= rsp_data;
                        g_r   <= g_r + 4'd1;
                        state <= R_NEXT;
                    end
                end
            end
            R_EMIT: begin
                if (b_ready) begin
                    b_buf <= 2048'd0;
                    g_r   <= 4'd0;
                    if (s_r + 16'd1 == ksteps_r) begin
                        s_r <= 16'd0;
                        t_r <= t_r + 16'd1;
                    end else begin
                        s_r <= s_r + 16'd1;
                    end
                    state <= R_NEXT;
                end
            end
            R_DONE: begin
                if (cfg_valid) state <= R_IDLE;   // re-arm on the next job
            end
            R_ERR: begin
                done_r <= 1'b1;
                if (cfg_valid) state <= R_IDLE;
            end
            default: state <= R_ERR;
            endcase
        end
    end

endmodule

`default_nettype wire

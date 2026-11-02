//=============================================================================
// weight_reader : committed weight tile -> B beats for the MPU (core domain)
//
//   Follows docs/npu_integration/07_weight_buffer.md sections 5 and 6.
//
//   The reader matches a select request against both banks' descriptors,
//   claims the matching bank, and streams one 2304-bit B beat per K-step:
//   8 groups x 256 bits of weights followed by 8 x 32 bits of E8M0 scale.
//   Groups beyond n_grp are driven as zero weights with unit scale.  After
//   the last beat has been accepted the bank is released and tile_done is
//   reported only once the full release handshake has completed.
//
//   This is the conservative single-outstanding-read version: one RAM read
//   in flight and one B holding register, so it does not sustain one beat per
//   core clock.  The pipelined version described in the weight-path document
//   is a separate step.
//=============================================================================
//=============================================================================
// 中文说明
//
//   把已经提交的权重 tile 读成 MPU 需要的 B beat（core 域）。实现依据
//   docs/npu_integration/07_weight_buffer.md 第 5、6 节。
//
//   工作流程：
//     - 用 select 请求去匹配两个 bank 各自的描述符，认领匹配的那个 bank；
//     - 每个 K 步流出一个 2304 位的 B beat：8 组 × 256 位权重，后面跟 8 × 32
//       位的 E8M0 scale；
//     - n_grp 之外的组输出"零权重 + 单位 scale"，而不是留着上一拍的残留——
//       因为 tensorcore 会把这些组一并算进累加；
//     - 最后一拍被接收之后释放 bank，**tile_done 要等整个释放握手完成才上报**。
//       这个顺序保证了下一个 tile 开始选 bank 时，前一个 tile 确实已经放手。
//
//   吞吐定位：这是保守的"单次未完成读"版本——一个 RAM 读在途、一个 B 保持
//   寄存器，因此无法做到每核心时钟一拍。权重通路文档里描述的流水化版本是
//   单独的一步工作，不是本模块的隐藏行为。
//=============================================================================
`timescale 1ns/1ps
`default_nettype none

`include "npu_defs.vh"

module weight_reader #(
    parameter GROUPS  = 8,
    parameter K_STEPS = 256
)(
    input  wire        clk_core,
    input  wire        rst_core_n,

    // MPU tile request
    input  wire        select_valid,
    output wire        select_ready,
    input  wire [15:0] select_job,
    input  wire [15:0] select_tile,
    input  wire [1:0]  select_mode,
    input  wire [3:0]  select_g,
    input  wire [15:0] select_s,

    // B beat stream
    output wire        b_valid,
    input  wire        b_ready,
    output wire [2303:0] b_data,
    output wire [15:0] b_step,
    output wire        b_last,

    output wire        tile_done_valid,
    input  wire        tile_done_ready,
    output wire [15:0] tile_done_status,

    // tile0 query from the scheduler
    input  wire [15:0] query_job,
    input  wire [15:0] query_tile,
    input  wire [1:0]  query_mode,
    input  wire [3:0]  query_g,
    input  wire [15:0] query_s,
    output wire        query_match,
    output wire        query_fault,

    // bank interfaces (2 banks)
    input  wire [1:0]         bank_desc_valid,
    input  wire [2*`NPU_WTILE_W-1:0] bank_desc,
    output wire [1:0]         bank_accept,
    output wire [1:0]         bank_release,
    input  wire [1:0]         bank_release_done,
    input  wire [31:0]        bank_release_status,

    output wire [2*GROUPS-1:0] bank_rd_en,
    output wire [$clog2(K_STEPS)-1:0] bank_rd_addr,
    input  wire [1:0]          bank_rd_valid,
    input  wire [2*GROUPS*256-1:0] bank_wlo_hi,
    input  wire [2*GROUPS*128-1:0] bank_scale,

    output wire        fatal
);

    localparam AW = $clog2(K_STEPS);

    localparam R_IDLE     = 3'd0;
    localparam R_ISSUE    = 3'd1;
    localparam R_WAIT     = 3'd2;
    localparam R_HOLD     = 3'd3;
    localparam R_WAIT_REL = 3'd4;
    localparam R_DONE     = 3'd5;
    localparam R_FAULT    = 3'd6;

    reg [2:0]  state;
    reg        sel;
    reg [1:0]  mode_r;
    reg [3:0]  g_r;
    reg [15:0] s_r;
    reg [15:0] step_r;
    reg [15:0] status_r;
    reg        fatal_r;
    reg        rel_pulse;
    reg [1:0]  accept_pulse;

    reg [GROUPS*256-1:0] w_lat;
    reg [GROUPS*128-1:0] sc_lat;

    assign fatal = fatal_r;

    // ------------------------------------------------------------------
    // descriptor matching
    //
    // Descriptors are copied into named wires first: Verilog-2001 does not
    // allow a bit-select applied to a part-select expression, so the field
    // macros cannot be applied to bank_desc[... +: W] directly.
    // ------------------------------------------------------------------
    wire [1:0] valid_mask = bank_desc_valid;

    wire [`NPU_WTILE_W-1:0] desc0 = bank_desc[`NPU_WTILE_W-1:0];
    wire [`NPU_WTILE_W-1:0] desc1 = bank_desc[2*`NPU_WTILE_W-1:`NPU_WTILE_W];

    function same_cfg;
        input [`NPU_WTILE_W-1:0] d;
        input [15:0] job;
        input [15:0] tile;
        input [1:0]  mode;
        input [3:0]  grp;
        input [15:0] steps;
        begin
            same_cfg = (`NPU_WTILE_JOB(d)   == job)  &&
                       (`NPU_WTILE_INDEX(d) == tile) &&
                       (`NPU_WTILE_MODE(d)  == mode) &&
                       (`NPU_WTILE_NGRP(d)  == grp)  &&
                       (`NPU_WTILE_KSTEPS(d) == steps);
        end
    endfunction

    wire id0 = (`NPU_WTILE_JOB(desc0) == select_job) &&
               (`NPU_WTILE_INDEX(desc0) == select_tile);
    wire id1 = (`NPU_WTILE_JOB(desc1) == select_job) &&
               (`NPU_WTILE_INDEX(desc1) == select_tile);

    wire [1:0] serve_mask = {valid_mask[1] && id1 &&
                             same_cfg(desc1, select_job, select_tile,
                                      select_mode, select_g, select_s),
                             valid_mask[0] && id0 &&
                             same_cfg(desc0, select_job, select_tile,
                                      select_mode, select_g, select_s)};

    wire qid0 = (`NPU_WTILE_JOB(desc0) == query_job) &&
                (`NPU_WTILE_INDEX(desc0) == query_tile);
    wire qid1 = (`NPU_WTILE_JOB(desc1) == query_job) &&
                (`NPU_WTILE_INDEX(desc1) == query_tile);

    wire [1:0] q_match = {valid_mask[1] && qid1 &&
                          same_cfg(desc1, query_job, query_tile,
                                   query_mode, query_g, query_s),
                          valid_mask[0] && qid0 &&
                          same_cfg(desc0, query_job, query_tile,
                                   query_mode, query_g, query_s)};

    assign query_match = |q_match;
    assign query_fault = ((valid_mask[1] && qid1) ||
                          (valid_mask[0] && qid0)) && !(|q_match);

    // ------------------------------------------------------------------
    // B beat assembly from the latched RAM response
    // ------------------------------------------------------------------
    reg [2303:0] b_data_c;
    integer j;
    always @* begin
        b_data_c = 2304'd0;
        for (j = 0; j < GROUPS; j = j + 1) begin
            if (j < g_r) begin
                b_data_c[256*j +: 256] = w_lat[j*256 +: 256];
                b_data_c[2048+32*j +: 32] =
                    mode_r[0] ? `NPU_E8M0_ONE
                              : sc_lat[j*128 + 32*step_r[1:0] +: 32];
            end else begin
                b_data_c[2048+32*j +: 32] = `NPU_E8M0_ONE;
            end
        end
    end

    assign b_data = b_data_c;
    assign b_valid = (state == R_HOLD);
    assign b_step  = step_r;
    assign b_last  = (step_r + 16'd1 == s_r);

    assign tile_done_valid  = (state == R_DONE);
    assign tile_done_status = status_r;

    assign select_ready = (state == R_IDLE) && (|serve_mask) && !fatal_r;

    // ------------------------------------------------------------------
    // read / release drive
    // ------------------------------------------------------------------
    wire [2*GROUPS-1:0] rd_mask_g;
    genvar gi;
    generate
        for (gi = 0; gi < GROUPS; gi = gi + 1) begin : g_rd
            assign rd_mask_g[gi]         = (gi < g_r);
            assign rd_mask_g[GROUPS+gi]  = (gi < g_r);
        end
    endgenerate

    // The selected half must be placed explicitly: both operands of a
    // ternary are GROUPS wide, so a bare select would zero-extend into the
    // low half of bank_rd_en regardless of which bank was chosen.
    wire [GROUPS-1:0] rd_mask_half = sel ? rd_mask_g[2*GROUPS-1:GROUPS]
                                         : rd_mask_g[GROUPS-1:0];

    assign bank_rd_en   = (state == R_ISSUE)
                          ? (sel ? {rd_mask_half, {GROUPS{1'b0}}}
                                 : {{GROUPS{1'b0}}, rd_mask_half})
                          : {2*GROUPS{1'b0}};
    assign bank_rd_addr = step_r[AW-1:0];

    assign bank_accept  = accept_pulse;
    assign bank_release = rel_pulse ? (sel ? 2'b10 : 2'b01) : 2'b00;

    // ------------------------------------------------------------------
    // FSM
    // ------------------------------------------------------------------
    always @(posedge clk_core or negedge rst_core_n) begin
        if (!rst_core_n) begin
            state       <= R_IDLE;
            sel         <= 1'b0;
            mode_r      <= 2'd0;
            g_r         <= 4'd0;
            s_r         <= 16'd0;
            step_r      <= 16'd0;
            status_r    <= `NPU_ST_OK;
            fatal_r     <= 1'b0;
            rel_pulse   <= 1'b0;
            accept_pulse <= 2'b00;
            w_lat       <= {(GROUPS*256){1'b0}};
            sc_lat      <= {(GROUPS*128){1'b0}};
        end else begin
            rel_pulse    <= 1'b0;
            accept_pulse <= 2'b00;

            case (state)
            R_IDLE: begin
                if (select_valid && select_ready) begin
                    sel      <= serve_mask[0] ? 1'b0 : 1'b1;
                    accept_pulse <= serve_mask[0] ? 2'b01 : 2'b10;
                    mode_r   <= select_mode;
                    g_r      <= select_g;
                    s_r      <= select_s;
                    step_r   <= 16'd0;
                    status_r <= `NPU_ST_OK;
                    state    <= R_ISSUE;
                end
            end
            R_ISSUE: begin
                state <= R_WAIT;
            end
            R_WAIT: begin
                if (bank_rd_valid[sel]) begin
                    w_lat  <= sel ? bank_wlo_hi[2*GROUPS*256-1:GROUPS*256]
                                  : bank_wlo_hi[GROUPS*256-1:0];
                    sc_lat <= sel ? bank_scale[2*GROUPS*128-1:GROUPS*128]
                                  : bank_scale[GROUPS*128-1:0];
                    state  <= R_HOLD;
                end
            end
            R_HOLD: begin
                if (b_valid && b_ready) begin
                    if (b_last) begin
                        rel_pulse <= 1'b1;
                        state     <= R_WAIT_REL;
                    end else begin
                        step_r <= step_r + 16'd1;
                        state  <= R_ISSUE;
                    end
                end
            end
            R_WAIT_REL: begin
                if (bank_release_done[sel]) begin
                    status_r <= sel ? bank_release_status[31:16]
                                    : bank_release_status[15:0];
                    if ((sel ? bank_release_status[31:16]
                             : bank_release_status[15:0]) != `NPU_ST_OK) begin
                        fatal_r <= 1'b1;
                        state   <= R_FAULT;
                    end else begin
                        state <= R_DONE;
                    end
                end
            end
            R_DONE: begin
                if (tile_done_ready) state <= R_IDLE;
            end
            R_FAULT: begin
                // sticky until reset; the platform must coordinate recovery
                state <= R_FAULT;
            end
            default: state <= R_IDLE;
            endcase
        end
    end

endmodule

`default_nettype wire

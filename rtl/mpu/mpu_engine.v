//=============================================================================
// mpu_engine : MATMUL task adapter around the existing gemv_stream core
//
//   Follows docs/npu_integration/08_mpu_engine.md.  The core is used as-is:
//   its A/B FIFOs, sequencer and accumulator are not modified, and the
//   adapter never restarts it on !busy.
//
//   task_adapter      reserve results (wb_job), select the first weight tile,
//                     start the core once, feed A/B, count receipts
//   activation_packer scratchpad -> A beats (separate module)
//   B tile controller one weight_reader select per tile, next select only
//                     after tile_done has confirmed the bank release
//   result_capture    counts acc_valid pulses (the pulse itself is forwarded
//                     unchanged to result_writeback at the top level)
//
//   Completion needs all of: a_accepted = b_accepted = tiles*k_steps,
//   results_captured = tiles, core done, every weight bank released and the
//   writeback summary for all tiles.  The event is held until accepted.
//=============================================================================
`timescale 1ns/1ps
`default_nettype none

`include "npu_defs.vh"

module mpu_engine #(
    parameter MEM_DATA_W   = 256,
    parameter RESULT_SLOTS = 4,
    parameter WBUF_K_STEPS = 256
)(
    input  wire        clk_core,
    input  wire        rst_core_n,

    // scheduler task input
    input  wire        task_valid,
    output wire        task_ready,
    input  wire [511:0] task_cmd,
    output wire        event_valid,
    input  wire        event_ready,
    output wire [127:0] event_data,

    // scratchpad A reads (client MPU_A)
    output wire        mem_req_valid,
    input  wire        mem_req_ready,
    output wire [`NPU_REQ_W(MEM_DATA_W, MEM_DATA_W/8)-1:0] mem_req,
    input  wire        mem_rsp_valid,
    output wire        mem_rsp_ready,
    input  wire [`NPU_RSP_W(MEM_DATA_W)-1:0] mem_rsp,

    // weight tile control / B stream
    output wire        w_select_valid,
    input  wire        w_select_ready,
    output wire [15:0] w_select_job,
    output wire [15:0] w_select_tile,
    output wire [1:0]  w_select_mode,
    output wire [3:0]  w_select_g,
    output wire [15:0] w_select_s,
    input  wire        w_b_valid,
    output wire        w_b_ready,
    input  wire [2303:0] w_b_data,
    input  wire        w_tile_done_valid,
    output wire        w_tile_done_ready,
    input  wire [15:0] w_tile_done_status,

    // result slot reservation / writeback summary
    output wire        wb_job_valid,
    input  wire        wb_job_ready,
    output wire [511:0] wb_job_cmd,
    input  wire        wb_done_valid,
    output wire        wb_done_ready,
    input  wire [15:0] wb_done_status,
    input  wire [31:0] wb_done_tiles,

    // core result pulse, forwarded unchanged to result_writeback
    output wire        acc_valid,
    output wire [4095:0] acc_data,

    output wire        fatal
);

    localparam REQ_W = `NPU_REQ_W(MEM_DATA_W, MEM_DATA_W/8);
    localparam RSP_W = `NPU_RSP_W(MEM_DATA_W);

    localparam M_IDLE      = 4'd0;
    localparam M_RESERVE   = 4'd1;
    localparam M_BSEL      = 4'd2;
    localparam M_START     = 4'd3;
    localparam M_FEED      = 4'd4;
    localparam M_DRAIN     = 4'd5;
    localparam M_WRITEBACK = 4'd6;
    localparam M_COMPLETE  = 4'd7;
    localparam M_FAULT     = 4'd8;

    reg [3:0]  state;

    // ------------------------------------------------------------------
    // command decode
    // ------------------------------------------------------------------
    wire [7:0]  d_opcode  = task_cmd[15:8];
    wire [7:0]  d_subop   = task_cmd[23:16];
    wire [15:0] d_job     = task_cmd[47:32];
    wire [7:0]  d_srcbuf  = task_cmd[71:64];
    wire [7:0]  d_dstbuf  = task_cmd[87:80];
    wire [7:0]  d_fmt     = task_cmd[95:88];
    wire [7:0]  d_slot    = task_cmd[103:96];

    wire [2:0]  d_mlen    = task_cmd[130:128];
    wire [3:0]  d_ngrp    = task_cmd[139:136];
    wire [1:0]  d_mode    = task_cmd[145:144];
    wire [15:0] d_tiles   = task_cmd[175:160];
    wire [15:0] d_ksteps  = task_cmd[191:176];
    wire [31:0] d_kvalid  = task_cmd[223:192];
    wire [31:0] d_nvalid  = task_cmd[255:224];
    wire [31:0] d_abase   = task_cmd[287:256];
    wire [31:0] d_astride = task_cmd[319:288];
    wire [31:0] d_atstr   = task_cmd[351:320];

    // K elements per step: 16 for W4, 4 for W16
    wire [31:0] d_kstep   = d_mode[0] ? 32'd4 : 32'd16;
    wire [31:0] d_ceil_k  = (d_kvalid + d_kstep - 32'd1) / d_kstep;

    wire d_op_ok    = (d_opcode == `NPU_OP_MATMUL) && (d_subop == 8'd0);
    wire d_shape_ok = (d_mlen >= 3'd1) && (d_mlen <= 3'd4) &&
                      (d_ngrp >= 4'd1) && (d_ngrp <= 4'd8) &&
                      (d_tiles >= 16'd1) && (d_tiles <= RESULT_SLOTS) &&
                      (d_ksteps >= 16'd1) && (d_ksteps <= WBUF_K_STEPS);
    wire d_k_ok     = (d_kvalid != 32'd0) && (d_ceil_k == {16'd0, d_ksteps});
    // N_valid must fit the declared tiles and use all but the last
    wire [31:0] d_ngrp32 = {28'd0, d_ngrp};
    wire d_n_ok     = (d_nvalid != 32'd0) &&
                      (d_nvalid <= (d_ngrp32 * 4 * {16'd0, d_tiles})) &&
                      (({16'd0, d_tiles} == 16'd1) ||
                       (d_nvalid > (d_ngrp32 * 4 * ({16'd0, d_tiles} - 32'd1))));
    wire d_abuf_ok  = (d_srcbuf != d_dstbuf);
    wire d_stride_ok = (d_atstr == 32'd0);
    wire d_fmt_ok   = (d_fmt[5:4] != `NPU_FMT_INT4);

    wire d_ok = d_op_ok && d_shape_ok && d_k_ok && d_n_ok &&
                d_abuf_ok && d_stride_ok && d_fmt_ok;

    wire [15:0] d_status =
        !d_op_ok    ? `NPU_ST_BAD_OPCODE :
        !d_shape_ok ? `NPU_ST_RANGE      :
        !d_k_ok     ? `NPU_ST_RANGE      :
        !d_n_ok     ? `NPU_ST_RANGE      :
        !d_stride_ok? `NPU_ST_BAD_FIELD  :
        !d_abuf_ok  ? `NPU_ST_BAD_FIELD  :
        !d_fmt_ok   ? `NPU_ST_UNSUPPORTED: `NPU_ST_OK;

    // ------------------------------------------------------------------
    // latched job context
    // ------------------------------------------------------------------
    reg [511:0] cmd_r;
    reg [15:0]  job_r;
    reg [7:0]   slot_r;
    reg [7:0]   op_r;
    reg [2:0]   mlen_r;
    reg [3:0]   ngrp_r;
    reg [1:0]   mode_r;
    reg [15:0]  tiles_r;
    reg [15:0]  ksteps_r;
    reg [15:0]  t_r;                 // current tile
    reg [15:0]  b_tile_cnt;          // B beats accepted in the current tile
    reg         tile_waiting;        // all B beats in; waiting for tile_done
    reg         b_feed_done;

    reg [31:0]  a_accepted, b_accepted;
    reg [31:0]  tiles_released;
    reg [31:0]  results_captured;
    reg         done_seen;
    reg         pack_done_seen;
    reg [15:0]  pack_error;
    reg [15:0]  status_r;
    reg         fatal_r;

    assign fatal = fatal_r;

    wire [31:0] exp_beats = {16'd0, tiles_r} * {16'd0, ksteps_r};

    // ------------------------------------------------------------------
    // gemv_stream core and A packer
    // ------------------------------------------------------------------
    reg         core_start;
    wire        core_a_ready, core_b_ready, core_acc_valid, core_done;

    wire        a_valid_p, a_ready_p;
    wire [1023:0] a_data_p;
    wire [15:0] a_tile_p, a_step_p;
    wire        a_last_p;
    wire        pack_done_p;
    wire [15:0] pack_error_p;

    reg         pack_cfg_valid;
    wire        pack_cfg_ready;

    wire        a_wr = a_valid_p && a_ready_p &&
                       (a_accepted < exp_beats) && !fatal_r;
    wire        b_wr = w_b_valid && w_b_ready;

    gemv_stream u_core (
        .clk(clk_core), .rst_n(rst_core_n),
        .mode(mode_r), .m_len(mlen_r), .n_grp(ngrp_r),
        .tiles(tiles_r), .k_steps(ksteps_r), .start(core_start),
        .a_wr(a_wr), .a_din(a_data_p),
        .b_wr(b_wr), .b_din(w_b_data),
        .a_ready(core_a_ready), .b_ready(core_b_ready),
        .acc_out(acc_data), .acc_valid(core_acc_valid),
        .busy(), .done(core_done)
    );

    assign acc_valid = core_acc_valid;
    assign a_ready_p = core_a_ready;

    activation_packer #(.MEM_DATA_W(MEM_DATA_W)) u_pack (
        .clk_core(clk_core), .rst_core_n(rst_core_n),
        .cfg_valid(pack_cfg_valid), .cfg_ready(pack_cfg_ready),
        .cfg_cmd(cmd_r),
        .mem_req_valid(mem_req_valid), .mem_req_ready(mem_req_ready),
        .mem_req(mem_req),
        .mem_rsp_valid(mem_rsp_valid), .mem_rsp_ready(mem_rsp_ready),
        .mem_rsp(mem_rsp),
        .a_valid(a_valid_p), .a_ready(a_ready_p), .a_data(a_data_p),
        .a_tile(a_tile_p), .a_step(a_step_p), .a_last(a_last_p),
        .pack_done(pack_done_p), .error(pack_error_p)
    );

    // ------------------------------------------------------------------
    // weight select / B acceptance / tile_done
    // ------------------------------------------------------------------
    assign w_select_valid = (state == M_BSEL);
    assign w_select_job   = job_r;
    assign w_select_tile  = t_r;
    assign w_select_mode  = mode_r;
    assign w_select_g     = ngrp_r;
    assign w_select_s     = ksteps_r;

    wire b_room = !tile_waiting && (b_tile_cnt < ksteps_r) && !b_feed_done;

    assign w_b_ready = (state == M_FEED) && b_room && core_b_ready;

    assign w_tile_done_ready = (state == M_FEED) && tile_waiting;

    // ------------------------------------------------------------------
    // writeback handshake
    // ------------------------------------------------------------------
    assign wb_job_valid = (state == M_RESERVE);
    assign wb_job_cmd   = cmd_r;
    assign wb_done_ready = (state == M_WRITEBACK);

    assign task_ready = (state == M_IDLE);

    // ------------------------------------------------------------------
    // event word
    // ------------------------------------------------------------------
    // The payload is registered one cycle before event_valid rises, so the
    // word is always stable while it is offered.
    reg [127:0] event_r;
    reg         ev_valid_r;
    assign event_data  = event_r;
    assign event_valid = ev_valid_r;

    // ------------------------------------------------------------------
    // result capture count (the pulse itself goes to result_writeback)
    // ------------------------------------------------------------------
    wire job_accept = (state == M_IDLE) && task_valid;

    always @(posedge clk_core or negedge rst_core_n) begin
        if (!rst_core_n) begin
            results_captured <= 32'd0;
            done_seen        <= 1'b0;
        end else if (job_accept) begin
            results_captured <= 32'd0;
            done_seen        <= 1'b0;
        end else begin
            if (core_acc_valid && (results_captured < 32'hFFFF_FFFF))
                results_captured <= results_captured + 32'd1;
            if (core_done) done_seen <= 1'b1;
        end
    end

    // ------------------------------------------------------------------
    // adapter FSM
    // ------------------------------------------------------------------
    always @(posedge clk_core or negedge rst_core_n) begin
        if (!rst_core_n) begin
            state          <= M_IDLE;
            cmd_r          <= 512'd0;
            job_r          <= 16'd0;
            slot_r         <= 8'd0;
            op_r           <= 8'd0;
            mlen_r         <= 3'd0;
            ngrp_r         <= 4'd0;
            mode_r         <= 2'd0;
            tiles_r        <= 16'd0;
            ksteps_r       <= 16'd0;
            t_r            <= 16'd0;
            b_tile_cnt     <= 16'd0;
            tile_waiting   <= 1'b0;
            b_feed_done    <= 1'b0;
            a_accepted     <= 32'd0;
            b_accepted     <= 32'd0;
            tiles_released <= 32'd0;
            pack_done_seen <= 1'b0;
            pack_error     <= `NPU_ST_OK;
            status_r       <= `NPU_ST_OK;
            fatal_r        <= 1'b0;
            core_start     <= 1'b0;
            pack_cfg_valid <= 1'b0;
            event_r        <= 128'd0;
            ev_valid_r     <= 1'b0;
        end else begin
            core_start     <= 1'b0;
            pack_cfg_valid <= 1'b0;

            case (state)
            // ----------------------------------------------------------
            M_IDLE: begin
                if (task_valid) begin
                    cmd_r   <= task_cmd;
                    job_r   <= d_job;
                    slot_r  <= d_slot;
                    mlen_r  <= d_mlen;
                    ngrp_r  <= d_ngrp;
                    mode_r  <= d_mode;
                    tiles_r <= d_tiles;
                    ksteps_r<= d_ksteps;
                    t_r     <= 16'd0;
                    b_tile_cnt <= 16'd0;
                    tile_waiting <= 1'b0;
                    b_feed_done  <= 1'b0;
                    a_accepted   <= 32'd0;
                    b_accepted   <= 32'd0;
                    tiles_released <= 32'd0;
                    pack_done_seen <= 1'b0;
                    pack_error     <= `NPU_ST_OK;
                    op_r           <= d_opcode;
                    status_r       <= d_ok ? `NPU_ST_OK : d_status;
                    if (d_ok) begin
                        state <= M_RESERVE;
                    end else begin
                        // reject without reserving; nothing was executed
                        tiles_r <= 16'd0;
                        state   <= M_COMPLETE;
                    end
                end
            end
            // ----------------------------------------------------------
            M_RESERVE: begin
                if (wb_job_ready) state <= M_BSEL;
            end
            // ----------------------------------------------------------
            // first weight tile must be committed before the core starts
            M_BSEL: begin
                if (w_select_ready) state <= M_START;
            end
            // ----------------------------------------------------------
            M_START: begin
                core_start     <= 1'b1;
                pack_cfg_valid <= 1'b1;
                state          <= M_FEED;
            end
            // ----------------------------------------------------------
            M_FEED: begin
                if (pack_done_p) begin
                    pack_done_seen <= 1'b1;
                    if (pack_error_p != `NPU_ST_OK) begin
                        status_r <= pack_error_p;
                        fatal_r  <= 1'b1;
                        state    <= M_FAULT;
                    end
                end

                if (a_wr) a_accepted <= a_accepted + 32'd1;

                if (b_wr) begin
                    b_accepted <= b_accepted + 32'd1;
                    if (b_tile_cnt + 16'd1 == ksteps_r) begin
                        b_tile_cnt   <= b_tile_cnt + 16'd1;
                        tile_waiting <= 1'b1;
                    end else begin
                        b_tile_cnt <= b_tile_cnt + 16'd1;
                    end
                end

                if (tile_waiting && w_tile_done_valid) begin
                    tiles_released <= tiles_released + 32'd1;
                    tile_waiting   <= 1'b0;
                    b_tile_cnt     <= 16'd0;
                    if (w_tile_done_status != `NPU_ST_OK) begin
                        status_r <= w_tile_done_status;
                        fatal_r  <= 1'b1;
                        state    <= M_FAULT;
                    end else if (t_r + 16'd1 == tiles_r) begin
                        b_feed_done <= 1'b1;
                    end else begin
                        t_r   <= t_r + 16'd1;
                        state <= M_BSEL;
                    end
                end

                if ((state == M_FEED) && pack_done_seen && b_feed_done &&
                    (a_accepted == exp_beats) && (b_accepted == exp_beats) &&
                    (tiles_released == {16'd0, tiles_r}))
                    state <= M_DRAIN;
            end
            // ----------------------------------------------------------
            M_DRAIN: begin
                if (!done_seen) begin
                    // core still producing results; nothing to do
                end else if ((results_captured != {16'd0, tiles_r}) ||
                             (a_accepted != exp_beats) ||
                             (b_accepted != exp_beats)) begin
                    status_r <= `NPU_ST_INTERNAL;
                    fatal_r  <= 1'b1;
                    state    <= M_FAULT;
                end else begin
                    state <= M_WRITEBACK;
                end
            end
            // ----------------------------------------------------------
            M_WRITEBACK: begin
                if (wb_done_valid) begin
                    if ((wb_done_status != `NPU_ST_OK) ||
                        (wb_done_tiles != {16'd0, tiles_r})) begin
                        status_r <= (wb_done_status != `NPU_ST_OK)
                                    ? wb_done_status : `NPU_ST_INTERNAL;
                    end else begin
                        status_r <= `NPU_ST_OK;
                    end
                    state <= M_COMPLETE;
                end
            end
            // ----------------------------------------------------------
            M_COMPLETE: begin
                if (!ev_valid_r) begin
                    event_r    <= {32'd0, {16'd0, tiles_r}, 16'd0, status_r,
                                   op_r, slot_r, job_r};
                    ev_valid_r <= 1'b1;
                end else if (event_ready) begin
                    ev_valid_r <= 1'b0;
                    state      <= M_IDLE;
                end
            end
            // ----------------------------------------------------------
            M_FAULT: begin
                fatal_r <= 1'b1;
                if (!ev_valid_r) begin
                    event_r    <= {32'd0, {16'd0, tiles_r}, 16'd0, status_r,
                                   op_r, slot_r, job_r};
                    ev_valid_r <= 1'b1;
                end else if (event_ready) begin
                    ev_valid_r <= 1'b0;
                end
                // sticky state: recovery requires the platform reset
            end
            default: state <= M_IDLE;
            endcase
        end
    end

endmodule

`default_nettype wire

//=============================================================================
// weight_loader : LOAD_W frame -> weight bank (io domain)
//
//   Follows docs/npu_integration/07_weight_buffer.md sections 2 and 3.
//
//   The frame is weight-major: for every group j and K-step s, the low and
//   high 128-bit halves arrive in that order, followed by the scale words
//   (group-major, ceil(S/4) words per group) for W4 modes.  The loader
//   allocates a free ping/pong bank, writes the RAMs, and only after the last
//   write is registered publishes the descriptor to the bank's core mailbox.
//   done_valid reports that the core has captured the tile; it does not
//   release the bank (weight_reader owns release).
//
//   A rejected or malformed frame is drained up to data_last: the DMA has no
//   abort path, so the loader must consume every beat of the frame even when
//   it will not publish anything.
//=============================================================================
`timescale 1ns/1ps
`default_nettype none

`include "npu_defs.vh"

module weight_loader #(
    parameter GROUPS  = 8,
    parameter K_STEPS = 256
)(
    input  wire        clk_io,
    input  wire        rst_io_n,

    // LOAD_W handoff from dma_engine
    input  wire        wload_valid,
    output wire        wload_ready,
    input  wire [511:0] wload_cmd,
    input  wire        data_valid,
    output wire        data_ready,
    input  wire [127:0] data,
    input  wire        data_last,
    input  wire        data_error,

    output wire        done_valid,
    input  wire        done_ready,
    output wire [15:0] done_status,

    // bank allocation / publication (io domain, both banks)
    input  wire [1:0]  bank_free,
    output reg  [1:0]  bank_fill_start,
    output reg  [1:0]  bank_publish,
    output reg  [1:0]  bank_abort,
    output wire [`NPU_WTILE_W-1:0] bank_desc,
    input  wire [1:0]  bank_commit_done,
    input  wire [31:0] bank_commit_status,     // 16 bit per bank

    // write port (routed to the selected bank by the write-enable bit)
    output wire [1:0]  bank_wr_en,
    output wire        wr_kind,
    output wire [2:0]  wr_grp,
    output wire        wr_half,
    output wire [$clog2(K_STEPS)-1:0] wr_addr,
    output wire [127:0] wr_data
);

    localparam AW = $clog2(K_STEPS);

    localparam L_IDLE  = 4'd0;
    localparam L_WGT   = 4'd1;
    localparam L_SCL   = 4'd2;
    localparam L_BUB   = 4'd3;
    localparam L_PUB   = 4'd4;
    localparam L_WAITC = 4'd5;
    localparam L_DONE  = 4'd6;
    localparam L_DRAIN = 4'd7;
    localparam L_ABT   = 4'd8;

    reg [3:0]  state;
    reg        sel;
    reg        sel_valid;

    reg [15:0] c_job, c_tile;
    reg [1:0]  c_mode;
    reg [3:0]  c_grp_n;
    reg [15:0] c_steps;
    reg [31:0] c_wbytes, c_sbytes;
    reg [15:0] c_wtotal;        // weight 128-bit beats = 2*g*S
    reg [15:0] c_stotal;        // scale words = g*ceil(S/4)
    reg [15:0] c_sq;            // ceil(S/4)

    reg [15:0] wcnt, w_step, w_grp;
    reg        w_half;
    reg [15:0] scnt;

    reg [15:0] status_r;
    reg        done_r;
    reg        rr;             // last allocated bank, for fairness

    wire is_w4 = (c_mode[0] == 1'b0);

    // ------------------------------------------------------------------
    // command decode / validation
    // ------------------------------------------------------------------
    wire [15:0] d_job   = wload_cmd[207:192];    // P2[15:0]
    wire [15:0] d_tile  = wload_cmd[223:208];    // P2[31:16]
    wire [1:0]  d_mode  = wload_cmd[225:224];    // P3[1:0]
    wire [3:0]  d_grp   = wload_cmd[235:232];    // P3[11:8]
    wire [15:0] d_steps = wload_cmd[255:240];    // P3[31:16]
    wire [31:0] d_total = wload_cmd[287:256];    // P4

    wire [31:0] d_wb = {16'd0, d_grp} * {16'd0, d_steps} * 32'd32;
    wire [15:0] d_sq = ({1'b0, d_steps} + 16'd3) >> 2;
    wire [31:0] d_sb = (d_mode[0] == 1'b0)
                       ? {16'd0, d_grp} * {16'd0, d_sq} * 32'd16 : 32'd0;

    wire d_cfg_ok = (wload_cmd[15:8] == `NPU_OP_LOAD) &&
                    (wload_cmd[23:16] == 8'd1) &&
                    (d_grp >= 1) && (d_grp <= GROUPS) &&
                    (d_steps >= 1) && (d_steps <= K_STEPS) &&
                    ({16'd0, d_grp} * {16'd0, d_steps} <= GROUPS*K_STEPS) &&
                    (d_total == (d_wb + d_sb)) && (d_total != 32'd0);

    // beats for the new command (combinational, captured on the handshake)
    wire [15:0] d_wtotal = {12'd0, d_grp} * {12'd0, d_steps} * 16'd2;

    // ------------------------------------------------------------------
    // descriptor presented to the bank
    // ------------------------------------------------------------------
    assign bank_desc = {sel, c_job, c_tile, c_mode, c_grp_n, c_steps,
                        c_wbytes, c_sbytes, `NPU_WTILE_LAYOUT_V1};

    // ------------------------------------------------------------------
    // handshake
    // ------------------------------------------------------------------
    assign wload_ready = (state == L_IDLE) && (bank_free != 2'b00);

    assign data_ready = (state == L_WGT) || (state == L_SCL) ||
                        (state == L_DRAIN);

    // ------------------------------------------------------------------
    // write port: driven straight from the incoming beat
    // ------------------------------------------------------------------
    assign bank_wr_en = (state == L_WGT && data_valid)
                        ? (sel ? 2'b10 : 2'b01)
                        : (state == L_SCL && data_valid)
                          ? (sel ? 2'b10 : 2'b01) : 2'b00;
    assign wr_kind = (state == L_SCL);
    assign wr_data = data;

    // Scale word q of group j is addressed as step 4*q: the bank derives the
    // scale index from the low bits of the step address.
    wire [15:0] sc_div = (c_sq == 16'd0) ? 16'd1 : c_sq;
    wire [15:0] scl_grp = scnt / sc_div;
    wire [15:0] scl_q   = scnt % sc_div;

    assign wr_grp  = (state == L_SCL) ? scl_grp[2:0] : w_grp[2:0];
    assign wr_half = (state == L_SCL) ? 1'b0 : w_half;
    assign wr_addr = (state == L_SCL) ? (scl_q << 2) : w_step[AW-1:0];

    // ------------------------------------------------------------------
    // main FSM
    // ------------------------------------------------------------------
    always @(posedge clk_io or negedge rst_io_n) begin
        if (!rst_io_n) begin
            state           <= L_IDLE;
            sel             <= 1'b0;
            sel_valid       <= 1'b0;
            c_job           <= 16'd0;
            c_tile          <= 16'd0;
            c_mode          <= 2'd0;
            c_grp_n         <= 4'd0;
            c_steps         <= 16'd0;
            c_wbytes        <= 32'd0;
            c_sbytes        <= 32'd0;
            c_wtotal        <= 16'd0;
            c_stotal        <= 16'd0;
            c_sq            <= 16'd0;
            wcnt            <= 16'd0;
            w_step          <= 16'd0;
            w_grp           <= 16'd0;
            w_half          <= 1'b0;
            scnt            <= 16'd0;
            status_r        <= `NPU_ST_OK;
            done_r          <= 1'b0;
            rr              <= 1'b0;
            bank_fill_start <= 2'b00;
            bank_publish    <= 2'b00;
            bank_abort      <= 2'b00;
        end else begin
            bank_fill_start <= 2'b00;
            bank_publish    <= 2'b00;
            bank_abort      <= 2'b00;
            done_r          <= 1'b0;

            case (state)
            // ----------------------------------------------------------
            L_IDLE: begin
                if (wload_valid && wload_ready) begin
                    c_job    <= d_job;
                    c_tile   <= d_tile;
                    c_mode   <= d_mode;
                    c_grp_n  <= d_grp;
                    c_steps  <= d_steps;
                    c_wbytes <= d_wb;
                    c_sbytes <= d_sb;
                    c_wtotal <= d_wtotal;
                    c_sq     <= d_sq;
                    c_stotal <= {1'b0, d_grp} * d_sq;
                    wcnt     <= 16'd0;
                    w_step   <= 16'd0;
                    w_grp    <= 16'd0;
                    w_half   <= 1'b0;
                    scnt     <= 16'd0;
                    status_r <= `NPU_ST_OK;

                    if (!d_cfg_ok) begin
                        sel_valid <= 1'b0;
                        status_r  <= `NPU_ST_BAD_FIELD;
                        state     <= L_DRAIN;
                    end else begin
                        sel_valid <= 1'b1;
                        if (bank_free[0] && (!bank_free[1] || !rr))
                            sel <= 1'b0;
                        else
                            sel <= 1'b1;
                        rr <= ~rr;
                        bank_fill_start <= (bank_free[0] &&
                                            (!bank_free[1] || !rr)) ? 2'b01
                                                                    : 2'b10;
                        state <= L_WGT;
                    end
                end
            end
            // ----------------------------------------------------------
            L_WGT: begin
                if (data_valid) begin
                    if (data_error) begin
                        status_r <= `NPU_ST_DMA_BUS;
                        state    <= data_last ? (sel_valid ? L_ABT : L_DONE)
                                              : L_DRAIN;
                    end else if (wcnt + 16'd1 == c_wtotal) begin
                        // last weight beat: a W4 frame continues with the scale
                        // section, a W16 frame ends here
                        if (is_w4) begin
                            if (data_last) begin
                                // frame ends before the scale section
                                status_r <= `NPU_ST_FRAME_LENGTH;
                                state    <= sel_valid ? L_ABT : L_DONE;
                            end else begin
                                state <= L_SCL;
                            end
                        end else if (!data_last) begin
                            // frame longer than the command: keep draining
                            status_r <= `NPU_ST_FRAME_LENGTH;
                            state    <= L_DRAIN;
                        end else begin
                            state <= L_BUB;
                        end
                    end else if (data_last) begin
                        status_r <= `NPU_ST_FRAME_LENGTH;
                        state    <= sel_valid ? L_ABT : L_DONE;
                    end else begin
                        wcnt <= wcnt + 16'd1;
                        if (!w_half) begin
                            w_half <= 1'b1;
                        end else begin
                            w_half <= 1'b0;
                            if (w_step + 16'd1 == c_steps) begin
                                w_step <= 16'd0;
                                w_grp  <= w_grp + 16'd1;
                            end else begin
                                w_step <= w_step + 16'd1;
                            end
                        end
                    end
                end
            end
            // ----------------------------------------------------------
            L_SCL: begin
                if (data_valid) begin
                    if (data_error) begin
                        status_r <= `NPU_ST_DMA_BUS;
                        state    <= data_last ? (sel_valid ? L_ABT : L_DONE)
                                              : L_DRAIN;
                    end else if (scnt + 16'd1 == c_stotal) begin
                        if (!data_last) begin
                            status_r <= `NPU_ST_FRAME_LENGTH;
                            state    <= L_DRAIN;
                        end else begin
                            state <= L_BUB;
                        end
                    end else if (data_last) begin
                        status_r <= `NPU_ST_FRAME_LENGTH;
                        state    <= sel_valid ? L_ABT : L_DONE;
                    end else begin
                        scnt <= scnt + 16'd1;
                    end
                end
            end
            // ----------------------------------------------------------
            // one bubble so the final RAM write is registered before the
            // publish is visible to the core side
            L_BUB:   state <= L_PUB;
            L_PUB: begin
                bank_publish <= (sel ? 2'b10 : 2'b01);
                state        <= L_WAITC;
            end
            L_WAITC: begin
                if (bank_commit_done[sel]) begin
                    status_r <= sel ? bank_commit_status[31:16]
                                    : bank_commit_status[15:0];
                    state    <= L_DONE;
                end
            end
            // ----------------------------------------------------------
            L_DRAIN: begin
                if (data_valid && data_last) begin
                    state <= sel_valid ? L_ABT : L_DONE;
                end
            end
            L_ABT: begin
                bank_abort <= (sel ? 2'b10 : 2'b01);
                state      <= L_DONE;
            end
            // ----------------------------------------------------------
            L_DONE: begin
                done_r <= 1'b1;
                if (done_ready) begin
                    state     <= L_IDLE;
                    sel_valid <= 1'b0;
                end
            end
            default: state <= L_IDLE;
            endcase
        end
    end

    assign done_valid  = done_r;
    assign done_status = status_r;

endmodule

`default_nettype wire

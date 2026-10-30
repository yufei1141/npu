//=============================================================================
// engine_stub : test-only engine model
//
//   Accepts one command, waits DELAY cycles, then presents exactly one
//   completion event built from the command header.  `fail` is sampled when
//   the task fires and selects FAILED instead of OK.  The event is held until
//   the completion manager accepts it.
//=============================================================================
`timescale 1ns/1ps
`default_nettype none

`include "npu_defs.vh"

module engine_stub #(
    parameter DELAY = 3
)(
    input  wire         clk,
    input  wire         rst_n,
    input  wire         task_valid,
    output wire         task_ready,
    input  wire [511:0] task_cmd,
    input  wire         fail,
    output wire         ev_valid,
    input  wire         ev_ready,
    output wire [127:0] ev
);

    reg        busy;
    reg        ev_valid_r;
    reg [3:0]  cnt;
    reg [15:0] job_r;
    reg [7:0]  slot_r;
    reg [7:0]  op_r;
    reg        fail_r;

    wire fire = task_valid && task_ready;

    assign task_ready = !busy && !ev_valid_r;
    assign ev_valid   = ev_valid_r;
    assign ev         = {80'd0,
                         (fail_r ? `NPU_ST_INTERNAL : `NPU_ST_OK),
                         op_r, slot_r, job_r};

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            busy       <= 1'b0;
            ev_valid_r <= 1'b0;
            cnt        <= 4'd0;
            job_r      <= 16'd0;
            slot_r     <= 8'd0;
            op_r       <= 8'd0;
            fail_r     <= 1'b0;
        end else begin
            if (fire) begin
                busy  <= 1'b1;
                cnt   <= DELAY[3:0];
                job_r <= task_cmd[47:32];
                slot_r <= task_cmd[103:96];
                op_r  <= task_cmd[15:8];
                fail_r <= fail;
            end else if (busy) begin
                if (cnt == 4'd0) begin
                    busy       <= 1'b0;
                    ev_valid_r <= 1'b1;
                end else begin
                    cnt <= cnt - 4'd1;
                end
            end

            if (ev_valid_r && ev_ready)
                ev_valid_r <= 1'b0;
        end
    end

endmodule

`default_nettype wire

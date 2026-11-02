//=============================================================================
// tb_command_frontend : enqueue/dequeue, static checks and duplicate job_id
//=============================================================================
`timescale 1ns/1ps
`default_nettype none

`include "npu_defs.vh"

module tb_command_frontend;
    reg clk = 1'b0;
    reg rst_n = 1'b0;
    reg accept_enable = 1'b0;

    reg         cmd_valid = 1'b0;
    wire        cmd_ready;
    reg [511:0] cmd_data = 512'd0;

    wire        issue_valid;
    reg         issue_ready = 1'b0;
    wire [511:0] issue_cmd;
    wire [15:0] issue_error;
    wire        issue_identity_error;

    reg         held_valid = 1'b0;
    reg [15:0]  held_job = 16'd0;

    wire [31:0] fifo_level;
    wire [63:0] accepted_count;
    wire [63:0] rejected_count;

    integer errors = 0;

    always #5 clk = ~clk;

    command_frontend #(
        .EVENT_SLOTS(16), .RESULT_SLOTS(2), .WBUF_K_STEPS(256),
        .MEM_DATA_W(256), .V_LANES(4), .CMD_FIFO_DEPTH(8)
    ) dut (
        .clk_core(clk), .rst_core_n(rst_n),
        .accept_enable(accept_enable),
        .cmd_valid(cmd_valid), .cmd_ready(cmd_ready), .cmd_data(cmd_data),
        .issue_valid(issue_valid), .issue_ready(issue_ready),
        .issue_cmd(issue_cmd), .issue_error(issue_error),
        .issue_identity_error(issue_identity_error),
        .held_valid(held_valid), .held_job(held_job),
        .fifo_level(fifo_level),
        .accepted_count(accepted_count), .rejected_count(rejected_count)
    );

    task chk;
        input cond;
        input [255:0] msg;
        begin
            if (!cond) begin
                $display("FAIL @%0t: %0s", $time, msg);
                errors = errors + 1;
            end
        end
    endtask

    task mk_fence;
        input [15:0] job;
        input [7:0]  slot;
        input [7:0]  ver;
        begin
            cmd_data = 512'd0;
            cmd_data[7:0]   = ver;
            cmd_data[15:8]  = `NPU_OP_FENCE;
            cmd_data[47:32] = job;
            cmd_data[103:96]= slot;
        end
    endtask

    task mk_vector_add;
        input [15:0] job;
        input [7:0]  slot;
        input [31:0] count;
        begin
            cmd_data = 512'd0;
            cmd_data[7:0]   = 8'h01;
            cmd_data[15:8]  = `NPU_OP_VECTOR;
            cmd_data[23:16] = 8'h00;          // ADD, no broadcast
            cmd_data[47:32] = job;
            cmd_data[71:64] = 8'd0;           // src0
            cmd_data[79:72] = 8'd1;           // src1 (read, not broadcast)
            cmd_data[87:80] = 8'd2;           // dst
            cmd_data[95:88] = 8'h80;          // all FP16, acc FP32
            cmd_data[103:96]= slot;
            cmd_data[159:128] = count;        // P0
            cmd_data[191:160] = 32'd0;        // P1 src0 base
            cmd_data[223:192] = 32'd0;        // P2 src1 base
            cmd_data[255:224] = 32'd0;        // P3 dst base
        end
    endtask

    // enqueue the command currently in cmd_data, wait for it to reach the
    // head, check the static result and then pop it
    task send_check;
        input [15:0]  exp;
        input [255:0] msg;
        begin
            @(negedge clk); cmd_valid = 1'b1;
            @(posedge clk); #1;
            chk(cmd_ready, "command accepted while FIFO has space");
            @(negedge clk); cmd_valid = 1'b0;
            @(posedge clk); #1;
            chk(issue_valid, "issue_valid after enqueue");
            chk(issue_error === exp, msg);
            issue_ready = 1'b1;
            @(posedge clk); #1;
            issue_ready = 1'b0;
        end
    endtask

    reg [511:0] saved_cmd;

    initial begin
        repeat (3) @(posedge clk);
        rst_n = 1'b1;
        repeat (2) @(posedge clk);

        // ------------------------------------------------------------
        // 1. accept and present a valid FENCE
        // ------------------------------------------------------------
        accept_enable = 1'b1;
        mk_fence(16'h0001, 8'd0, 8'h01);
        saved_cmd = cmd_data;
        @(negedge clk);
        cmd_valid = 1'b1;
        @(posedge clk); #1;
        chk(cmd_ready, "FENCE should be ready");
        @(negedge clk);
        cmd_valid = 1'b0;
        @(posedge clk); #1;
        chk(issue_valid, "issue_valid after enqueue");
        chk(issue_cmd === saved_cmd, "issue_cmd must equal enqueued cmd");
        chk(issue_error === 16'd0, "valid FENCE error must be 0");
        chk(issue_identity_error === 1'b0, "valid FENCE identity must be 0");
        chk(fifo_level === 32'd1, "fifo level 1");

        // hold under backpressure
        issue_ready = 1'b0;
        repeat (3) @(posedge clk);
        chk(issue_cmd === saved_cmd, "issue_cmd stable under backpressure");

        // ------------------------------------------------------------
        // 2. duplicate job_id while the first is still queued
        // ------------------------------------------------------------
        mk_fence(16'h0001, 8'd1, 8'h01);
        @(negedge clk);
        cmd_valid = 1'b1;
        @(posedge clk); #1;
        chk(cmd_ready, "duplicate should still be accepted");
        @(negedge clk);
        cmd_valid = 1'b0;
        @(posedge clk); #1;
        chk(fifo_level === 32'd2, "fifo level 2");

        // pop the first; the queued duplicate is now at the head
        issue_ready = 1'b1;
        @(posedge clk); #1;
        chk(issue_valid, "second entry present");
        chk(issue_identity_error === 1'b1, "duplicate identity_error");
        chk(issue_error === `NPU_ST_BAD_FIELD, "duplicate issue_error");
        // consume the duplicate so the queue is empty again
        @(posedge clk); #1;
        chk(issue_valid === 1'b0, "queue empty after duplicate pop");
        issue_ready = 1'b0;
        @(posedge clk); #1;

        // ------------------------------------------------------------
        // 3. static checks
        // ------------------------------------------------------------
        mk_fence(16'h0002, 8'd2, 8'h02);       // bad version
        send_check(`NPU_ST_BAD_VERSION, "BAD_VERSION");

        mk_fence(16'h0003, 8'd16, 8'h01);      // slot out of range
        send_check(`NPU_ST_RANGE, "slot RANGE");

        // bad reserved header bits
        mk_fence(16'h0004, 8'd3, 8'h01);
        cmd_data[127:112] = 16'h0001;
        send_check(`NPU_ST_BAD_FIELD, "reserved BAD_FIELD");

        // valid VECTOR ADD
        mk_vector_add(16'h0005, 8'd4, 32'd16);
        send_check(`NPU_ST_OK, "VECTOR ADD accepted");

        // VECTOR with zero count
        mk_vector_add(16'h0006, 8'd5, 32'd0);
        send_check(`NPU_ST_BAD_FIELD, "zero count BAD_FIELD");

        // scalar broadcast (subop bit 7) accepts an odd count: it reads
        // src1[0] once, so there is no pairing to make even
        mk_vector_add(16'h0008, 8'd0, 32'd5);
        cmd_data[23:16] = 8'h82;               // MUL, scalar broadcast
        send_check(`NPU_ST_OK, "scalar bcast odd count accepted");

        mk_vector_add(16'h0009, 8'd1, 32'd16);
        cmd_data[23:16] = 8'h82;
        send_check(`NPU_ST_OK, "scalar bcast even count accepted");

        // pair broadcast (subop bit 6) does need an even count
        mk_vector_add(16'h000A, 8'd2, 32'd5);
        cmd_data[23:16] = 8'h42;               // MUL, pair broadcast
        send_check(`NPU_ST_BAD_FIELD, "pair bcast odd count BAD_FIELD");

        mk_vector_add(16'h000B, 8'd3, 32'd16);
        cmd_data[23:16] = 8'h42;
        send_check(`NPU_ST_OK, "pair bcast even count accepted");

        // ------------------------------------------------------------
        // 4. accept_enable only gates enqueue
        // ------------------------------------------------------------
        accept_enable = 1'b0;
        mk_fence(16'h0007, 8'd6, 8'h01);
        @(negedge clk); cmd_valid = 1'b1;
        @(posedge clk); #1;
        chk(!cmd_ready, "cmd_ready must drop when accept disabled");
        @(negedge clk); cmd_valid = 1'b0;

        // drain everything
        issue_ready = 1'b1;
        repeat (12) @(posedge clk);
        #1;
        chk(fifo_level === 32'd0, "fifo drained");
        chk(issue_valid === 1'b0, "issue_valid low when empty");
        chk(accepted_count == 64'd11, "accepted_count should be 11");
        chk(rejected_count == 64'd6, "rejected_count should be 6");

        if (errors == 0)
            $display("PASS: tb_command_frontend");
        else
            $display("FAIL: tb_command_frontend (%0d errors)", errors);
        $finish;
    end

    initial begin
        #20000;
        $display("TIMEOUT: tb_command_frontend");
        $finish;
    end

endmodule
`default_nettype wire

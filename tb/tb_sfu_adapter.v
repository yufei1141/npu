//=============================================================================
// tb_sfu_adapter : SFU tasks against the real scratchpad and a stub core
//
//   DUT on memory client 2 (SFU), buffers 2 and 3 both in bank 3.  The stub
//   core copies its input to the output (frame 0) and returns the sign-flipped
//   copy on frame 1, which makes the two-frame SINCOS order observable without
//   depending on the placeholder arithmetic in sfu/.  A test-time reg can make
//   the stub report a non-zero status.
//
//   Covers: EXP2 over FP32, lane padding (0 for EXP2, 1.0 for RCP), FP16
//   source and destination conversion, multi-batch tail, SINCOS sin then cos
//   into two disjoint planes, a numeric status from the core, a scratchpad
//   error on the destination, and the rejecting cases (capability off, INT4,
//   zero count, INPLACE, dirty P4/P5, misalignment, overlapping SINCOS
//   planes, wrong opcode, stray P2).
//=============================================================================
`timescale 1ns/1ps
`default_nettype none

`include "npu_defs.vh"

// ----------------------------------------------------------------------
// stub core: sfu_top port shape, deterministic results
// ----------------------------------------------------------------------
module sfu_stub_core #(
    parameter W      = 8,
    parameter DATA_W = 32
)(
    input  wire clk,
    input  wire rst_n,
    input  wire flush,
    input  wire in_valid,
    output wire in_ready,
    input  wire [3:0] op,
    input  wire [1:0] fmt_in,
    input  wire [1:0] fmt_out,
    input  wire [1:0] mode,
    input  wire [W*DATA_W-1:0] din,
    output reg  out_valid,
    input  wire out_ready,
    output reg  [W*DATA_W-1:0] dout,
    output reg  [7:0] status,
    // last accepted request, for op/mode mapping checks
    output reg  [3:0] seen_op,
    output reg  [1:0] seen_mode
);

    reg [1:0] st;                // 0 idle, 1 delay, 2 present
    reg [1:0] dly;
    reg [W*DATA_W-1:0] din_r;
    reg [3:0] op_r;
    reg       frame;
    reg [7:0] status_r;
    integer   k;

    assign in_ready = (st == 2'd0) && !flush;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            st        <= 2'd0;
            dly       <= 2'd0;
            din_r     <= {(W*DATA_W){1'b0}};
            op_r      <= 4'd0;
            frame     <= 1'b0;
            out_valid <= 1'b0;
            dout      <= {(W*DATA_W){1'b0}};
            status    <= 8'h00;
            status_r  <= 8'h00;
            seen_op   <= 4'd0;
            seen_mode <= 2'd0;
        end else if (flush) begin
            st        <= 2'd0;
            out_valid <= 1'b0;
        end else begin
            case (st)
            2'd0: begin
                if (in_valid && in_ready) begin
                    din_r     <= din;
                    op_r      <= op;
                    seen_op   <= op;
                    seen_mode <= mode;
                    frame  <= 1'b0;
                    dly    <= 2'd2;
                    st     <= 2'd1;
                end
            end
            2'd1: begin
                if (dly != 2'd0) begin
                    dly <= dly - 2'd1;
                end else begin
                    for (k = 0; k < W; k = k + 1) begin
                        if (op_r == 4'd8 && frame)
                            dout[k*DATA_W +: DATA_W] <= din_r[k*DATA_W +: DATA_W] ^ 32'h8000_0000;
                        else
                            dout[k*DATA_W +: DATA_W] <= din_r[k*DATA_W +: DATA_W];
                    end
                    status_r  <= tb_sfu_adapter.err_mode ? 8'h02 : 8'h00;
                    status    <= tb_sfu_adapter.err_mode ? 8'h02 : 8'h00;
                    out_valid <= 1'b1;
                    st        <= 2'd2;
                end
            end
            default: begin
                if (out_ready) begin
                    out_valid <= 1'b0;
                    if (op_r == 4'd8 && !frame) begin
                        frame <= 1'b1;
                        dly   <= 2'd1;
                        st    <= 2'd1;
                    end else begin
                        st <= 2'd0;
                    end
                end
            end
            endcase
        end
    end

endmodule

// ----------------------------------------------------------------------
// testbench
// ----------------------------------------------------------------------
module tb_sfu_adapter;

    localparam DW    = 256;
    localparam B     = 32;
    localparam REQ_W = `NPU_REQ_W(DW, B);
    localparam RSP_W = `NPU_RSP_W(DW);
    localparam F_FP16 = 2'd0;
    localparam F_FP32 = 2'd2;
    localparam F_INT4 = 2'd3;
    localparam BUF    = 8'd3;
    localparam BUF2   = 8'd2;

    reg clk, rst_n;
    initial begin clk = 0; rst_n = 0; end
    always #5 clk = ~clk;

    reg err_mode;

    // ------------------------------------------------------------------
    // DUT
    // ------------------------------------------------------------------
    reg          task_valid;
    wire         task_ready;
    reg  [511:0] task_cmd;

    wire         ev_valid;
    reg          ev_ready;
    wire [127:0] ev_data;

    wire         rq_valid;
    wire         rq_ready;
    wire [REQ_W-1:0] rq;
    wire         rs_valid;
    wire         rs_ready;
    wire [RSP_W-1:0] rs;

    reg  [8:0]   cap_ops;
    wire [2:0]   cap_fmt = 3'b111;
    wire         fatal;

    wire         core_in_valid, core_in_ready;
    wire [3:0]   core_op;
    wire [1:0]   core_fmt_in, core_fmt_out, core_mode;
    wire [255:0] core_din;
    wire         core_out_valid, core_out_ready;
    wire [255:0] core_dout;
    wire [7:0]   core_status;
    wire         core_flush;

    sfu_adapter #(.MEM_DATA_W(DW)) u_dut (
        .clk_core(clk), .rst_core_n(rst_n),
        .task_valid(task_valid), .task_ready(task_ready), .task_cmd(task_cmd),
        .event_valid(ev_valid), .event_ready(ev_ready), .event_data(ev_data),
        .mem_req_valid(rq_valid), .mem_req_ready(rq_ready), .mem_req(rq),
        .mem_rsp_valid(rs_valid), .mem_rsp_ready(rs_ready), .mem_rsp(rs),
        .capability_ops(cap_ops), .capability_formats(cap_fmt),
        .core_in_valid(core_in_valid), .core_in_ready(core_in_ready),
        .core_op(core_op),
        .core_fmt_in(core_fmt_in), .core_fmt_out(core_fmt_out),
        .core_mode(core_mode), .core_din(core_din),
        .core_out_valid(core_out_valid), .core_out_ready(core_out_ready),
        .core_dout(core_dout), .core_status(core_status),
        .core_flush(core_flush),
        .fatal(fatal)
    );

    sfu_stub_core #(.W(8), .DATA_W(32)) u_core (
        .clk(clk), .rst_n(rst_n), .flush(core_flush),
        .in_valid(core_in_valid), .in_ready(core_in_ready),
        .op(core_op), .fmt_in(core_fmt_in), .fmt_out(core_fmt_out),
        .mode(core_mode), .din(core_din),
        .out_valid(core_out_valid), .out_ready(core_out_ready),
        .dout(core_dout), .status(core_status),
        .seen_op(seen_op), .seen_mode(seen_mode)
    );

    wire [3:0] seen_op;
    wire [1:0] seen_mode;

    // ------------------------------------------------------------------
    // memory subsystem, client 2 = SFU
    // ------------------------------------------------------------------
    wire [4:0] c_req_valid = {1'b0, 1'b0, rq_valid, 1'b0, 1'b0};
    wire [4:0] c_req_ready;
    wire [5*REQ_W-1:0] c_req;
    wire [4:0] c_rsp_valid;
    wire [4:0] c_rsp_ready = {1'b0, 1'b0, rs_ready, 1'b0, 1'b0};
    wire [5*RSP_W-1:0] c_rsp;

    assign c_req[0*REQ_W +: REQ_W] = {REQ_W{1'b0}};
    assign c_req[1*REQ_W +: REQ_W] = {REQ_W{1'b0}};
    assign c_req[2*REQ_W +: REQ_W] = rq;
    assign c_req[3*REQ_W +: REQ_W] = {REQ_W{1'b0}};
    assign c_req[4*REQ_W +: REQ_W] = {REQ_W{1'b0}};

    assign rq_ready = c_req_ready[2];
    assign rs_valid = c_rsp_valid[2];
    assign rs       = c_rsp[2*RSP_W +: RSP_W];

    memory_subsystem #(
        .MEM_DATA_W(DW), .BANKS(8), .BANK_DEPTH(1024), .L_RAM(1)
    ) u_mem (
        .clk_core(clk), .rst_core_n(rst_n),
        .buf_enable(8'h0C),
        .buf_bank_flat({32'd0, 8'd3, 8'd3, 16'd0}),
        .buf_base_flat({32'd0, 32'd0, 192'd0}),
        .buf_size_flat({128'd0, 32'd8192, 32'd8192, 64'd0}),
        .c_req_valid(c_req_valid), .c_req_ready(c_req_ready), .c_req(c_req),
        .c_rsp_valid(c_rsp_valid), .c_rsp_ready(c_rsp_ready), .c_rsp(c_rsp),
        .outstanding_count(), .access_fault_valid(), .access_fault_info(),
        .bank_busy()
    );

    // ------------------------------------------------------------------
    // byte / word helpers
    // ------------------------------------------------------------------
    integer fails;
    integer i;

    task poke8;
        input integer addr;
        input [7:0] v;
        begin
            u_mem.g_bank[3].u_ram.mem[addr >> 5][8*(addr & 31) +: 8] = v;
        end
    endtask

    task poke32;
        input integer addr;
        input [31:0] v;
        begin
            poke8(addr + 0, v[7:0]);
            poke8(addr + 1, v[15:8]);
            poke8(addr + 2, v[23:16]);
            poke8(addr + 3, v[31:24]);
        end
    endtask

    task poke16;
        input integer addr;
        input [15:0] v;
        begin
            poke8(addr + 0, v[7:0]);
            poke8(addr + 1, v[15:8]);
        end
    endtask

    function [7:0] peek8;
        input integer addr;
        reg [255:0] w;
        begin
            w = u_mem.g_bank[3].u_ram.mem[addr >> 5];
            peek8 = w[8*(addr & 31) +: 8];
        end
    endfunction

    function [31:0] peek32;
        input integer addr;
        begin
            peek32 = {peek8(addr+3), peek8(addr+2), peek8(addr+1), peek8(addr)};
        end
    endfunction

    function [15:0] peek16;
        input integer addr;
        begin
            peek16 = {peek8(addr+1), peek8(addr)};
        end
    endfunction

    // ------------------------------------------------------------------
    // command build / run
    // ------------------------------------------------------------------
    reg [511:0] cmd;

    task build;
        input [7:0] op, sub, fs0, fd, b0, b1, b2;
        input [31:0] p0, p1, p2, p3, p4;
        begin
            cmd = 512'd0;
            cmd[7:0]     = 8'h01;
            cmd[15:8]    = op;
            cmd[23:16]   = sub;
            cmd[71:64]   = b0;
            cmd[79:72]   = b1;
            cmd[87:80]   = b2;
            cmd[89:88]   = fs0;
            cmd[91:90]   = 2'd0;        // src1 format is unused
            cmd[93:92]   = fd;
            cmd[95:94]   = 2'd0;        // acc format is unused
            cmd[103:96]  = 8'h01;
            cmd[159:128] = p0;
            cmd[191:160] = p1;
            cmd[223:192] = p2;
            cmd[255:224] = p3;
            cmd[287:256] = p4;
        end
    endtask

    reg [15:0] ev_status;
    reg [31:0] ev_units;

    task run;
        input [15:0] exp_status;
        input [31:0] exp_units;
        input [8*20-1:0] what;
        begin
            @(negedge clk);
            task_cmd   = cmd;
            task_valid = 1'b1;
            #1;
            if (!task_ready) $display("  %0s: task not accepted", what);
            @(posedge clk);
            @(negedge clk);
            task_valid = 1'b0;
            while (!ev_valid) @(negedge clk);
            ev_status = ev_data[47:32];
            ev_units  = ev_data[95:64];
            if (ev_status !== exp_status) begin
                fails = fails + 1;
                $display("FAIL %0s: status %h exp %h", what, ev_status, exp_status);
            end
            if (ev_units !== exp_units) begin
                fails = fails + 1;
                $display("FAIL %0s: units %0d exp %0d", what, ev_units, exp_units);
            end
            if (fatal !== 1'b0) begin
                fails = fails + 1;
                $display("FAIL %0s: fatal raised", what);
            end
            ev_ready = 1'b1;
            @(posedge clk);
            @(negedge clk);
            ev_ready = 1'b0;
        end
    endtask

    task chk32;
        input integer addr;
        input [31:0] exp;
        input [8*16-1:0] what;
        reg [31:0] got;
        begin
            got = peek32(addr);
            if (got !== exp) begin
                fails = fails + 1;
                $display("FAIL %0s @%0d: %h exp %h", what, addr, got, exp);
            end
        end
    endtask

    task chk8;
        input integer addr;
        input [7:0] exp;
        input [8*16-1:0] what;
        reg [7:0] got;
        begin
            got = peek8(addr);
            if (got !== exp) begin
                fails = fails + 1;
                $display("FAIL %0s @%0d: %h exp %h", what, addr, got, exp);
            end
        end
    endtask

    task chk16;
        input integer addr;
        input [15:0] exp;
        input [8*16-1:0] what;
        reg [15:0] got;
        begin
            got = peek16(addr);
            if (got !== exp) begin
                fails = fails + 1;
                $display("FAIL %0s @%0d: %h exp %h", what, addr, got, exp);
            end
        end
    endtask

    // op/mode the core saw for the last accepted request
    task chk_core;
        input [3:0] exp_op;
        input [1:0] exp_mode;
        input [8*16-1:0] what;
        begin
            if (u_core.seen_op !== exp_op) begin
                fails = fails + 1;
                $display("FAIL %0s: core op %0d exp %0d", what,
                         u_core.seen_op, exp_op);
            end
            if (u_core.seen_mode !== exp_mode) begin
                fails = fails + 1;
                $display("FAIL %0s: core mode %0d exp %0d", what,
                         u_core.seen_mode, exp_mode);
            end
        end
    endtask

    // padded lane of the last issued batch
    task chk_pad;
        input integer lane;
        input [31:0] exp;
        input [8*16-1:0] what;
        reg [31:0] got;
        begin
            got = u_core.din_r[lane*32 +: 32];
            if (got !== exp) begin
                fails = fails + 1;
                $display("FAIL %0s pad[%0d]: %h exp %h", what, lane, got, exp);
            end
        end
    endtask

    // ------------------------------------------------------------------
    // test sequence
    // ------------------------------------------------------------------
    localparam F1_0 = 32'h3F80_0000;
    localparam F1_5 = 32'h3FC0_0000;
    localparam F2_0 = 32'h4000_0000;
    localparam F2_5 = 32'h4020_0000;
    localparam F3_0 = 32'h4040_0000;
    localparam F3_5 = 32'h4060_0000;
    localparam F4_0 = 32'h4080_0000;
    localparam F5_0 = 32'h40A0_0000;

    initial begin
        fails      = 0;
        task_valid = 1'b0;
        ev_ready   = 1'b0;
        task_cmd   = 512'd0;
        cap_ops    = 9'h1FF;
        err_mode   = 1'b0;

        repeat (4) @(posedge clk);
        rst_n = 1'b1;
        repeat (2) @(posedge clk);

        // ---- EXP2 FP32, count 5: identity, tail lanes padded with 0 ----
        poke32(32'h100, F1_0); poke32(32'h104, F2_0);
        poke32(32'h108, F3_0); poke32(32'h10C, F4_0);
        poke32(32'h110, F5_0);
        for (i = 0; i < 32; i = i + 1) poke8(32'h200 + i, 8'hAA);
        build(`NPU_OP_SFU, 8'd0, F_FP32, F_FP32, BUF, 8'd0, BUF,
              32'd5, 32'h100, 32'd0, 32'h200, 32'd0);
        run(`NPU_ST_OK, 32'd5, "exp2 fp32");
        chk32(32'h200, F1_0, "exp2[0]");
        chk32(32'h204, F2_0, "exp2[1]");
        chk32(32'h208, F3_0, "exp2[2]");
        chk32(32'h20C, F4_0, "exp2[3]");
        chk32(32'h210, F5_0, "exp2[4]");
        chk8(32'h214, 8'hAA, "exp2 tail");
        chk_pad(5, 32'h0000_0000, "exp2");
        chk_pad(7, 32'h0000_0000, "exp2");

        // ---- RCP FP16 in/out, count 3: tail lanes padded with 1.0 ----
        poke16(32'h300, 16'h3E00);      // 1.5
        poke16(32'h302, 16'h4100);      // 2.5
        poke16(32'h304, 16'h4300);      // 3.5
        for (i = 0; i < 16; i = i + 1) poke8(32'h320 + i, 8'h55);
        build(`NPU_OP_SFU, 8'd1, F_FP16, F_FP16, BUF, 8'd0, BUF,
              32'd3, 32'h300, 32'd0, 32'h320, 32'd0);
        run(`NPU_ST_OK, 32'd3, "rcp fp16");
        chk16(32'h320, 16'h3E00, "rcp[0]");
        chk16(32'h322, 16'h4100, "rcp[1]");
        chk16(32'h324, 16'h4300, "rcp[2]");
        chk8(32'h326, 8'h55, "rcp tail");
        chk_pad(3, F1_0, "rcp");
        chk_pad(7, F1_0, "rcp");
        chk_pad(2, F3_5, "rcp");

        // ---- SINCOS, count 4: sin plane then cos plane ----
        poke32(32'h400, F1_0); poke32(32'h404, F2_0);
        poke32(32'h408, F3_0); poke32(32'h40C, F4_0);
        for (i = 0; i < 32; i = i + 1) begin
            poke8(32'h440 + i, 8'h00);
            poke8(32'h480 + i, 8'h00);
        end
        build(`NPU_OP_SFU, 8'd3, F_FP32, F_FP32, BUF, 8'd0, BUF,
              32'd4, 32'h400, 32'h480, 32'h440, 32'd0);
        run(`NPU_ST_OK, 32'd8, "sincos");
        chk32(32'h440, F1_0, "sin[0]");
        chk32(32'h444, F2_0, "sin[1]");
        chk32(32'h448, F3_0, "sin[2]");
        chk32(32'h44C, F4_0, "sin[3]");
        chk32(32'h480, 32'hBF80_0000, "cos[0]");
        chk32(32'h484, 32'hC000_0000, "cos[1]");
        chk32(32'h488, 32'hC040_0000, "cos[2]");
        chk32(32'h48C, 32'hC080_0000, "cos[3]");

        // ---- multi-batch: 10 elements, second batch has 2 valid lanes ----
        for (i = 0; i < 10; i = i + 1) poke32(32'h500 + 4*i, F1_0 + (i << 23));
        build(`NPU_OP_SFU, 8'd0, F_FP32, F_FP32, BUF, 8'd0, BUF,
              32'd10, 32'h500, 32'd0, 32'h540, 32'd0);
        run(`NPU_ST_OK, 32'd10, "exp2 count 10");
        chk32(32'h540, F1_0, "b1[0]");
        chk32(32'h560, 32'h4380_0000, "b2[0]");     // 256.0
        chk32(32'h564, 32'h4400_0000, "b2[1]");     // 512.0
        chk_pad(2, 32'h0000_0000, "batch2");
        chk_pad(7, 32'h0000_0000, "batch2");

        // ---- core status: numeric failure, nothing written ----
        poke32(32'h600, F1_0); poke32(32'h604, F2_0);
        for (i = 0; i < 16; i = i + 1) poke8(32'h620 + i, 8'h77);
        err_mode = 1'b1;
        build(`NPU_OP_SFU, 8'd0, F_FP32, F_FP32, BUF, 8'd0, BUF,
              32'd2, 32'h600, 32'd0, 32'h620, 32'd0);
        run(`NPU_ST_NUMERIC, 32'd0, "core status");
        chk8(32'h620, 8'h77, "num tail");
        chk8(32'h624, 8'h77, "num tail2");
        err_mode = 1'b0;

        // ---- SINCOS numeric on frame 0: frame 1 must still be drained ----
        err_mode = 1'b1;
        build(`NPU_OP_SFU, 8'd3, F_FP32, F_FP32, BUF, 8'd0, BUF,
              32'd2, 32'h600, 32'h6A0, 32'h680, 32'd0);
        run(`NPU_ST_NUMERIC, 32'd0, "sincos status");
        err_mode = 1'b0;

        // ---- scratchpad error on the destination (buffer 5 is disabled) --
        poke32(32'h700, F1_0);
        build(`NPU_OP_SFU, 8'd0, F_FP32, F_FP32, BUF, 8'd0, 8'd5,
              32'd1, 32'h700, 32'd0, 32'h20, 32'd0);
        run(`NPU_ST_MEM_ACCESS, 32'd0, "dest buf off");

        // ---- subops added with the full core: explicit core-op mapping ----
        poke32(32'h100, F1_0); poke32(32'h104, F2_0);
        build(`NPU_OP_SFU, 8'd4, F_FP32, F_FP32, BUF, 8'd0, BUF,
              32'd2, 32'h100, 32'd0, 32'h200, 32'd0);
        run(`NPU_ST_OK, 32'd2, "log2");
        chk32(32'h200, F1_0, "log2[0]");
        chk_core(4'd1, 2'd0, "log2");

        build(`NPU_OP_SFU, 8'd5, F_FP32, F_FP32, BUF, 8'd0, BUF,
              32'd2, 32'h100, 32'd0, 32'h200, 32'd0);
        run(`NPU_ST_OK, 32'd2, "sqrt");
        chk32(32'h200, F1_0, "sqrt[0]");
        chk_core(4'd4, 2'd0, "sqrt");

        build(`NPU_OP_SFU, 8'd6, F_FP32, F_FP32, BUF, 8'd0, BUF,
              32'd2, 32'h100, 32'd0, 32'h200, 32'd0);
        run(`NPU_ST_OK, 32'd2, "sin");
        chk32(32'h200, F1_0, "sin[0]");
        chk_core(4'd5, 2'd0, "sin");

        build(`NPU_OP_SFU, 8'd7, F_FP32, F_FP32, BUF, 8'd0, BUF,
              32'd2, 32'h100, 32'd0, 32'h200, 32'd0);
        run(`NPU_ST_OK, 32'd2, "cos");
        chk32(32'h200, F1_0, "cos[0]");
        chk_core(4'd6, 2'd0, "cos");

        build(`NPU_OP_SFU, 8'd8, F_FP32, F_FP32, BUF, 8'd0, BUF,
              32'd2, 32'h100, 32'd0, 32'h200, 32'd0);
        run(`NPU_ST_OK, 32'd2, "tanh");
        chk32(32'h200, F1_0, "tanh[0]");
        chk_core(4'd7, 2'd0, "tanh");

        // ---- P4[1:0] selects the core mode and is held per task ----
        build(`NPU_OP_SFU, 8'd0, F_FP32, F_FP32, BUF, 8'd0, BUF,
              32'd2, 32'h100, 32'd0, 32'h200, 32'd1);
        run(`NPU_ST_OK, 32'd2, "accurate mode");
        chk_core(4'd0, 2'd1, "accurate mode");

        build(`NPU_OP_SFU, 8'd0, F_FP32, F_FP32, BUF, 8'd0, BUF,
              32'd2, 32'h100, 32'd0, 32'h200, 32'd0);
        run(`NPU_ST_OK, 32'd2, "fast mode again");
        chk_core(4'd0, 2'd0, "fast mode again");

        // ---- rejecting cases ----
        build(`NPU_OP_SFU, 8'd9, F_FP32, F_FP32, BUF, 8'd0, BUF,
              32'd2, 32'h100, 32'd0, 32'h200, 32'd0);
        run(`NPU_ST_BAD_FIELD, 32'd0, "subop 9");

        build(`NPU_OP_SFU, 8'd0, F_FP32, F_FP32, BUF, 8'd0, BUF,
              32'd2, 32'h100, 32'd0, 32'h200, 32'd2);
        run(`NPU_ST_BAD_FIELD, 32'd0, "mode 2");

        build(`NPU_OP_SFU, 8'd0, F_FP32, F_FP32, BUF, 8'd0, BUF,
              32'd2, 32'h100, 32'd0, 32'h200, 32'd0);
        cap_ops = 9'b0_0000_0000;
        run(`NPU_ST_UNSUPPORTED, 32'd0, "capability off");
        cap_ops = 9'h1FF;

        build(`NPU_OP_SFU, 8'd0, F_FP32, F_INT4, BUF, 8'd0, BUF,
              32'd2, 32'h100, 32'd0, 32'h200, 32'd0);
        run(`NPU_ST_UNSUPPORTED, 32'd0, "int4 dest");

        build(`NPU_OP_SFU, 8'd0, F_FP32, F_FP32, BUF, 8'd0, BUF,
              32'd0, 32'h100, 32'd0, 32'h200, 32'd0);
        run(`NPU_ST_BAD_FIELD, 32'd0, "zero count");

        // P4[1:0] is function_mode; anything above bit 1 must be rejected
        build(`NPU_OP_SFU, 8'd0, F_FP32, F_FP32, BUF, 8'd0, BUF,
              32'd2, 32'h100, 32'd0, 32'h200, 32'd4);
        run(`NPU_ST_BAD_FIELD, 32'd0, "p4 dirty");

        build(`NPU_OP_SFU, 8'd0, F_FP32, F_FP32, BUF, 8'd0, BUF,
              32'd2, 32'h100, 32'd0, 32'h200, 32'd0);
        cmd[295:288] = 8'h01;           // P5 must be zero
        run(`NPU_ST_BAD_FIELD, 32'd0, "p5 dirty");

        build(`NPU_OP_SFU, 8'd0, F_FP32, F_FP32, BUF, 8'd0, BUF,
              32'd2, 32'h100, 32'd0, 32'h202, 32'd0);
        run(`NPU_ST_ALIGN, 32'd0, "unaligned dst");

        build(`NPU_OP_SFU, 8'd0, F_FP32, F_FP32, BUF, 8'd0, BUF,
              32'd2, 32'h100, 32'd0, 32'h200, 32'd0);
        cmd[24 + `NPU_FLAG_INPLACE] = 1'b1;
        run(`NPU_ST_BAD_FIELD, 32'd0, "inplace");

        build(`NPU_OP_SFU, 8'd3, F_FP32, F_FP32, BUF, 8'd0, BUF,
              32'd8, 32'h100, 32'h210, 32'h200, 32'd0);
        run(`NPU_ST_BAD_FIELD, 32'd0, "sincos overlap");

        build(`NPU_OP_SFU, 8'd0, F_FP32, F_FP32, BUF, 8'd0, BUF,
              32'd2, 32'h100, 32'd0, 32'h200, 32'd0);
        cmd[223:192] = 32'h10;          // P2 is SINCOS only
        run(`NPU_ST_BAD_FIELD, 32'd0, "stray p2");

        build(`NPU_OP_VECTOR, 8'd0, F_FP32, F_FP32, BUF, 8'd0, BUF,
              32'd2, 32'h100, 32'd0, 32'h200, 32'd0);
        run(`NPU_ST_BAD_OPCODE, 32'd0, "bad opcode");

        build(`NPU_OP_SFU, 8'd3, F_FP32, F_FP32, BUF, 8'd0, BUF,
              32'h8000_0000, 32'h100, 32'h200, 32'h200, 32'd0);
        run(`NPU_ST_RANGE, 32'd0, "sincos count range");

        if (fails == 0) $display("PASS: tb_sfu_adapter");
        else            $display("FAIL: tb_sfu_adapter (%0d)", fails);
        $finish;
    end

    // watchdog
    initial begin
        #200000;
        $display("FAIL: tb_sfu_adapter (timeout)");
        $finish;
    end

endmodule

`default_nettype wire

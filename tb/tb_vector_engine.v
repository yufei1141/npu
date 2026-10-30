//=============================================================================
// tb_vector_engine : VECTOR / REDUCE tasks against the real scratchpad
//
//   DUT on memory client 1 (VPU), buffer 3 -> bank 3.  Every case preloads
//   source words, runs one task, waits for the completion event and compares
//   the destination bytes with expected FP32/FP16 values.  Covers ADD, MUL
//   in place, SELECT with a byte mask, COMPARE_GT, MUL_SCALAR, CONVERT to
//   FP16, scalar broadcast, REDUCE SUM, masked REDUCE MAX, all-masked SUM,
//   ROTATE_PAIR and two rejecting cases (zero count, INT4 source).
//=============================================================================
`timescale 1ns/1ps
`default_nettype none

`include "npu_defs.vh"

module tb_vector_engine;

    localparam DW    = 256;
    localparam B     = 32;
    localparam REQ_W = `NPU_REQ_W(DW, B);
    localparam RSP_W = `NPU_RSP_W(DW);
    localparam F_FP16 = 2'd0;
    localparam F_BF16 = 2'd1;
    localparam F_FP32 = 2'd2;
    localparam F_INT4 = 2'd3;
    localparam BUF    = 8'd3;
    localparam IMG    = 8192;

    reg clk, rst_n;
    initial begin clk = 0; rst_n = 0; end
    always #5 clk = ~clk;

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

    wire [7:0]   fp_exception;
    wire         fatal;

    vector_engine #(.MEM_DATA_W(DW)) u_dut (
        .clk_core(clk), .rst_core_n(rst_n),
        .task_valid(task_valid), .task_ready(task_ready), .task_cmd(task_cmd),
        .event_valid(ev_valid), .event_ready(ev_ready), .event_data(ev_data),
        .mem_req_valid(rq_valid), .mem_req_ready(rq_ready), .mem_req(rq),
        .mem_rsp_valid(rs_valid), .mem_rsp_ready(rs_ready), .mem_rsp(rs),
        .fp_exception(fp_exception), .fatal(fatal)
    );

    // ------------------------------------------------------------------
    // memory subsystem, client 1 = VPU, buffer 3 -> bank 3
    // ------------------------------------------------------------------
    wire [4:0] c_req_valid = {1'b0, 1'b0, 1'b0, rq_valid, 1'b0};
    wire [4:0] c_req_ready;
    wire [5*REQ_W-1:0] c_req;
    wire [4:0] c_rsp_valid;
    wire [4:0] c_rsp_ready = {1'b0, 1'b0, 1'b0, rs_ready, 1'b0};
    wire [5*RSP_W-1:0] c_rsp;

    assign c_req[1*REQ_W +: REQ_W] = rq;
    assign c_req[0*REQ_W +: REQ_W] = {REQ_W{1'b0}};
    assign c_req[2*REQ_W +: REQ_W] = {REQ_W{1'b0}};
    assign c_req[3*REQ_W +: REQ_W] = {REQ_W{1'b0}};
    assign c_req[4*REQ_W +: REQ_W] = {REQ_W{1'b0}};

    assign rq_ready = c_req_ready[1];
    assign rs_valid = c_rsp_valid[1];
    assign rs       = c_rsp[1*RSP_W +: RSP_W];

    memory_subsystem #(
        .MEM_DATA_W(DW), .BANKS(8), .BANK_DEPTH(1024), .L_RAM(1)
    ) u_mem (
        .clk_core(clk), .rst_core_n(rst_n),
        // buffers 2 and 3 both live in bank 3; buffer 2 is only used as a
        // second logical buffer so a broadcast source can differ from dst
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
        input [7:0] op, sub, fs0, fs1, fd, b0, b1, b2, bm;
        input [31:0] p0, p1, p2, p3, p4, p6;
        begin
            cmd = 512'd0;
            cmd[7:0]     = 8'h01;
            cmd[15:8]    = op;
            cmd[23:16]   = sub;
            cmd[71:64]   = b0;
            cmd[79:72]   = b1;
            cmd[87:80]   = b2;
            cmd[89:88]   = fs0;
            cmd[91:90]   = fs1;
            cmd[93:92]   = fd;
            cmd[95:94]   = F_FP32;      // accumulator is always FP32
            cmd[103:96]  = 8'h01;
            cmd[159:128] = p0;
            cmd[191:160] = p1;
            cmd[223:192] = p2;
            cmd[255:224] = p3;
            cmd[287:256] = p4;
            cmd[295:288] = bm;
            cmd[351:320] = p6;
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
    localparam F4_5 = 32'h40A0_0000;
    localparam F5_0 = 32'h40A0_0000;
    localparam F5_5 = 32'h40B0_0000;

    initial begin
        fails      = 0;
        task_valid = 1'b0;
        ev_ready   = 1'b0;
        task_cmd   = 512'd0;

        repeat (4) @(posedge clk);
        rst_n = 1'b1;
        repeat (2) @(posedge clk);

        // ---- ADD FP32, count 5 ----
        for (i = 0; i < 5; i = i + 1)
            poke32(32'h100 + 4*i, F1_0 + (i << 23));    // 1,2,3,4,5
        for (i = 0; i < 5; i = i + 1)
            poke32(32'h200 + 4*i, 32'h3F00_0000);       // 0.5 each
        build(`NPU_OP_VECTOR, 8'd0, F_FP32, F_FP32, F_FP32,
              BUF, BUF, BUF, 8'd0, 32'd5, 32'h100, 32'h200, 32'h300, 32'd0, 32'd0);
        run(`NPU_ST_OK, 32'd5, "add fp32");
        chk32(32'h300, F1_5, "add[0]");
        chk32(32'h304, F2_5, "add[1]");
        chk32(32'h308, 32'h4090_0000, "add[2]");
        chk32(32'h30C, 32'h4108_0000, "add[3]");
        chk32(32'h310, 32'h4184_0000, "add[4]");

        // ---- MUL in place ----
        poke32(32'h400, F2_0); poke32(32'h404, F3_0); poke32(32'h408, F4_0);
        poke32(32'h440, F4_0); poke32(32'h444, F5_0); poke32(32'h448, 32'h40C0_0000);
        build(`NPU_OP_VECTOR, 8'd2, F_FP32, F_FP32, F_FP32,
              BUF, BUF, BUF, 8'd0, 32'd3, 32'h400, 32'h440, 32'h400, 32'd0, 32'd0);
        cmd[24 + `NPU_FLAG_INPLACE] = 1'b1;   // flags byte starts at bit 24
        run(`NPU_ST_OK, 32'd3, "mul inplace");
        chk32(32'h400, 32'h4100_0000, "mul[0]");    // 8
        chk32(32'h404, 32'h4170_0000, "mul[1]");    // 15
        chk32(32'h408, 32'h41C0_0000, "mul[2]");    // 24

        // ---- SELECT with a byte mask ----
        poke8(32'h500, 8'd1); poke8(32'h501, 8'd0);
        poke8(32'h502, 8'd7); poke8(32'h503, 8'd0);
        poke32(32'h520, F1_0); poke32(32'h524, F2_0);
        poke32(32'h528, F3_0); poke32(32'h52C, F4_0);
        poke32(32'h540, 32'h4110_0000); poke32(32'h544, 32'h4110_0000);
        poke32(32'h548, 32'h4110_0000); poke32(32'h54C, 32'h4110_0000);
        build(`NPU_OP_VECTOR, 8'd4, F_FP32, F_FP32, F_FP32,
              BUF, BUF, BUF, BUF, 32'd4, 32'h520, 32'h540, 32'h560, 32'd0, 32'h500);
        run(`NPU_ST_OK, 32'd4, "select");
        chk32(32'h560, F1_0, "sel[0]");
        chk32(32'h564, 32'h4110_0000, "sel[1]");
        chk32(32'h568, F3_0, "sel[2]");
        chk32(32'h56C, 32'h4110_0000, "sel[3]");

        // ---- COMPARE_GT, one byte per element ----
        poke32(32'h600, F1_0); poke32(32'h604, F5_0); poke32(32'h608, F3_0);
        poke32(32'h620, F2_0); poke32(32'h624, F2_0); poke32(32'h628, F2_0);
        poke8(32'h640, 8'hAA); poke8(32'h641, 8'hAA); poke8(32'h642, 8'hAA);
        build(`NPU_OP_VECTOR, 8'd8, F_FP32, F_FP32, F_FP16,
              BUF, BUF, BUF, 8'd0, 32'd3, 32'h600, 32'h620, 32'h640, 32'd0, 32'd0);
        run(`NPU_ST_OK, 32'd3, "compare_gt");
        chk8(32'h640, 8'd0, "gt[0]");
        chk8(32'h641, 8'd1, "gt[1]");
        chk8(32'h642, 8'd1, "gt[2]");

        // ---- MUL_SCALAR ----
        poke32(32'h660, F1_5); poke32(32'h664, F2_5);
        build(`NPU_OP_VECTOR, 8'd6, F_FP32, F_FP32, F_FP32,
              BUF, 8'd0, BUF, 8'd0, 32'd2, 32'h660, 32'd0, 32'h680, F2_0, 32'd0);
        run(`NPU_ST_OK, 32'd2, "mul_scalar");
        chk32(32'h680, F3_0, "muls[0]");
        chk32(32'h684, F5_0, "muls[1]");

        // ---- CONVERT FP32 -> FP16 ----
        poke32(32'h6A0, F1_5); poke32(32'h6A4, F2_5);
        build(`NPU_OP_VECTOR, 8'd5, F_FP32, F_FP32, F_FP16,
              BUF, 8'd0, BUF, 8'd0, 32'd2, 32'h6A0, 32'd0, 32'h6C0, 32'd0, 32'd0);
        run(`NPU_ST_OK, 32'd2, "convert fp16");
        chk16(32'h6C0, 16'h3E00, "cvt[0]");
        chk16(32'h6C2, 16'h4100, "cvt[1]");

        // ---- ADD with scalar broadcast (bit7) ----
        // the broadcast source must live in another logical buffer than dst
        poke32(32'h700, F1_0); poke32(32'h704, F2_0);
        poke32(32'h20, 32'h3F00_0000);              // only src1[0] exists
        build(`NPU_OP_VECTOR, 8'h80, F_FP32, F_FP32, F_FP32,
              BUF, 8'd2, BUF, 8'd0, 32'd2, 32'h700, 32'h20, 32'h740, 32'd0, 32'd0);
        run(`NPU_ST_OK, 32'd2, "add bcast");
        chk32(32'h740, F1_5, "bcast[0]");
        chk32(32'h744, F2_5, "bcast[1]");

        // ---- REDUCE SUM ----
        poke32(32'h760, F1_0); poke32(32'h764, F2_0);
        poke32(32'h768, F3_0); poke32(32'h76C, F4_0);
        build(`NPU_OP_REDUCE, 8'd0, F_FP32, F_FP16, F_FP32,
              BUF, 8'd0, BUF, 8'd0, 32'd4, 32'h760, 32'd0, 32'h780, 32'd0, 32'd0);
        run(`NPU_ST_OK, 32'd1, "reduce sum");
        chk32(32'h780, 32'h4120_0000, "sum");       // 10.0

        // ---- masked REDUCE MAX ----
        poke8(32'h7A0, 8'd0); poke8(32'h7A1, 8'd1);
        poke8(32'h7A2, 8'd0); poke8(32'h7A3, 8'd1);
        poke32(32'h7B0, F1_0); poke32(32'h7B4, 32'h40E0_0000);
        poke32(32'h7B8, F3_0); poke32(32'h7BC, 32'h4110_0000);
        build(`NPU_OP_REDUCE, 8'h81, F_FP32, F_FP16, F_FP32,
              BUF, 8'd0, BUF, BUF, 32'd4, 32'h7B0, 32'd0, 32'h7C0, 32'd0, 32'h7A0);
        run(`NPU_ST_OK, 32'd1, "reduce max mask");
        chk32(32'h7C0, 32'h4110_0000, "rmax");      // 9.0

        // ---- all-masked SUM writes +0 ----
        poke8(32'h7D0, 8'd0); poke8(32'h7D1, 8'd0);
        poke8(32'h7D2, 8'd0); poke8(32'h7D3, 8'd0);
        poke32(32'h7E0, 32'hDEAD_BEEF);
        build(`NPU_OP_REDUCE, 8'h80, F_FP32, F_FP16, F_FP32,
              BUF, 8'd0, BUF, BUF, 32'd4, 32'h7B0, 32'd0, 32'h7E0, 32'd0, 32'h7D0);
        run(`NPU_ST_OK, 32'd1, "all masked sum");
        chk32(32'h7E0, 32'h0000_0000, "sum0");

        // ---- ROTATE_PAIR ----
        poke32(32'h800, F1_0); poke32(32'h804, F2_0);
        poke32(32'h808, F3_0); poke32(32'h80C, F4_0);
        build(`NPU_OP_VECTOR, 8'd9, F_FP32, F_FP32, F_FP32,
              BUF, 8'd0, BUF, 8'd0, 32'd4, 32'h800, 32'd0, 32'h820, 32'd0, 32'd0);
        run(`NPU_ST_OK, 32'd4, "rotate pair");
        chk32(32'h820, 32'hC000_0000, "rot[0]");    // -2
        chk32(32'h824, F1_0, "rot[1]");             // 1
        chk32(32'h828, 32'hC080_0000, "rot[2]");    // -4
        chk32(32'h82C, F3_0, "rot[3]");             // 3

        // ---- rejecting cases ----
        build(`NPU_OP_VECTOR, 8'd0, F_FP32, F_FP32, F_FP32,
              BUF, BUF, BUF, 8'd0, 32'd0, 32'h100, 32'h200, 32'h300, 32'd0, 32'd0);
        run(`NPU_ST_BAD_FIELD, 32'd0, "zero count");

        build(`NPU_OP_VECTOR, 8'd0, F_INT4, F_FP32, F_FP32,
              BUF, BUF, BUF, 8'd0, 32'd2, 32'h100, 32'h200, 32'h300, 32'd0, 32'd0);
        run(`NPU_ST_UNSUPPORTED, 32'd0, "int4 source");

        // out-of-range destination: past the buffer end
        build(`NPU_OP_VECTOR, 8'd0, F_FP32, F_FP32, F_FP32,
              BUF, BUF, BUF, 8'd0, 32'hFFFF_FFFF, 32'h100, 32'h200, 32'h300, 32'd0, 32'd0);
        run(`NPU_ST_RANGE, 32'd0, "range");

        if (fails == 0) $display("PASS: tb_vector_engine");
        else            $display("FAIL: tb_vector_engine (%0d)", fails);
        $finish;
    end

    // watchdog
    initial begin
        #200000;
        $display("FAIL: tb_vector_engine (timeout)");
        $finish;
    end

endmodule

`default_nettype wire

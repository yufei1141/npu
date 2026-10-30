//=============================================================================
// tb_tensor_layout : LAYOUT tasks against the real scratchpad
//
//   DUT is tensor_layout on its own; vector_engine's routing mux is covered by
//   tb_vector_engine.  Source buffer 2 -> bank 2, destination buffer 3 ->
//   bank 3, so every check can name a constant bank in the hierarchy.
//
//   Cases:
//     TRANSPOSE2D FP32 2x3 with padded strides
//     TRANSPOSE2D FP16 1x1 (single element)
//     TRANSPOSE2D FP16 3x2 with element-crossing-word addresses, destination
//       pre-filled so untouched bytes prove the strobe behaviour
//     PACK_W16 K=3 N=3 g=1 (K tail and N tail padding inside one group)
//     PACK_W16 K=5 N=4 g=2 (second group is entirely N padding)
//     rejections: INPLACE, src1 nonzero, bad subop, FP32 pack, zero dimension,
//       bad group count, N > 4g, unaligned destination, short stride,
//       unaligned base, nonzero P6, INT4, mismatched formats
//=============================================================================
`timescale 1ns/1ps
`default_nettype none

`include "npu_defs.vh"

module tb_tensor_layout;

    localparam DW    = 256;
    localparam B     = 32;
    localparam REQ_W = `NPU_REQ_W(DW, B);
    localparam RSP_W = `NPU_RSP_W(DW);
    localparam F_FP16 = 2'd0;
    localparam F_FP32 = 2'd2;
    localparam F_INT4 = 2'd3;
    localparam SRC_BUF = 8'd2;      // bank 2
    localparam DST_BUF = 8'd3;      // bank 3

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

    wire         active;
    wire [31:0]  element_index;
    wire         fatal;

    tensor_layout #(.MEM_DATA_W(DW), .MAX_K_STEPS(256)) u_dut (
        .clk_core(clk), .rst_core_n(rst_n),
        .task_valid(task_valid), .task_ready(task_ready), .task_cmd(task_cmd),
        .event_valid(ev_valid), .event_ready(ev_ready), .event_data(ev_data),
        .mem_req_valid(rq_valid), .mem_req_ready(rq_ready), .mem_req(rq),
        .mem_rsp_valid(rs_valid), .mem_rsp_ready(rs_ready), .mem_rsp(rs),
        .active(active), .element_index(element_index), .fatal(fatal)
    );

    // ------------------------------------------------------------------
    // scratchpad, client 1 = VPU
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
        .buf_enable(8'h0C),
        .buf_bank_flat({32'd0, 8'd3, 8'd2, 16'd0}),
        .buf_base_flat(256'd0),
        .buf_size_flat({160'd0, 32'd8192, 32'd8192, 32'd0, 32'd0}),
        .c_req_valid(c_req_valid), .c_req_ready(c_req_ready), .c_req(c_req),
        .c_rsp_valid(c_rsp_valid), .c_rsp_ready(c_rsp_ready), .c_rsp(c_rsp),
        .outstanding_count(), .access_fault_valid(), .access_fault_info(),
        .bank_busy()
    );

    // ------------------------------------------------------------------
    // byte helpers (constant bank index per side)
    // ------------------------------------------------------------------
    integer fails;
    integer i;

    task psrc8;
        input integer addr;
        input [7:0] v;
        begin
            u_mem.g_bank[2].u_ram.mem[addr >> 5][8*(addr & 31) +: 8] = v;
        end
    endtask

    task pdst8;
        input integer addr;
        input [7:0] v;
        begin
            u_mem.g_bank[3].u_ram.mem[addr >> 5][8*(addr & 31) +: 8] = v;
        end
    endtask

    task psrc16;
        input integer addr;
        input [15:0] v;
        begin
            psrc8(addr, v[7:0]);
            psrc8(addr + 1, v[15:8]);
        end
    endtask

    task psrc32;
        input integer addr;
        input [31:0] v;
        begin
            psrc16(addr, v[15:0]);
            psrc16(addr + 2, v[31:16]);
        end
    endtask

    function [7:0] pdst8f;
        input integer addr;
        reg [255:0] w;
        begin
            w = u_mem.g_bank[3].u_ram.mem[addr >> 5];
            pdst8f = w[8*(addr & 31) +: 8];
        end
    endfunction

    function [15:0] pdst16;
        input integer addr;
        begin
            pdst16 = {pdst8f(addr + 1), pdst8f(addr)};
        end
    endfunction

    function [31:0] pdst32;
        input integer addr;
        begin
            pdst32 = {pdst16(addr + 2), pdst16(addr)};
        end
    endfunction

    task chk8;
        input integer addr;
        input [7:0] exp;
        input [8*16-1:0] what;
        reg [7:0] got;
        begin
            got = pdst8f(addr);
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
            got = pdst16(addr);
            if (got !== exp) begin
                fails = fails + 1;
                $display("FAIL %0s @%0d: %h exp %h", what, addr, got, exp);
            end
        end
    endtask

    task chk32;
        input integer addr;
        input [31:0] exp;
        input [8*16-1:0] what;
        reg [31:0] got;
        begin
            got = pdst32(addr);
            if (got !== exp) begin
                fails = fails + 1;
                $display("FAIL %0s @%0d: %h exp %h", what, addr, got, exp);
            end
        end
    endtask

    // ------------------------------------------------------------------
    // command build / run
    // ------------------------------------------------------------------
    reg [511:0] cmd;

    task build;
        input [7:0] sub, fs, fd, flags;
        input [7:0] b0, b1, b2;
        input [31:0] p0, p1, p2, p3, p4, p5;
        begin
            cmd = 512'd0;
            cmd[7:0]     = 8'h01;
            cmd[15:8]    = `NPU_OP_LAYOUT;
            cmd[23:16]   = sub;
            cmd[31:24]   = flags;
            cmd[47:32]   = 16'd21;
            cmd[71:64]   = b0;
            cmd[79:72]   = b1;
            cmd[87:80]   = b2;
            cmd[89:88]   = fs;
            cmd[93:92]   = fd;
            cmd[103:96]  = 8'd2;
            cmd[159:128] = p0;
            cmd[191:160] = p1;
            cmd[223:192] = p2;
            cmd[255:224] = p3;
            cmd[287:256] = p4;
            cmd[319:288] = p5;
        end
    endtask

    reg [15:0] ev_status;
    reg [31:0] ev_units;

    task run;
        input [15:0] exp_status;
        input [31:0] exp_units;
        input [8*24-1:0] what;
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
            if (ev_data[31:24] !== `NPU_OP_LAYOUT ||
                ev_data[23:16] !== 8'd2 || ev_data[15:0] !== 16'd21) begin
                fails = fails + 1;
                $display("FAIL %0s: event identity op=%h slot=%0d job=%0d",
                         what, ev_data[31:24], ev_data[23:16], ev_data[15:0]);
            end
            ev_ready = 1'b1;
            @(posedge clk);
            @(negedge clk);
            ev_ready = 1'b0;
        end
    endtask

    // ------------------------------------------------------------------
    // sequence
    // ------------------------------------------------------------------
    localparam T = `NPU_LAYOUT_TRANSPOSE;
    localparam P = `NPU_LAYOUT_PACK_W16;

    initial begin
        fails = 0;
        task_valid = 1'b0;
        task_cmd   = 512'd0;
        ev_ready   = 1'b0;

        repeat (4) @(posedge clk);
        rst_n = 1'b1;
        repeat (2) @(posedge clk);

        // ==============================================================
        // 1. TRANSPOSE2D FP32, 2 rows x 3 cols
        //    src 0x040 stride 0x020, dst 0x100 stride 0x020
        // ==============================================================
        for (i = 0; i < 2; i = i + 1) begin : fill1
            integer c;
            for (c = 0; c < 3; c = c + 1) begin
                psrc32(32'h040 + i*32 + c*4, 32'hA000_0000 +
                       (i*32'h100) + c);
            end
        end
        pdst8(32'h0FF, 8'hDE);
        pdst8(32'h160, 8'hBE);

        build(T, F_FP32, F_FP32, 8'h00, SRC_BUF, 8'd0, DST_BUF,
              32'd2, 32'd3, 32'h040, 32'h100, 32'h020, 32'h020);
        run(`NPU_ST_OK, 32'd6, "transpose fp32");

        for (i = 0; i < 2; i = i + 1) begin : chk1
            integer c;
            for (c = 0; c < 3; c = c + 1) begin
                chk32(32'h100 + c*32 + i*4, 32'hA000_0000 + i*32'h100 + c,
                      "td32");
            end
        end
        chk8(32'h0FF, 8'hDE, "td32 below");
        chk8(32'h160, 8'hBE, "td32 above");

        // ==============================================================
        // 2. TRANSPOSE2D FP16, single element
        // ==============================================================
        psrc16(32'h200, 16'hBEEF);
        build(T, F_FP16, F_FP16, 8'h00, SRC_BUF, 8'd0, DST_BUF,
              32'd1, 32'd1, 32'h200, 32'h240, 32'h002, 32'h002);
        run(`NPU_ST_OK, 32'd1, "transpose 1x1");
        chk16(32'h240, 16'hBEEF, "td 1x1");

        // ==============================================================
        // 3. TRANSPOSE2D FP16, 3 rows x 2 cols, odd offsets, destination
        //    pre-filled so untouched bytes must survive
        //    src 0x302 stride 6, dst 0x350 stride 6
        // ==============================================================
        for (i = 0; i < 3; i = i + 1) begin : fill3
            integer c;
            for (c = 0; c < 2; c = c + 1) begin
                psrc16(32'h302 + i*6 + c*2, 16'h1100 + i*16 + c);
            end
        end
        for (i = 32'h350; i < 32'h370; i = i + 1) pdst8(i, 8'h55);

        build(T, F_FP16, F_FP16, 8'h00, SRC_BUF, 8'd0, DST_BUF,
              32'd3, 32'd2, 32'h302, 32'h350, 32'h006, 32'h008);
        run(`NPU_ST_OK, 32'd6, "transpose fp16 odd");

        for (i = 0; i < 3; i = i + 1) begin : chk3
            integer c;
            for (c = 0; c < 2; c = c + 1) begin
                chk16(32'h350 + c*8 + i*2, 16'h1100 + i*16 + c, "td16");
            end
        end
        // destination stride 8 over 6 written bytes leaves a 2-byte gap per
        // column; the pre-fill there must survive
        chk8(32'h350 + 6, 8'h55, "td16 gap 0");
        chk8(32'h350 + 7, 8'h55, "td16 gap 1");
        chk8(32'h358 + 6, 8'h55, "td16 gap 2");
        chk8(32'h36E, 8'h55, "td16 tail");

        // ==============================================================
        // 4. PACK_W16 K=3 N=3 g=1
        //    src 0x400 stride 8, dst 0x500
        // ==============================================================
        for (i = 0; i < 3; i = i + 1) begin : fill4
            integer n;
            for (n = 0; n < 3; n = n + 1) begin
                psrc16(32'h400 + i*8 + n*2, 16'h0100 + i*4 + n);
            end
        end
        for (i = 32'h4F0; i < 32'h530; i = i + 1) pdst8(i, 8'h00);

        build(P, F_FP16, F_FP16, 8'h00, SRC_BUF, 8'd0, DST_BUF,
              32'd3, 32'd3, 32'h400, 32'h500, 32'h008, 32'd1);
        run(`NPU_ST_OK, 32'd9, "pack 3x3");

        for (i = 0; i < 4; i = i + 1) begin : chk4i
            integer p;
            for (p = 0; p < 4; p = p + 1) begin
                if (i < 3 && p < 3)
                    chk16(32'h500 + p*8 + i*2, 16'h0100 + i*4 + p, "pack3");
                else
                    chk16(32'h500 + p*8 + i*2, 16'h0000, "pack3 pad");
            end
        end
        chk8(32'h520, 8'h00, "pack3 above");

        // ==============================================================
        // 5. PACK_W16 K=5 N=4 g=2 -> S=2, 128 destination bytes
        // ==============================================================
        for (i = 0; i < 5; i = i + 1) begin : fill5
            integer n;
            for (n = 0; n < 4; n = n + 1) begin
                psrc16(32'h700 + i*16 + n*2, 16'h2000 + i*8 + n);
            end
        end
        for (i = 32'h600; i < 32'h690; i = i + 1) pdst8(i, 8'hA5);

        build(P, F_FP16, F_FP16, 8'h00, SRC_BUF, 8'd0, DST_BUF,
              32'd5, 32'd4, 32'h700, 32'h600, 32'h010, 32'd2);
        run(`NPU_ST_OK, 32'd20, "pack 5x4 g2");

        for (i = 0; i < 2; i = i + 1) begin : chk5j
            integer s;
            for (s = 0; s < 2; s = s + 1) begin : chk5s
                integer p, ii, k, n, off;
                for (p = 0; p < 4; p = p + 1) begin
                    for (ii = 0; ii < 4; ii = ii + 1) begin
                        k   = 4*s + ii;
                        n   = 4*i + p;
                        off = 32'h600 + 32*(i*2 + s) + 8*p + 2*ii;
                        if (k < 5 && n < 4)
                            chk16(off, 16'h2000 + k*8 + n, "pack5");
                        else
                            chk16(off, 16'h0000, "pack5 pad");
                    end
                end
            end
        end
        chk8(32'h680, 8'hA5, "pack5 above");

        // ==============================================================
        // 6. rejections
        // ==============================================================
        build(T, F_FP16, F_FP16, 8'h02, SRC_BUF, 8'd0, DST_BUF,   // INPLACE
              32'd1, 32'd1, 32'h200, 32'h240, 32'h002, 32'h002);
        run(`NPU_ST_BAD_FIELD, 32'd0, "reject inplace");

        build(T, F_FP16, F_FP16, 8'h00, SRC_BUF, 8'd1, DST_BUF,   // src1 set
              32'd1, 32'd1, 32'h200, 32'h240, 32'h002, 32'h002);
        run(`NPU_ST_BAD_FIELD, 32'd0, "reject src1");

        build(8'd2, F_FP16, F_FP16, 8'h00, SRC_BUF, 8'd0, DST_BUF,
              32'd1, 32'd1, 32'h200, 32'h240, 32'h002, 32'h002);
        run(`NPU_ST_BAD_FIELD, 32'd0, "reject subop");

        build(P, F_FP32, F_FP32, 8'h00, SRC_BUF, 8'd0, DST_BUF,
              32'd4, 32'd4, 32'h200, 32'h240, 32'h020, 32'd1);
        run(`NPU_ST_BAD_FIELD, 32'd0, "reject pack fp32");

        build(P, F_FP16, F_FP16, 8'h00, SRC_BUF, 8'd0, DST_BUF,
              32'd0, 32'd4, 32'h200, 32'h240, 32'h020, 32'd1);
        run(`NPU_ST_BAD_FIELD, 32'd0, "reject pack K=0");

        build(P, F_FP16, F_FP16, 8'h00, SRC_BUF, 8'd0, DST_BUF,
              32'd4, 32'd4, 32'h200, 32'h240, 32'h020, 32'd0);
        run(`NPU_ST_BAD_FIELD, 32'd0, "reject g=0");

        build(P, F_FP16, F_FP16, 8'h00, SRC_BUF, 8'd0, DST_BUF,
              32'd4, 32'd4, 32'h200, 32'h240, 32'h020, 32'd9);
        run(`NPU_ST_BAD_FIELD, 32'd0, "reject g=9");

        build(P, F_FP16, F_FP16, 8'h00, SRC_BUF, 8'd0, DST_BUF,
              32'd4, 32'd5, 32'h200, 32'h240, 32'h020, 32'd1);  // N > 4g
        run(`NPU_ST_BAD_FIELD, 32'd0, "reject N>4g");

        build(P, F_FP16, F_FP16, 8'h00, SRC_BUF, 8'd0, DST_BUF,
              32'd4, 32'd4, 32'h200, 32'h244, 32'h020, 32'd1);  // dst +4
        run(`NPU_ST_ALIGN, 32'd0, "reject dst align");

        build(P, F_FP16, F_FP16, 8'h00, SRC_BUF, 8'd0, DST_BUF,
              32'd4, 32'd4, 32'h200, 32'h240, 32'h006, 32'd1);  // stride < 8
        run(`NPU_ST_BAD_FIELD, 32'd0, "reject short stride");

        build(T, F_FP32, F_FP32, 8'h00, SRC_BUF, 8'd0, DST_BUF,
              32'd2, 32'd3, 32'h042, 32'h100, 32'h020, 32'h020);
        run(`NPU_ST_ALIGN, 32'd0, "reject src align");

        build(T, F_FP16, F_FP16, 8'h00, SRC_BUF, 8'd0, DST_BUF,
              32'd2, 32'd3, 32'h040, 32'h100, 32'h004, 32'h020); // stride < 6
        run(`NPU_ST_BAD_FIELD, 32'd0, "reject td stride");

        build(T, F_FP16, F_FP16, 8'h00, SRC_BUF, 8'd0, DST_BUF,
              32'd1, 32'd1, 32'h200, 32'h240, 32'h002, 32'h002);
        cmd[351:320] = 32'd1;                                     // P6 dirty
        run(`NPU_ST_BAD_FIELD, 32'd0, "reject p6");

        build(T, F_INT4, F_INT4, 8'h00, SRC_BUF, 8'd0, DST_BUF,
              32'd1, 32'd1, 32'h200, 32'h240, 32'h002, 32'h002);
        run(`NPU_ST_UNSUPPORTED, 32'd0, "reject int4");

        build(T, F_FP16, F_FP32, 8'h00, SRC_BUF, 8'd0, DST_BUF,
              32'd1, 32'd1, 32'h200, 32'h240, 32'h002, 32'h002);
        run(`NPU_ST_BAD_FIELD, 32'd0, "reject format mix");

        if (fatal !== 1'b0) begin
            fails = fails + 1;
            $display("FAIL: fatal raised");
        end

        if (fails == 0) $display("PASS: tb_tensor_layout");
        else            $display("FAIL: tb_tensor_layout (%0d)", fails);
        $finish;
    end

    initial begin
        #200000;
        $display("FAIL: tb_tensor_layout (timeout)");
        $finish;
    end

endmodule

`default_nettype wire

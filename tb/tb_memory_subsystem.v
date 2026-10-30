//=============================================================================
// tb_memory_subsystem : self-checking testbench
//
//   Covers: write/read round trip, byte strobes, error responses
//   (alignment, range, disabled buffer, unknown buffer), round-robin
//   arbitration of one bank, parallel service of distinct banks, and the
//   outstanding/bank_busy bookkeeping.
//
//   Buffer i -> bank i, base_word = 64*i, size = 512 bytes, so a 256-bit
//   word is 32 bytes and each buffer holds 16 words.
//=============================================================================
`timescale 1ns/1ps
`default_nettype none

`include "npu_defs.vh"

module tb_memory_subsystem;

    localparam DW    = 256;
    localparam B     = 32;
    localparam REQ_W = `NPU_REQ_W(DW, B);
    localparam RSP_W = `NPU_RSP_W(DW);
    localparam L_RAM = 2;

    reg  clk, rst_n;
    initial clk = 1'b0;
    always #5 clk = ~clk;

    reg  [7:0]   buf_enable;
    reg  [63:0]  buf_bank_flat;
    reg  [255:0] buf_base_flat;
    reg  [255:0] buf_size_flat;

    // Per-client storage: concurrent branches must never read-modify-write a
    // shared vector, so each client owns its request slot and valid/ready bit.
    reg  [REQ_W-1:0] req_q [0:4];
    reg              rsp_rdy [0:4];
    reg              req_vld [0:4];

    wire [4:0]   c_req_ready;
    wire [4:0]   c_rsp_valid;
    wire [5*RSP_W-1:0] c_rsp;

    reg  [4:0]        c_req_valid;
    reg  [5*REQ_W-1:0] c_req;
    reg  [4:0]        c_rsp_ready;

    integer fj;
    always @* begin
        for (fj = 0; fj < 5; fj = fj + 1) begin
            c_req_valid[fj] = req_vld[fj];
            c_rsp_ready[fj] = rsp_rdy[fj];
            c_req[fj*REQ_W +: REQ_W] = req_q[fj];
        end
    end

    wire [15:0]  outstanding_count;
    wire         access_fault_valid;
    wire [63:0]  access_fault_info;
    wire [7:0]   bank_busy;

    integer fails;
    integer i;

    always @(posedge clk)
        if (access_fault_valid)
            $display("[%0t] access fault: client=%0d buf=%0d off=%0d reason=%0d",
                     $time,
                     access_fault_info[50:48],
                     access_fault_info[47:40],
                     access_fault_info[39:8],
                     access_fault_info[7:0]);

    memory_subsystem #(
        .MEM_DATA_W(DW), .BANKS(8), .BANK_DEPTH(1024), .L_RAM(L_RAM)
    ) dut (
        .clk_core(clk), .rst_core_n(rst_n),
        .buf_enable(buf_enable),
        .buf_bank_flat(buf_bank_flat),
        .buf_base_flat(buf_base_flat),
        .buf_size_flat(buf_size_flat),
        .c_req_valid(c_req_valid), .c_req_ready(c_req_ready),
        .c_req(c_req), .c_rsp_valid(c_rsp_valid),
        .c_rsp_ready(c_rsp_ready), .c_rsp(c_rsp),
        .outstanding_count(outstanding_count),
        .access_fault_valid(access_fault_valid),
        .access_fault_info(access_fault_info),
        .bank_busy(bank_busy)
    );

    // ------------------------------------------------------------------
    // helpers
    // ------------------------------------------------------------------
    task automatic issue;
        input integer ci;
        input         wr;
        input [7:0]   b;
        input [31:0]  off;
        input [DW-1:0] data;
        input [B-1:0]  strb;
        input [7:0]   tag;
        begin
            @(negedge clk);
            req_q[ci]   = {tag, strb, data, off, b, wr};
            req_vld[ci] = 1'b1;
            #1;
            while (c_req_ready[ci] !== 1'b1) begin
                @(negedge clk);
                #1;
            end
            @(posedge clk);
            @(negedge clk);
            #1;
            req_vld[ci] = 1'b0;
        end
    endtask

    task automatic expect_rsp;
        input integer  ci;
        input          exp_err;
        input [7:0]    exp_reason;
        input [DW-1:0] exp_data;
        input [7:0]    exp_tag;
        integer sh;
        reg [DW-1:0] d;
        reg [7:0]    t, r;
        reg          e;
        begin
            sh = ci * RSP_W;
            #1;
            while (c_rsp_valid[ci] !== 1'b1) begin
                @(negedge clk);
                #1;
            end
            d = c_rsp[sh + 0 +: DW];
            t = c_rsp[sh + DW +: 8];
            e = c_rsp[sh + DW + 8];
            r = c_rsp[sh + DW + 9 +: 8];
            if (e !== exp_err) begin
                fails = fails + 1;
                $display("FAIL clk=%0d c%0d err=%0b exp=%0b (reason %0d)",
                         $time, ci, e, exp_err, r);
            end
            if (exp_err && (r !== exp_reason)) begin
                fails = fails + 1;
                $display("FAIL clk=%0d c%0d reason=%0d exp=%0d",
                         $time, ci, r, exp_reason);
            end
            if (t !== exp_tag) begin
                fails = fails + 1;
                $display("FAIL clk=%0d c%0d tag=%0h exp=%0h",
                         $time, ci, t, exp_tag);
            end
            if (!exp_err && (d !== exp_data)) begin
                fails = fails + 1;
                $display("FAIL clk=%0d c%0d data=%h exp=%h",
                         $time, ci, d, exp_data);
            end
            rsp_rdy[ci] = 1'b1;
            @(posedge clk);
            @(negedge clk);
            #1;
            rsp_rdy[ci] = 1'b0;
        end
    endtask

    task automatic expect_idle;
        input integer n;
        integer k;
        begin
            for (k = 0; k < n; k = k + 1) @(negedge clk);
            #1;
            if (outstanding_count !== 16'd0) begin
                fails = fails + 1;
                $display("FAIL outstanding_count=%0d, exp 0", outstanding_count);
            end
            if (bank_busy !== 8'h00) begin
                fails = fails + 1;
                $display("FAIL bank_busy=%0h, exp 00", bank_busy);
            end
        end
    endtask

    // round-robin / parallel-issue observation on bank 0
    reg [4:0] bank0_mask;
    reg       par_seen;
    always @(posedge clk) begin
        if (dut.bank_fire[0]) begin
            bank0_mask[dut.bank_sel[0]] <= 1'b1;
            $display("[%0t] bank0 grant -> client %0d", $time, dut.bank_sel[0]);
        end
        if (dut.bank_fire[0] && dut.bank_fire[1] && dut.bank_fire[2])
            par_seen <= 1'b1;
    end

    // ------------------------------------------------------------------
    // stimulus
    // ------------------------------------------------------------------
    initial begin
        fails = 0;
        bank0_mask = 5'd0;
        par_seen   = 1'b0;

        buf_enable    = 8'hFF;
        buf_bank_flat = 64'h0706050403020100;
        buf_base_flat = 256'd0;
        buf_size_flat = 256'd0;
        for (i = 0; i < 8; i = i + 1) begin
            buf_base_flat[32*i +: 32] = 64 * i;
            buf_size_flat[32*i +: 32] = 512;
        end

        for (i = 0; i < 5; i = i + 1) begin
            req_vld[i] = 1'b0;
            rsp_rdy[i] = 1'b0;
            req_q[i]   = {REQ_W{1'b0}};
        end
        rst_n        = 1'b0;
        repeat (4) @(negedge clk);
        rst_n = 1'b1;
        @(negedge clk);

        // ---------------- 1. write / read round trip -------------------
        $display("-- test 1: write/read");
        issue(0, 1'b1, 8'd0, 32'd0, {8{32'h11223344}}, {B{1'b1}}, 8'h10);
        expect_rsp(0, 1'b0, 8'd0, {DW{1'b0}}, 8'h10);

        issue(1, 1'b0, 8'd0, 32'd0, {DW{1'b0}}, {B{1'b0}}, 8'h11);
        expect_rsp(1, 1'b0, 8'd0, {8{32'h11223344}}, 8'h11);

        // ---------------- 2. byte strobes ------------------------------
        $display("-- test 2: byte strobes");
        issue(2, 1'b1, 8'd1, 32'd0, {DW{1'b1}}, {B{1'b1}}, 8'h20);
        expect_rsp(2, 1'b0, 8'd0, {DW{1'b0}}, 8'h20);

        issue(2, 1'b1, 8'd1, 32'd0, 256'hAA, 32'h0000_0001, 8'h21);
        expect_rsp(2, 1'b0, 8'd0, {DW{1'b0}}, 8'h21);

        issue(2, 1'b0, 8'd1, 32'd0, {DW{1'b0}}, {B{1'b0}}, 8'h22);
        expect_rsp(2, 1'b0, 8'd0, {{248{1'b1}}, 8'hAA}, 8'h22);

        // ---------------- 3. alignment ---------------------------------
        $display("-- test 3: alignment");
        issue(3, 1'b0, 8'd0, 32'd16, {DW{1'b0}}, {B{1'b0}}, 8'h30);
        expect_rsp(3, 1'b1, `NPU_MEMR_ALIGN, {DW{1'b0}}, 8'h30);

        // ---------------- 4. range -------------------------------------
        $display("-- test 4: range");
        issue(3, 1'b0, 8'd0, 32'd512, {DW{1'b0}}, {B{1'b0}}, 8'h40);
        expect_rsp(3, 1'b1, `NPU_MEMR_RANGE, {DW{1'b0}}, 8'h40);

        // last legal word of the buffer is off 480
        issue(3, 1'b1, 8'd0, 32'd480, {DW{1'b1}}, {B{1'b1}}, 8'h41);
        expect_rsp(3, 1'b0, 8'd0, {DW{1'b0}}, 8'h41);
        issue(3, 1'b0, 8'd0, 32'd480, {DW{1'b0}}, {B{1'b0}}, 8'h42);
        expect_rsp(3, 1'b0, 8'd0, {DW{1'b1}}, 8'h42);

        // ---------------- 5. disabled buffer ---------------------------
        $display("-- test 5: disabled buffer");
        buf_enable[2] = 1'b0;
        issue(0, 1'b0, 8'd2, 32'd0, {DW{1'b0}}, {B{1'b0}}, 8'h50);
        expect_rsp(0, 1'b1, `NPU_MEMR_BUF, {DW{1'b0}}, 8'h50);
        buf_enable[2] = 1'b1;

        // ---------------- 6. unknown buffer ----------------------------
        $display("-- test 6: unknown buffer id");
        issue(0, 1'b0, 8'd9, 32'd0, {DW{1'b0}}, {B{1'b0}}, 8'h60);
        expect_rsp(0, 1'b1, `NPU_MEMR_BUF, {DW{1'b0}}, 8'h60);

        // ---------------- 7. parallel banks ----------------------------
        $display("-- test 7: parallel distinct banks");
        fork
            issue(0, 1'b1, 8'd0, 32'd0, {DW{1'b1}},        {B{1'b1}}, 8'h70);
            issue(1, 1'b1, 8'd1, 32'd0, {DW{1'b1}} >> 1,   {B{1'b1}}, 8'h71);
            issue(2, 1'b1, 8'd2, 32'd0, {DW{1'b1}} >> 2,   {B{1'b1}}, 8'h72);
        join
        fork
            expect_rsp(0, 1'b0, 8'd0, {DW{1'b0}}, 8'h70);
            expect_rsp(1, 1'b0, 8'd0, {DW{1'b0}}, 8'h71);
            expect_rsp(2, 1'b0, 8'd0, {DW{1'b0}}, 8'h72);
        join
        if (!par_seen) begin
            fails = fails + 1;
            $display("FAIL: banks 0/1/2 never fired in the same cycle");
        end

        // ---------------- 8. bank contention ---------------------------
        // The bank holds its response until the owning client takes it, so
        // each branch must drain its own response to let the bank move on.
        $display("-- test 8: four clients, one bank");
        bank0_mask = 5'd0;
        fork
            begin
                issue(0, 1'b1, 8'd0, 32'd64,  {DW{32'hA0A0A0A0}}, {B{1'b1}}, 8'h80);
                expect_rsp(0, 1'b0, 8'd0, {DW{1'b0}}, 8'h80);
            end
            begin
                issue(1, 1'b1, 8'd0, 32'd96,  {DW{32'hB1B1B1B1}}, {B{1'b1}}, 8'h81);
                expect_rsp(1, 1'b0, 8'd0, {DW{1'b0}}, 8'h81);
            end
            begin
                issue(2, 1'b1, 8'd0, 32'd128, {DW{32'hC2C2C2C2}}, {B{1'b1}}, 8'h82);
                expect_rsp(2, 1'b0, 8'd0, {DW{1'b0}}, 8'h82);
            end
            begin
                issue(3, 1'b1, 8'd0, 32'd160, {DW{32'hD3D3D3D3}}, {B{1'b1}}, 8'h83);
                expect_rsp(3, 1'b0, 8'd0, {DW{1'b0}}, 8'h83);
            end
        join
        if (bank0_mask !== 5'b0_1111) begin
            fails = fails + 1;
            $display("FAIL: bank0 served mask %0b, exp 01111", bank0_mask);
        end

        bank0_mask = 5'd0;
        fork
            begin
                issue(0, 1'b0, 8'd0, 32'd64,  {DW{1'b0}}, {B{1'b0}}, 8'h90);
                expect_rsp(0, 1'b0, 8'd0, {DW{32'hA0A0A0A0}}, 8'h90);
            end
            begin
                issue(1, 1'b0, 8'd0, 32'd96,  {DW{1'b0}}, {B{1'b0}}, 8'h91);
                expect_rsp(1, 1'b0, 8'd0, {DW{32'hB1B1B1B1}}, 8'h91);
            end
            begin
                issue(2, 1'b0, 8'd0, 32'd128, {DW{1'b0}}, {B{1'b0}}, 8'h92);
                expect_rsp(2, 1'b0, 8'd0, {DW{32'hC2C2C2C2}}, 8'h92);
            end
            begin
                issue(3, 1'b0, 8'd0, 32'd160, {DW{1'b0}}, {B{1'b0}}, 8'h93);
                expect_rsp(3, 1'b0, 8'd0, {DW{32'hD3D3D3D3}}, 8'h93);
            end
        join
        if (bank0_mask !== 5'b0_1111) begin
            fails = fails + 1;
            $display("FAIL: bank0 readback served mask %0b, exp 01111", bank0_mask);
        end

        // ---------------- 9. response backpressure ---------------------
        $display("-- test 9: response held under backpressure");
        issue(4, 1'b1, 8'd4, 32'd0, {DW{32'h5A5A5A5A}}, {B{1'b1}}, 8'hA0);
        repeat (6) @(negedge clk);
        #1;
        if (outstanding_count !== 16'd1) begin
            fails = fails + 1;
            $display("FAIL: outstanding_count=%0d during held response",
                     outstanding_count);
        end
        expect_rsp(4, 1'b0, 8'd0, {DW{1'b0}}, 8'hA0);

        expect_idle(4);

        if (fails == 0) $display("PASS: tb_memory_subsystem");
        else            $display("FAIL: tb_memory_subsystem (%0d failures)", fails);
        $finish;
    end

    initial begin
        #50000;
        $display("FAIL: timeout");
        $display("  client_busy=%b vld=%b rdy=%b rspv=%b",
                 dut.client_busy, {req_vld[4],req_vld[3],req_vld[2],req_vld[1],req_vld[0]},
                 {rsp_rdy[4],rsp_rdy[3],rsp_rdy[2],rsp_rdy[1],rsp_rdy[0]},
                 c_rsp_valid);
        for (i = 0; i < 8; i = i + 1)
            $display("  bank%0d state=%0d owner=%0d rr=%0d gnt=%b sel=%0d busy=%b",
                     i, dut.bstate[i], dut.bowner[i], dut.brr[i],
                     dut.bank_gnt[i], dut.bank_sel[i], bank_busy[i]);
        $finish;
    end

endmodule

`default_nettype wire

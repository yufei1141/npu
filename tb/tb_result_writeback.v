//=============================================================================
// tb_result_writeback : slots, conversion and strobed destination writes
//
//   The DUT is connected to the real memory_subsystem (client 3).  The
//   destination buffer is pre-filled with a sentinel byte, the job is run,
//   and the whole image is compared against a byte model.  Covers:
//     - FP32 / FP16 / BF16 destination narrowing
//     - element- but not word-aligned destination base
//     - rows that share a memory word (sentinel bytes must survive)
//     - N not a multiple of the tile width (last tile is short)
//     - capture while the serializer is still writing an earlier tile
//     - done held under back pressure
//     - a write that fails in the memory subsystem (status MEM_ACCESS + fatal)
//=============================================================================
`timescale 1ns/1ps
`default_nettype none

`include "npu_defs.vh"

module tb_result_writeback;

    localparam SLOTS  = 4;
    localparam DW     = 256;
    localparam B      = 32;
    localparam REQ_W  = `NPU_REQ_W(DW, B);
    localparam RSP_W  = `NPU_RSP_W(DW);
    localparam F_FP16 = 2'd0;
    localparam F_BF16 = 2'd1;
    localparam F_FP32 = 2'd2;
    localparam DSTBUF = 8'd3;
    localparam IMG    = 8192;

    reg clk, rst_n;
    initial begin clk = 0; rst_n = 0; end
    always #5 clk = ~clk;

    // ------------------------------------------------------------------
    // DUT
    // ------------------------------------------------------------------
    reg         job_valid;
    wire        job_ready;
    reg  [511:0] job_cmd;

    reg          acc_valid;
    reg  [4095:0] acc_data;

    wire         rq_valid;
    wire         rq_ready;
    wire [REQ_W-1:0] rq;
    wire         rs_valid;
    wire         rs_ready;
    wire [RSP_W-1:0] rs;

    wire         done_valid;
    reg          done_ready;
    wire [15:0]  done_status;
    wire [31:0]  done_tiles;
    wire [2:0]   reserved_count, occupied_count;
    wire         fatal;

    result_writeback #(
        .MEM_DATA_W(DW), .RESULT_SLOTS(SLOTS)
    ) u_dut (
        .clk_core(clk), .rst_core_n(rst_n),
        .job_valid(job_valid), .job_ready(job_ready), .job_cmd(job_cmd),
        .acc_valid(acc_valid), .acc_data(acc_data),
        .mem_req_valid(rq_valid), .mem_req_ready(rq_ready), .mem_req(rq),
        .mem_rsp_valid(rs_valid), .mem_rsp_ready(rs_ready), .mem_rsp(rs),
        .done_valid(done_valid), .done_ready(done_ready),
        .done_status(done_status), .done_tiles(done_tiles),
        .reserved_count(reserved_count), .occupied_count(occupied_count),
        .fatal(fatal)
    );

    // ------------------------------------------------------------------
    // memory subsystem, client 3 = RESULT_WB, buffer 3 -> bank 3
    // ------------------------------------------------------------------
    wire [4:0] c_req_valid = {1'b0, rq_valid, 1'b0, 1'b0, 1'b0};
    wire [4:0] c_req_ready;
    wire [5*REQ_W-1:0] c_req;
    wire [4:0] c_rsp_valid;
    wire [4:0] c_rsp_ready = {1'b0, rs_ready, 1'b0, 1'b0, 1'b0};
    wire [5*RSP_W-1:0] c_rsp;

    assign c_req[3*REQ_W +: REQ_W] = rq;
    assign c_req[0*REQ_W +: REQ_W] = {REQ_W{1'b0}};
    assign c_req[1*REQ_W +: REQ_W] = {REQ_W{1'b0}};
    assign c_req[2*REQ_W +: REQ_W] = {REQ_W{1'b0}};
    assign c_req[4*REQ_W +: REQ_W] = {REQ_W{1'b0}};

    assign rq_ready = c_req_ready[3];
    assign rs_valid = c_rsp_valid[3];
    assign rs       = c_rsp[3*RSP_W +: RSP_W];

    memory_subsystem #(
        .MEM_DATA_W(DW), .BANKS(8), .BANK_DEPTH(1024), .L_RAM(1)
    ) u_mem (
        .clk_core(clk), .rst_core_n(rst_n),
        .buf_enable(8'h08),
        .buf_bank_flat({32'd0, 8'd3, 24'd0}),
        .buf_base_flat({32'd0, 32'd0, 192'd0}),
        .buf_size_flat({128'd0, 32'd8192, 96'd0}),
        .c_req_valid(c_req_valid), .c_req_ready(c_req_ready), .c_req(c_req),
        .c_rsp_valid(c_rsp_valid), .c_rsp_ready(c_rsp_ready), .c_rsp(c_rsp),
        .outstanding_count(), .access_fault_valid(), .access_fault_info(),
        .bank_busy()
    );

    // ------------------------------------------------------------------
    // byte model
    // ------------------------------------------------------------------
    reg [7:0] exp_img [0:IMG-1];
    reg [7:0] sentinel;

    integer fails;
    integer a;

    task set_ram_byte;
        input integer addr;
        input [7:0] v;
        begin
            u_mem.g_bank[3].u_ram.mem[addr >> 5][8*(addr & 31) +: 8] = v;
        end
    endtask

    function [7:0] get_ram_byte;
        input integer addr;
        reg [255:0] w;
        begin
            w = u_mem.g_bank[3].u_ram.mem[addr >> 5];
            get_ram_byte = w[8*(addr & 31) +: 8];
        end
    endfunction

    function [31:0] mkval;
        input integer fmt;
        input integer idx;
        begin
            if (fmt == F_BF16) mkval = 32'h3F80_0000 | (idx << 16);
            else               mkval = 32'h3F80_0000 | (idx << 13);
        end
    endfunction

    // expected destination bits, low half for 16-bit formats
    function [31:0] mkexp;
        input integer fmt;
        input integer idx;
        begin
            if (fmt == F_FP32)      mkexp = 32'h3F80_0000 | (idx << 13);
            else if (fmt == F_BF16) mkexp = {16'd0, 16'h3F80 | idx[6:0]};
            else                    mkexp = {16'd0, 16'h3C00 | idx[9:0]};
        end
    endfunction

    function integer elem_bytes;
        input integer fmt;
        begin
            elem_bytes = (fmt == F_FP32) ? 4 : 2;
        end
    endfunction

    // ------------------------------------------------------------------
    // job drive
    // ------------------------------------------------------------------
    task send_job;
        input [15:0] tiles;
        input [2:0]  mlen;
        input [3:0]  g;
        input [31:0] nvalid;
        input [31:0] base;
        input [31:0] stride;
        input [1:0]  fmt;
        input [7:0]  dbuf;
        begin
            @(negedge clk);
            job_cmd = 512'd0;
            job_cmd[7:0]     = 8'h01;
            job_cmd[15:8]    = `NPU_OP_MATMUL;
            job_cmd[47:32]   = 16'h0001;
            job_cmd[87:80]   = dbuf;
            job_cmd[95:88]   = {2'b00, fmt, 4'b0000};
            job_cmd[130:128] = mlen;
            job_cmd[139:136] = g;
            job_cmd[175:160] = tiles;
            job_cmd[191:176] = 16'd3;
            job_cmd[223:192] = 32'd48;
            job_cmd[255:224] = nvalid;
            job_cmd[383:352] = base;
            job_cmd[415:384] = stride;
            job_valid = 1'b1;
            // ready depends on the payload through d_tiles: let the
            // combinational decode settle before testing it
            #1;
            if (job_ready !== 1'b1) begin
                @(negedge clk);
                while (job_ready !== 1'b1) @(negedge clk);
            end
            @(posedge clk);
            @(negedge clk);
            job_valid = 1'b0;
        end
    endtask

    function integer tile_cols;
        input integer g;
        input integer nvalid;
        input integer t;
        integer rem;
        begin
            rem = nvalid - 4*g*t;
            tile_cols = (rem >= 4*g) ? 4*g : ((rem > 0) ? rem : 0);
        end
    endfunction

    // one result pulse for tile t, all 4 rows x 32 columns; only the valid
    // subset carries meaningful values, the rest is poison
    task pulse_tile;
        input integer fmt;
        input integer t;
        integer m, c, idx;
        begin
            @(negedge clk);
            for (m = 0; m < 4; m = m + 1)
                for (c = 0; c < 32; c = c + 1) begin
                    idx = t*64 + m*32 + c;
                    acc_data[32*(32*m + c) +: 32] = mkval(fmt, idx);
                end
            acc_valid = 1'b1;
            @(posedge clk);
            @(negedge clk);
            acc_valid = 1'b0;
            acc_data = 4096'd0;
        end
    endtask

    // build the byte-exact expected image for one job
    task build_image;
        input integer fmt;
        input integer tiles;
        input integer mlen;
        input integer g;
        input integer nvalid;
        input integer base;
        input integer stride;
        integer t, m, c, idx, addr, nb, k;
        reg [31:0] v;
        begin
            for (a = 0; a < IMG; a = a + 1) exp_img[a] = get_ram_byte(a);
            nb = elem_bytes(fmt);
            for (t = 0; t < tiles; t = t + 1)
                for (m = 0; m < mlen; m = m + 1)
                    for (c = 0; c < tile_cols(g, nvalid, t); c = c + 1) begin
                        idx  = t*64 + m*32 + c;
                        v    = mkexp(fmt, idx);
                        addr = base + m*stride + (4*g*t + c)*nb;
                        for (k = 0; k < nb; k = k + 1)
                            exp_img[addr + k] = v[8*k +: 8];
                    end
        end
    endtask

    task check_image;
        input integer what;
        integer bad;
        begin
            bad = 0;
            for (a = 0; a < IMG; a = a + 1) begin
                if (get_ram_byte(a) !== exp_img[a]) begin
                    if (bad < 4)
                        $display("  mismatch @%0d got %h exp %h",
                                 a, get_ram_byte(a), exp_img[a]);
                    bad = bad + 1;
                end
            end
            if (bad != 0) begin
                fails = fails + 1;
                $display("FAIL: test %0d image, %0d bytes differ", what, bad);
            end
        end
    endtask

    task expect_done;
        input [15:0] st;
        input [31:0] tl;
        input integer what;
        integer guard;
        begin
            guard = 0;
            while (done_valid !== 1'b1 && guard < 20000) begin
                @(negedge clk);
                guard = guard + 1;
            end
            if (done_valid !== 1'b1) begin
                fails = fails + 1;
                $display("FAIL: test %0d no done", what);
            end else begin
                if (done_status !== st) begin
                    fails = fails + 1;
                    $display("FAIL: test %0d status %h exp %h", what,
                             done_status, st);
                end
                if (done_tiles !== tl) begin
                    fails = fails + 1;
                    $display("FAIL: test %0d tiles %0d exp %0d", what,
                             done_tiles, tl);
                end
                // held under back pressure for a few cycles
                repeat (5) @(negedge clk);
                if (done_valid !== 1'b1) begin
                    fails = fails + 1;
                    $display("FAIL: test %0d done dropped early", what);
                end
                @(negedge clk);
                done_ready = 1'b1;
                @(negedge clk);
                done_ready = 1'b0;
            end
        end
    endtask

    task run_job;
        input integer fmt;
        input integer tiles;
        input integer mlen;
        input integer g;
        input integer nvalid;
        input integer base;
        input integer stride;
        input [7:0]   dbuf;
        input integer what;
        integer t;
        begin
            build_image(fmt, tiles, mlen, g, nvalid, base, stride);
            send_job(tiles, mlen, g, nvalid, base, stride, fmt, dbuf);
            for (t = 0; t < tiles; t = t + 1) begin
                pulse_tile(fmt, t);
                repeat (3) @(negedge clk);
            end
            expect_done(`NPU_ST_OK, tiles, what);
            repeat (3) @(negedge clk);
            check_image(what);
            if (reserved_count !== 3'd0 || occupied_count !== 3'd0) begin
                fails = fails + 1;
                $display("FAIL: test %0d counts R=%0d O=%0d", what,
                         reserved_count, occupied_count);
            end
            if (fatal !== 1'b0) begin
                fails = fails + 1;
                $display("FAIL: test %0d unexpected fatal", what);
            end
        end
    endtask

    // ------------------------------------------------------------------
    // stimulus
    // ------------------------------------------------------------------
    initial begin
        fails = 0;
        sentinel = 8'hEE;
        job_valid = 0; job_cmd = 0;
        acc_valid = 0; acc_data = 0;
        done_ready = 0;

        for (a = 0; a < IMG; a = a + 1) set_ram_byte(a, sentinel);

        repeat (4) @(negedge clk);
        rst_n = 1'b1;
        repeat (2) @(negedge clk);

        // A: FP32 destination, 2 rows x 8 cols, 2 tiles, N = 13
        run_job(F_FP32, 2, 2, 2, 13, 1004, 64, DSTBUF, 1);
        // B: FP16 destination, 2 rows x 3 cols, 2 tiles, N = 3,
        //    row stride 16 bytes so two rows share one memory word
        run_job(F_FP16, 2, 2, 1, 3, 1004, 16, DSTBUF, 2);
        // C: BF16 destination, 4 rows x 4 cols, 1 tile, stride 20 bytes
        run_job(F_BF16, 1, 4, 1, 4, 0, 20, DSTBUF, 3);

        // D: destination buffer is not enabled -> MEM_ACCESS + fatal
        build_image(F_FP16, 1, 1, 1, 2, 0, 4);
        send_job(1, 1, 1, 2, 0, 4, F_FP16, 8'd5);
        pulse_tile(F_FP16, 0);
        expect_done(`NPU_ST_MEM_ACCESS, 0, 4);
        if (fatal !== 1'b1) begin
            fails = fails + 1;
            $display("FAIL: test 4 fatal not raised");
        end
        if (job_ready !== 1'b0) begin
            fails = fails + 1;
            $display("FAIL: test 4 job accepted after fatal");
        end

        if (fails == 0) $display("PASS: tb_result_writeback");
        else            $display("FAIL: tb_result_writeback (%0d)", fails);
        $finish;
    end

    // watchdog
    initial begin
        #2000000;
        $display("FAIL: tb_result_writeback timeout");
        $finish;
    end

endmodule

`default_nettype wire

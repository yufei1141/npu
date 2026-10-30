//=============================================================================
// tb_dma_engine : self-checking DMA + memory subsystem integration test
//
//   A. one-row LOAD with a 33-byte tail (two full words + a partial word)
//   B. two-row LOAD with external and local row strides
//   C. two-row STORE echoing B, checked byte by byte in the external model
//   D. LOAD_W: command handed to the weight-loader stub and data streamed
//      straight through, no scratchpad traffic
//=============================================================================
`timescale 1ns/1ps
`default_nettype none

`include "npu_defs.vh"

module tb_dma_engine;

    localparam DW    = 256;
    localparam B     = 32;
    localparam REQ_W = `NPU_REQ_W(DW, B);
    localparam RSP_W = `NPU_RSP_W(DW);

    reg clk_core, clk_io, rst_n;
    initial begin clk_core = 1'b0; clk_io = 1'b0; rst_n = 1'b0; end
    always #5 clk_core = ~clk_core;
    always #3 clk_io   = ~clk_io;

    integer fails;
    integer i, k;

    // ------------------------------------------------------------------
    // scratchpad
    // ------------------------------------------------------------------
    reg  [7:0]   buf_en;
    reg  [63:0]  buf_bank;
    reg  [255:0] buf_base;
    reg  [255:0] buf_size;

    wire [4:0]         m_req_ready, m_rsp_valid;
    reg  [4:0]         m_req_valid, m_rsp_ready;
    reg  [5*REQ_W-1:0] m_req;
    wire [5*RSP_W-1:0] m_rsp;
    wire [15:0]        outstanding;
    wire               afv;
    wire [63:0]        afi;
    wire [7:0]         bbusy;

    memory_subsystem #(
        .MEM_DATA_W(DW), .BANKS(8), .BANK_DEPTH(1024), .L_RAM(2)
    ) u_mem (
        .clk_core(clk_core), .rst_core_n(rst_n),
        .buf_enable(buf_en), .buf_bank_flat(buf_bank),
        .buf_base_flat(buf_base), .buf_size_flat(buf_size),
        .c_req_valid(m_req_valid), .c_req_ready(m_req_ready),
        .c_req(m_req), .c_rsp_valid(m_rsp_valid),
        .c_rsp_ready(m_rsp_ready), .c_rsp(m_rsp),
        .outstanding_count(outstanding),
        .access_fault_valid(afv), .access_fault_info(afi),
        .bank_busy(bbusy)
    );

    // ------------------------------------------------------------------
    // DMA
    // ------------------------------------------------------------------
    reg         task_valid;
    wire        task_ready;
    reg  [511:0] task_cmd;
    wire        ev_valid;
    reg         ev_ready;
    wire [127:0] ev_data;

    wire        d_req_valid, d_req_ready;
    wire [REQ_W-1:0] d_req;
    wire        d_rsp_valid, d_rsp_ready;
    wire [RSP_W-1:0] d_rsp;

    wire        rd_req_valid, rd_req_ready;
    wire [63:0] rd_req_addr;
    wire [31:0] rd_req_bytes;
    wire [15:0] rd_req_tag;
    wire        rd_valid, rd_ready;
    wire [127:0] rd_data;
    wire [15:0] rd_keep;
    wire        rd_last;
    wire [15:0] rd_tag;
    wire        rd_error;

    wire        wr_req_valid, wr_req_ready;
    wire [63:0] wr_req_addr;
    wire [31:0] wr_req_bytes;
    wire [15:0] wr_req_tag;
    wire        wr_valid, wr_ready;
    wire [127:0] wr_data;
    wire [15:0] wr_keep;
    wire        wr_last;
    wire [15:0] wr_tag;
    wire        wr_rsp_valid, wr_rsp_ready;
    wire [15:0] wr_rsp_tag;
    wire        wr_rsp_error;

    wire        wl_valid, wl_ready;
    wire [511:0] wl_cmd;
    wire        wd_valid, wd_ready;
    wire [127:0] wd_data;
    wire        wd_last, wd_error;
    wire        wdone_valid, wdone_ready;
    wire [15:0] wdone_status;
    wire        dma_fatal;

    dma_engine #(.MEM_DATA_W(DW), .EXT_WR_W(128)) u_dma (
        .clk_core(clk_core), .rst_core_n(rst_n),
        .clk_io(clk_io), .rst_io_n(rst_n),
        .task_valid(task_valid), .task_ready(task_ready), .task_cmd(task_cmd),
        .event_valid(ev_valid), .event_ready(ev_ready), .event_data(ev_data),
        .mem_req_valid(d_req_valid), .mem_req_ready(d_req_ready),
        .mem_req(d_req), .mem_rsp_valid(d_rsp_valid),
        .mem_rsp_ready(d_rsp_ready), .mem_rsp(d_rsp),
        .rd_req_valid(rd_req_valid), .rd_req_ready(rd_req_ready),
        .rd_req_addr(rd_req_addr), .rd_req_bytes(rd_req_bytes),
        .rd_req_tag(rd_req_tag), .rd_valid(rd_valid), .rd_ready(rd_ready),
        .rd_data(rd_data), .rd_keep(rd_keep), .rd_last(rd_last),
        .rd_tag(rd_tag), .rd_error(rd_error),
        .wr_req_valid(wr_req_valid), .wr_req_ready(wr_req_ready),
        .wr_req_addr(wr_req_addr), .wr_req_bytes(wr_req_bytes),
        .wr_req_tag(wr_req_tag), .wr_valid(wr_valid), .wr_ready(wr_ready),
        .wr_data(wr_data), .wr_keep(wr_keep), .wr_last(wr_last),
        .wr_tag(wr_tag), .wr_rsp_valid(wr_rsp_valid),
        .wr_rsp_ready(wr_rsp_ready), .wr_rsp_tag(wr_rsp_tag),
        .wr_rsp_error(wr_rsp_error),
        .wload_valid(wl_valid), .wload_ready(wl_ready), .wload_cmd(wl_cmd),
        .wdata_valid(wd_valid), .wdata_ready(wd_ready), .wdata(wd_data),
        .wdata_last(wd_last), .wdata_error(wd_error),
        .wload_done_valid(wdone_valid), .wload_done_ready(wdone_ready),
        .wload_status(wdone_status), .fatal(dma_fatal)
    );

    // client 4 is the DMA core bridge; the other clients stay idle
    always @* begin
        m_req_valid   = 5'd0;
        m_req_valid[4] = d_req_valid;
        m_req = {(5*REQ_W){1'b0}};
        m_req[4*REQ_W +: REQ_W] = d_req;
        m_rsp_ready = 5'b11111;
    end
    assign d_req_ready = m_req_ready[4];
    assign d_rsp_valid = m_rsp_valid[4];
    assign d_rsp       = m_rsp[4*RSP_W +: RSP_W];

    // ------------------------------------------------------------------
    // external memory
    // ------------------------------------------------------------------
    extmem_model #(.MEM_BYTES(65536), .WR_RSP_DELAY(3)) u_ext (
        .clk(clk_io), .rst_n(rst_n),
        .rd_req_valid(rd_req_valid), .rd_req_ready(rd_req_ready),
        .rd_req_addr(rd_req_addr), .rd_req_bytes(rd_req_bytes),
        .rd_req_tag(rd_req_tag), .rd_valid(rd_valid), .rd_ready(rd_ready),
        .rd_data(rd_data), .rd_keep(rd_keep), .rd_last(rd_last),
        .rd_tag(rd_tag),
        .wr_req_valid(wr_req_valid), .wr_req_ready(wr_req_ready),
        .wr_req_addr(wr_req_addr), .wr_req_bytes(wr_req_bytes),
        .wr_req_tag(wr_req_tag), .wr_valid(wr_valid), .wr_ready(wr_ready),
        .wr_data(wr_data), .wr_keep(wr_keep), .wr_last(wr_last),
        .wr_tag(wr_tag), .wr_rsp_valid(wr_rsp_valid),
        .wr_rsp_ready(wr_rsp_ready), .wr_rsp_tag(wr_rsp_tag),
        .wr_rsp_error(wr_rsp_error)
    );

    // ------------------------------------------------------------------
    // weight-loader stub (io domain)
    // ------------------------------------------------------------------
    reg         wl_ready_r, wd_ready_r, wdone_valid_r;
    reg [15:0]  wdone_status_r;
    reg [511:0] wl_cmd_cap;
    reg [511:0] wl_data_cap;
    reg [3:0]   wl_beats;
    reg         wl_last_seen;
    reg         wl_pending;

    assign wl_ready      = wl_ready_r;
    assign wd_ready      = wd_ready_r;
    assign wdone_valid   = wdone_valid_r;
    assign wdone_status  = wdone_status_r;

    always @(posedge clk_io or negedge rst_n) begin
        if (!rst_n) begin
            wdone_valid_r <= 1'b0;
            wdone_status_r <= `NPU_ST_OK;
            wl_beats <= 4'd0;
            wl_last_seen <= 1'b0;
            wl_pending <= 1'b0;
            wl_cmd_cap <= 512'd0;
            wl_data_cap <= 512'd0;
        end else begin
            if (wl_valid && wl_ready)
                wl_cmd_cap <= wl_cmd;
            if (wd_valid && wd_ready) begin
                wl_data_cap[wl_beats*128 +: 128] <= wd_data;
                wl_beats <= wl_beats + 4'd1;
                if (wd_last) wl_last_seen <= 1'b1;
            end
            if (wl_last_seen && !wl_pending && !wdone_valid_r) begin
                wl_pending <= 1'b1;
                wdone_valid_r <= 1'b1;
                wdone_status_r <= `NPU_ST_OK;
            end
            if (wdone_valid_r && wdone_ready)
                wdone_valid_r <= 1'b0;
        end
    end
    initial begin wl_ready_r = 1'b1; wd_ready_r = 1'b1; end

    // ------------------------------------------------------------------
    // helpers
    // ------------------------------------------------------------------
    function [511:0] mk_cmd;
        input [7:0]  op;
        input [7:0]  sub;
        input [15:0] job;
        input [7:0]  dstb;
        input [7:0]  srcb;
        input [31:0] p0, p1, p2, p3, p4, p5, p6;
        begin
            mk_cmd = 512'd0;
            mk_cmd[7:0]     = 8'h01;
            mk_cmd[15:8]    = op;
            mk_cmd[23:16]   = sub;
            mk_cmd[47:32]   = job;
            mk_cmd[71:64]   = srcb;
            mk_cmd[87:80]   = dstb;
            mk_cmd[103:96]  = 8'h00;
            mk_cmd[159:128] = p0;
            mk_cmd[191:160] = p1;
            mk_cmd[223:192] = p2;
            mk_cmd[255:224] = p3;
            mk_cmd[287:256] = p4;
            mk_cmd[319:288] = p5;
            mk_cmd[351:320] = p6;
        end
    endfunction

    task automatic send_cmd;
        input [511:0] c;
        begin
            @(negedge clk_core);
            task_cmd   = c;
            task_valid = 1'b1;
            #1;
            while (task_ready !== 1'b1) begin
                @(negedge clk_core);
                #1;
            end
            @(posedge clk_core);
            @(negedge clk_core);
            #1;
            task_valid = 1'b0;
        end
    endtask

    task automatic expect_event;
        input [15:0]  exp_status;
        input [31:0]  exp_units;
        input [15:0]  exp_job;
        reg   [15:0]  st;
        begin
            #1;
            while (ev_valid !== 1'b1) begin
                @(negedge clk_core);
                #1;
            end
            st = ev_data[47:32];
            if (st !== exp_status) begin
                fails = fails + 1;
                $display("FAIL ev status=%0h exp=%0h", st, exp_status);
            end
            if (ev_data[95:64] !== exp_units) begin
                fails = fails + 1;
                $display("FAIL ev units=%0d exp=%0d", ev_data[95:64], exp_units);
            end
            if (ev_data[15:0] !== exp_job) begin
                fails = fails + 1;
                $display("FAIL ev job=%0d exp=%0d", ev_data[15:0], exp_job);
            end
            ev_ready = 1'b1;
            @(posedge clk_core);
            @(negedge clk_core);
            #1;
            ev_ready = 1'b0;
        end
    endtask

    task automatic chk_byte;
        input [7:0] got;
        input [7:0] exp;
        input [255:0] what;
        begin
            if (got !== exp) begin
                fails = fails + 1;
                $display("FAIL %0s: got %0h exp %0h", what, got, exp);
            end
        end
    endtask

    // ------------------------------------------------------------------
    // stimulus
    // ------------------------------------------------------------------
    reg [7:0] expb;
    reg [7:0] gotb;

    initial begin
        fails = 0;
        task_valid = 1'b0;
        task_cmd   = 512'd0;
        ev_ready   = 1'b0;
        buf_en     = 8'hFF;
        buf_bank   = 64'h0706050403020100;
        buf_base   = 256'd0;
        buf_size   = 256'd0;
        for (i = 0; i < 8; i = i + 1) begin
            buf_base[32*i +: 32] = 64 * i;
            buf_size[32*i +: 32] = 4096;
        end
        for (i = 0; i < 65536; i = i + 1)
            u_ext.mem[i] = 8'h00;

        repeat (6) @(negedge clk_core);
        rst_n = 1'b1;
        repeat (4) @(negedge clk_core);

        // ------------------------------------------------------------
        // A: 33-byte LOAD
        // ------------------------------------------------------------
        $display("-- test A: LOAD 33 bytes");
        for (i = 0; i < 40; i = i + 1)
            u_ext.mem[16'h100 + i] = 8'h10 + i[7:0];

        send_cmd(mk_cmd(`NPU_OP_LOAD, 8'd0, 16'h0001, 8'd0, 8'd0,
                        32'h0000_0100, 32'd0, 32'd0, 32'd33,
                        32'd1, 32'd0, 32'd0));
        expect_event(`NPU_ST_OK, 32'd33, 16'h0001);

        for (k = 0; k < 32; k = k + 1) begin
            gotb = u_mem.g_bank[0].u_ram.mem[0][8*k +: 8];
            chk_byte(gotb, 8'h10 + k[7:0], "A word0");
        end
        gotb = u_mem.g_bank[0].u_ram.mem[1][7:0];
        chk_byte(gotb, 8'h30, "A word1 byte0");

        // ------------------------------------------------------------
        // B: two rows, 20 bytes, strides 64 / 64
        // ------------------------------------------------------------
        $display("-- test B: LOAD two rows with strides");
        for (i = 0; i < 20; i = i + 1) begin
            u_ext.mem[16'h200 + i] = 8'h40 + i[7:0];
            u_ext.mem[16'h240 + i] = 8'h60 + i[7:0];
        end
        send_cmd(mk_cmd(`NPU_OP_LOAD, 8'd0, 16'h0002, 8'd0, 8'd0,
                        32'h0000_0200, 32'd0, 32'd0, 32'd20,
                        32'd2, 32'd64, 32'd64));
        expect_event(`NPU_ST_OK, 32'd40, 16'h0002);

        // 20 bytes fit inside one 32-byte scratchpad word, so only word 0 is
        // written for row 0; row 1 starts at local offset 64 (word 2).
        for (k = 0; k < 20; k = k + 1) begin
            gotb = u_mem.g_bank[0].u_ram.mem[0][8*k +: 8];
            chk_byte(gotb, 8'h40 + k[7:0], "B row0 word0");
        end
        for (k = 0; k < 20; k = k + 1) begin
            gotb = u_mem.g_bank[0].u_ram.mem[2][8*k +: 8];
            chk_byte(gotb, 8'h60 + k[7:0], "B row1 word2");
        end

        // ------------------------------------------------------------
        // C: STORE the same 20-byte rows to 0x300 / 0x340
        // ------------------------------------------------------------
        $display("-- test C: STORE two rows");
        send_cmd(mk_cmd(`NPU_OP_STORE, 8'd0, 16'h0003, 8'd0, 8'd0,
                        32'h0000_0300, 32'd0, 32'd0, 32'd20,
                        32'd2, 32'd64, 32'd64));
        expect_event(`NPU_ST_OK, 32'd40, 16'h0003);

        for (k = 0; k < 20; k = k + 1) begin
            gotb = u_ext.mem[16'h300 + k];
            chk_byte(gotb, 8'h40 + k[7:0], "C row0");
            gotb = u_ext.mem[16'h340 + k];
            chk_byte(gotb, 8'h60 + k[7:0], "C row1");
        end
        gotb = u_ext.mem[16'h300 + 20];
        chk_byte(gotb, 8'h00, "C row0 padding untouched");

        // ------------------------------------------------------------
        // D: LOAD_W
        // ------------------------------------------------------------
        $display("-- test D: LOAD_W");
        for (i = 0; i < 64; i = i + 1)
            u_ext.mem[16'h400 + i] = 8'hA0 + i[7:0];

        send_cmd(mk_cmd(`NPU_OP_LOAD, 8'd1, 16'h0004, 8'd0, 8'd0,
                        32'h0000_0400, 32'd0, 32'd0, 32'd0,
                        32'd64, 32'd0, 32'd0));
        expect_event(`NPU_ST_OK, 32'd64, 16'h0004);

        if (wl_cmd_cap[287:256] !== 32'd64) begin
            fails = fails + 1;
            $display("FAIL wload cmd bytes=%0d exp 64", wl_cmd_cap[287:256]);
        end
        if (wl_beats !== 4'd4) begin
            fails = fails + 1;
            $display("FAIL wload beats=%0d exp 4", wl_beats);
        end
        for (k = 0; k < 64; k = k + 1) begin
            gotb = wl_data_cap[8*k +: 8];
            chk_byte(gotb, 8'hA0 + k[7:0], "D wload data");
        end

        // ------------------------------------------------------------
        if (dma_fatal !== 1'b0) begin
            fails = fails + 1;
            $display("FAIL: fatal asserted");
        end
        repeat (10) @(negedge clk_core);
        if (u_dma.core_state !== 3'd0) begin
            fails = fails + 1;
            $display("FAIL: DMA not idle");
        end

        if (fails == 0) $display("PASS: tb_dma_engine");
        else            $display("FAIL: tb_dma_engine (%0d failures)", fails);
        $finish;
    end

    initial begin
        #2000000;
        $display("FAIL: timeout");
        $display("  core_state=%0d ld_state=%0d io_state=%0d i_row=%0d i_rows=%0d",
                 u_dma.core_state, u_dma.ld_state, u_dma.io_state,
                 u_dma.i_row, u_dma.i_rows);
        $display("  c_row=%0d c_rows=%0d c_woff=%0d c_off=%0d ld_empty=%b",
                 u_dma.c_row, u_dma.c_rows, u_dma.c_woff, u_dma.c_off,
                 u_dma.ld_empty);
        $finish;
    end

endmodule

`default_nettype wire

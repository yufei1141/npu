//=============================================================================
// tb_top_gemv : end-to-end W4A16 GEMV through the real npu_top
//
//   Drives the platform sequence (reset -> IDLE -> RUN), then the same task
//   stream the Verilator harness runs: LOAD the A row, LOAD_W the packed
//   weight frame, FENCE, MATMUL, STORE.  Unlike the Verilator harness this
//   testbench sees the whole hierarchy, so it can also peek the scratchpad,
//   the weight banks and the accumulation path when a check fails.
//
//   Stimulus (build/real/mk_tgb.py):
//     tgb_a.hex     FP16 A row, placed at ext 0x1000
//     tgb_w.hex     packed frame, placed at ext 0x4000
//     tgb_cmds.hex  five command words
//     tgb_cfg.txt   "<K> <k_steps> <n_grp> <a_row_bytes>"
//=============================================================================
`timescale 1ns/1ps
`default_nettype none

`include "npu_defs.vh"

module tb_top_gemv;

    localparam MEM_W = 256;

    reg clk_core, clk_io, reset_n;
    initial begin clk_core = 0; clk_io = 0; reset_n = 0; end
    always #5  clk_core = ~clk_core;
    always #5  clk_io   = ~clk_io;

    reg          platform_isolated;
    reg          run_req;
    reg          cmd_valid;
    wire         cmd_ready;
    reg  [511:0] cmd_data;

    wire         cpl_valid;
    reg          cpl_ready;
    wire [127:0] cpl_data;

    reg          cfg_valid, cfg_write;
    wire         cfg_ready;
    reg  [15:0]  cfg_addr;
    reg  [63:0]  cfg_wdata;
    wire         cfg_rsp_valid;
    reg          cfg_rsp_ready;
    wire [63:0]  cfg_rdata;
    wire         cfg_error;

    wire         irq;
    wire [2:0]   top_state;
    wire         fatal;
    wire [15:0]  fatal_code;

    wire         rd_req_valid, rd_req_ready;
    wire [63:0]  rd_req_addr;
    wire [31:0]  rd_req_bytes;
    wire [15:0]  rd_req_tag;
    wire         rd_valid, rd_ready;
    wire [127:0] rd_data;
    wire [15:0]  rd_keep;
    wire         rd_last;
    wire [15:0]  rd_tag;
    wire         rd_error;

    wire         wr_req_valid, wr_req_ready;
    wire [63:0]  wr_req_addr;
    wire [31:0]  wr_req_bytes;
    wire [15:0]  wr_req_tag;
    wire         wr_valid, wr_ready;
    wire [127:0] wr_data;
    wire [15:0]  wr_keep;
    wire         wr_last;
    wire [15:0]  wr_tag;
    wire         wr_rsp_valid, wr_rsp_ready;
    wire [15:0]  wr_rsp_tag;
    wire         wr_rsp_error;

    npu_top #(
        .MEM_DATA_W(MEM_W), .IO_DATA_W(128), .EXT_WR_W(128),
        .EVENT_SLOTS(16), .RESULT_SLOTS(4), .WBUF_K_STEPS(256),
        .V_LANES(4), .BANKS(8), .BANK_DEPTH(1024), .GROUPS(8)
    ) u_dut (
        .clk_core(clk_core), .clk_io(clk_io), .reset_n(reset_n),
        .platform_isolated(platform_isolated), .run_req(run_req),
        .cmd_valid(cmd_valid), .cmd_ready(cmd_ready), .cmd_data(cmd_data),
        .cpl_valid(cpl_valid), .cpl_ready(cpl_ready), .cpl_data(cpl_data),
        .cfg_valid(cfg_valid), .cfg_ready(cfg_ready), .cfg_write(cfg_write),
        .cfg_addr(cfg_addr), .cfg_wdata(cfg_wdata),
        .cfg_rsp_valid(cfg_rsp_valid), .cfg_rsp_ready(cfg_rsp_ready),
        .cfg_rdata(cfg_rdata), .cfg_error(cfg_error),
        .irq(irq), .top_state(top_state), .fatal(fatal),
        .fatal_code(fatal_code),
        .rd_req_valid(rd_req_valid), .rd_req_ready(rd_req_ready),
        .rd_req_addr(rd_req_addr), .rd_req_bytes(rd_req_bytes),
        .rd_req_tag(rd_req_tag),
        .rd_valid(rd_valid), .rd_ready(rd_ready), .rd_data(rd_data),
        .rd_keep(rd_keep), .rd_last(rd_last), .rd_tag(rd_tag),
        .rd_error(rd_error),
        .wr_req_valid(wr_req_valid), .wr_req_ready(wr_req_ready),
        .wr_req_addr(wr_req_addr), .wr_req_bytes(wr_req_bytes),
        .wr_req_tag(wr_req_tag),
        .wr_valid(wr_valid), .wr_ready(wr_ready), .wr_data(wr_data),
        .wr_keep(wr_keep), .wr_last(wr_last), .wr_tag(wr_tag),
        .wr_rsp_valid(wr_rsp_valid), .wr_rsp_ready(wr_rsp_ready),
        .wr_rsp_tag(wr_rsp_tag), .wr_rsp_error(wr_rsp_error)
    );

    extmem_model #(.MEM_BYTES(65536)) u_ext (
        .clk(clk_io), .rst_n(reset_n),
        .rd_req_valid(rd_req_valid), .rd_req_ready(rd_req_ready),
        .rd_req_addr(rd_req_addr), .rd_req_bytes(rd_req_bytes),
        .rd_req_tag(rd_req_tag),
        .rd_valid(rd_valid), .rd_ready(rd_ready), .rd_data(rd_data),
        .rd_keep(rd_keep), .rd_last(rd_last), .rd_tag(rd_tag),
        .wr_req_valid(wr_req_valid), .wr_req_ready(wr_req_ready),
        .wr_req_addr(wr_req_addr), .wr_req_bytes(wr_req_bytes),
        .wr_req_tag(wr_req_tag),
        .wr_valid(wr_valid), .wr_ready(wr_ready), .wr_data(wr_data),
        .wr_keep(wr_keep), .wr_last(wr_last), .wr_tag(wr_tag),
        .wr_rsp_valid(wr_rsp_valid), .wr_rsp_ready(wr_rsp_ready),
        .wr_rsp_tag(wr_rsp_tag), .wr_rsp_error(wr_rsp_error)
    );

    // ------------------------------------------------------------------
    // helpers
    // ------------------------------------------------------------------
    integer fails;
    integer i;
    reg [511:0] cmd_mem [0:4];

    task cfg_wr;
        input [15:0] a;
        input [63:0] d;
        input        exp_err;
        input [8*16-1:0] what;
        begin
            @(negedge clk_core);
            cfg_valid = 1'b1; cfg_write = 1'b1;
            cfg_addr  = a;    cfg_wdata = d;
            @(posedge clk_core);
            @(negedge clk_core);
            cfg_valid = 1'b0;
            while (!cfg_rsp_valid) @(negedge clk_core);
            if (cfg_error !== exp_err) begin
                fails = fails + 1;
                $display("FAIL %0s: cfg_error %b exp %b", what, cfg_error, exp_err);
            end
            cfg_rsp_ready = 1'b1;
            @(posedge clk_core);
            @(negedge clk_core);
            cfg_rsp_ready = 1'b0;
        end
    endtask

    task send_cmd;
        input [511:0] c;
        begin
            @(negedge clk_core);
            while (!cmd_ready) @(negedge clk_core);
            cmd_data  = c;
            cmd_valid = 1'b1;
            @(posedge clk_core);
            @(negedge clk_core);
            cmd_valid = 1'b0;
        end
    endtask

    reg [127:0] cpl;
    task get_cpl;
        input [8*16-1:0] what;
        begin
            while (!cpl_valid) @(negedge clk_core);
            cpl = cpl_data;
            $display("%0s: job=%0d slot=%0d op=%02x status=%04x detail=%04x units=%0d",
                     what, cpl[15:0], cpl[23:16], cpl[31:24], cpl[47:32],
                     cpl[63:48], cpl[95:64]);
            if (fatal) begin
                fails = fails + 1;
                $display("FAIL %0s: fatal code %04x", what, fatal_code);
            end
            cpl_ready = 1'b1;
            @(posedge clk_core);
            @(negedge clk_core);
            cpl_ready = 1'b0;
        end
    endtask

    task wait_state;
        input [2:0] s;
        input integer limit;
        integer n;
        begin
            n = 0;
            while (top_state !== s && n < limit) begin
                @(negedge clk_core);
                n = n + 1;
            end
            if (top_state !== s) begin
                fails = fails + 1;
                $display("FAIL: state %0d not reached (now %0d)", s, top_state);
            end
        end
    endtask

    function [31:0] ext32;
        input integer addr;
        begin
            ext32 = {u_ext.mem[addr + 3], u_ext.mem[addr + 2],
                     u_ext.mem[addr + 1], u_ext.mem[addr]};
        end
    endfunction

    // ------------------------------------------------------------------
    // stimulus load
    // ------------------------------------------------------------------
    integer fd, rc, k_len, k_steps, n_grp, a_row_bytes, frame_bytes;
    integer n_a, n_w;
    reg [7:0] tmp [0:8191];

    initial begin
        n_a = 0; n_w = 0;
        fd = $fopen("build/real/tgb_cfg.txt", "r");
        rc = $fscanf(fd, "%d %d %d %d %d", k_len, k_steps, n_grp,
                     a_row_bytes, frame_bytes);
        $fclose(fd);
        if (rc != 5) begin
            $display("RESULT: FAIL (tgb_cfg.txt)");
            $finish;
        end

        $readmemh("build/real/tgb_cmds.hex", cmd_mem);

        $readmemh("build/real/tgb_a.hex", tmp);
        for (i = 0; i < a_row_bytes; i = i + 1)
            u_ext.mem[16'h1000 + i] = tmp[i];
        n_a = a_row_bytes;

        $readmemh("build/real/tgb_w.hex", tmp);
        for (i = 0; i < frame_bytes; i = i + 1)
            u_ext.mem[16'h4000 + i] = tmp[i];
        n_w = frame_bytes;

        $display("stimulus: K=%0d steps=%0d grp=%0d a=%0d bytes frame=%0d bytes",
                 k_len, k_steps, n_grp, n_a, n_w);
    end

    // ------------------------------------------------------------------
    // sequence
    // ------------------------------------------------------------------
    initial begin
        fails = 0;
        platform_isolated = 1'b1;
        run_req    = 1'b0;
        cmd_valid  = 1'b0;
        cmd_data   = 512'd0;
        cpl_ready  = 1'b0;
        cfg_valid  = 1'b0;
        cfg_write  = 1'b0;
        cfg_addr   = 16'd0;
        cfg_wdata  = 64'd0;
        cfg_rsp_ready = 1'b0;

        repeat (6) @(posedge clk_core);
        reset_n = 1'b1;
        wait_state(3'd2, 50);

        // buffers 0 / 1 / 2 on banks 0 / 1 / 2, base 0, 8 KiB each
        cfg_wr(16'h0100, 64'h0000_0000_0000_0001, 1'b0, "map buf0");
        cfg_wr(16'h0108, 64'd0, 1'b0, "base buf0");
        cfg_wr(16'h0110, 64'h0000_0000_0000_2000, 1'b0, "size buf0");
        cfg_wr(16'h0120, 64'h0000_0000_0000_0101, 1'b0, "map buf1");
        cfg_wr(16'h0128, 64'd0, 1'b0, "base buf1");
        cfg_wr(16'h0130, 64'h0000_0000_0000_2000, 1'b0, "size buf1");
        cfg_wr(16'h0140, 64'h0000_0000_0000_0201, 1'b0, "map buf2");
        cfg_wr(16'h0148, 64'd0, 1'b0, "base buf2");
        cfg_wr(16'h0150, 64'h0000_0000_0000_2000, 1'b0, "size buf2");

        run_req = 1'b1;
        wait_state(3'd3, 20);

        for (i = 0; i < 5; i = i + 1) begin
            send_cmd(cmd_mem[i]);
            get_cpl("cmd");
        end

        // ---- results ----
        for (i = 0; i < 4; i = i + 1)
            $display("result[%0d] = %h", i, ext32(16'h8000 + 4 * i));
        for (i = 0; i < 4; i = i + 1)
            if (ext32(16'h8000 + 4 * i) !== 32'h4480_0000) begin
                fails = fails + 1;
                $display("FAIL: result[%0d] = %h exp 44800000 (1024.0)",
                         i, ext32(16'h8000 + 4 * i));
            end

        if (fails == 0) $display("RESULT: PASS");
        else            $display("RESULT: FAIL (%0d)", fails);
        $finish;
    end

    initial begin
        #2000000;
        $display("RESULT: FAIL (timeout) state=%0d", top_state);
        $finish;
    end

endmodule

`default_nettype wire

//=============================================================================
// tb_mpu_engine : MATMUL adapter around the real gemv_stream core
//
//   A. invalid MATMUL -> event with BAD_OPCODE, no result reservation
//   B. W4, M=2, g=2, tiles=2, k_steps=3: A beats are compared against the
//      scratchpad image, B beats are counted, both tiles are selected and
//      released in order, acc pulses and the event are checked
//   C. a weight bank that reports a release error must fault the engine
//
//   The weight path is a local stub: it answers select immediately, streams
//   k_steps B beats with zero weights (unit E8M0 scale) and then raises
//   tile_done.  With zero weights every accumulator output is zero, so the
//   captured acc word can be checked without reimplementing the datapath.
//=============================================================================
`timescale 1ns/1ps
`default_nettype none

`include "npu_defs.vh"

module tb_mpu_engine;

    localparam DW      = 256;
    localparam REQ_W   = `NPU_REQ_W(DW, DW/8);
    localparam RSP_W   = `NPU_RSP_W(DW);
    localparam TILES   = 2;
    localparam KSTEPS  = 3;
    localparam MLEN    = 2;
    localparam NGRP    = 2;
    localparam KVALID  = 48;           // 3 steps * 16
    localparam A_BASE  = 2;            // element aligned, not word aligned
    localparam A_STRIDE= 128;
    localparam SRCBUF  = 0;
    localparam DSTBUF  = 1;

    reg clk_core, rst_n;
    initial begin clk_core = 0; rst_n = 0; end
    always #5 clk_core = ~clk_core;

    integer fails;
    integer i, m, s, e, a;
    integer beat;

    // ------------------------------------------------------------------
    // scratchpad
    // ------------------------------------------------------------------
    wire [4:0]          c_req_valid, c_req_ready, c_rsp_ready;
    wire [5*REQ_W-1:0]  c_req;
    wire [4:0]          c_rsp_valid;
    wire [5*RSP_W-1:0]  c_rsp;
    wire                afv;
    wire [63:0]         afi;
    wire [7:0]          bbusy;

    wire         a_req_valid, a_req_ready;
    wire [REQ_W-1:0] a_req;
    wire         a_rsp_valid;
    wire [RSP_W-1:0] a_rsp;

    assign c_req_valid = {4'b0000, a_req_valid};
    assign c_req       = {{(4*REQ_W){1'b0}}, a_req};
    assign c_rsp_ready = 5'b00001;
    assign a_req_ready = c_req_ready[0];
    assign a_rsp_valid = c_rsp_valid[0];
    assign a_rsp       = c_rsp[RSP_W-1:0];

    memory_subsystem #(
        .MEM_DATA_W(DW), .BANKS(8), .BANK_DEPTH(1024), .L_RAM(1)
    ) u_mem (
        .clk_core(clk_core), .rst_core_n(rst_n),
        .buf_enable(8'h01), .buf_bank_flat(64'd0),
        .buf_base_flat(256'd0),
        .buf_size_flat({224'd0, 32'd4096}),
        .c_req_valid(c_req_valid), .c_req_ready(c_req_ready),
        .c_req(c_req), .c_rsp_valid(c_rsp_valid), .c_rsp_ready(c_rsp_ready),
        .c_rsp(c_rsp),
        .outstanding_count(), .access_fault_valid(afv),
        .access_fault_info(afi), .bank_busy(bbusy)
    );

    // ------------------------------------------------------------------
    // DUT
    // ------------------------------------------------------------------
    reg         task_valid;
    wire        task_ready;
    reg  [511:0] task_cmd;
    wire        ev_valid;
    wire [127:0] ev_data;
    reg         ev_ready;

    wire        req_ready;
    wire        rsp_valid, rsp_ready;

    wire        wsel_valid, wsel_ready;
    wire [15:0] wsel_job, wsel_tile, wsel_s;
    wire [1:0]  wsel_mode;
    wire [3:0]  wsel_g;
    reg         wb_valid;
    wire        wb_ready;
    wire [2303:0] wb_data;
    reg         td_valid;
    wire        td_ready;
    reg [15:0]  td_status;

    wire        wbjob_valid, wbjob_ready;
    wire [511:0] wbjob_cmd;
    reg         wbdone_valid;
    wire        wbdone_ready;
    wire [15:0] wbdone_status;
    wire [31:0] wbdone_tiles;

    wire        acc_valid;
    wire [4095:0] acc_data;
    wire        fatal;

    mpu_engine #(
        .MEM_DATA_W(DW), .RESULT_SLOTS(4), .WBUF_K_STEPS(256)
    ) u_dut (
        .clk_core(clk_core), .rst_core_n(rst_n),
        .task_valid(task_valid), .task_ready(task_ready),
        .task_cmd(task_cmd),
        .event_valid(ev_valid), .event_ready(ev_ready), .event_data(ev_data),
        .mem_req_valid(a_req_valid), .mem_req_ready(a_req_ready),
        .mem_req(a_req),
        .mem_rsp_valid(a_rsp_valid), .mem_rsp_ready(rsp_ready),
        .mem_rsp(a_rsp),
        .w_select_valid(wsel_valid), .w_select_ready(wsel_ready),
        .w_select_job(wsel_job), .w_select_tile(wsel_tile),
        .w_select_mode(wsel_mode), .w_select_g(wsel_g), .w_select_s(wsel_s),
        .w_b_valid(wb_valid), .w_b_ready(wb_ready), .w_b_data(wb_data),
        .w_tile_done_valid(td_valid), .w_tile_done_ready(td_ready),
        .w_tile_done_status(td_status),
        .wb_job_valid(wbjob_valid), .wb_job_ready(wbjob_ready),
        .wb_job_cmd(wbjob_cmd),
        .wb_done_valid(wbdone_valid), .wb_done_ready(wbdone_ready),
        .wb_done_status(wbdone_status), .wb_done_tiles(wbdone_tiles),
        .acc_valid(acc_valid), .acc_data(acc_data),
        .fatal(fatal)
    );

    // mem_req_ready is driven by the memory subsystem; use the real one
    // (client 0 ready comes back through c_req_ready[0]).
    wire mem_ready0 = c_req_ready[0];

    // ------------------------------------------------------------------
    // weight + writeback stubs
    // ------------------------------------------------------------------
    localparam W_IDLE = 2'd0;
    localparam W_STRM = 2'd1;
    localparam W_REL  = 2'd2;

    reg [1:0]  wst;
    reg [15:0] wst_tile;
    reg [15:0] wst_cnt;
    reg [15:0] sel_count;
    reg [15:0] b_count;
    reg [15:0] b_beats;
    reg [15:0] td_status_cfg;

    assign wsel_ready = (wst == W_IDLE);

    always @(posedge clk_core or negedge rst_n) begin
        if (!rst_n) begin
            wst       <= W_IDLE;
            wst_tile  <= 16'd0;
            wst_cnt   <= 16'd0;
            wb_valid  <= 1'b0;
            td_valid  <= 1'b0;
            td_status <= `NPU_ST_OK;
            sel_count <= 16'd0;
            b_count   <= 16'd0;
        end else begin
            case (wst)
            W_IDLE: begin
                wb_valid <= 1'b0;
                if (wsel_valid && wsel_ready) begin
                    wst_tile  <= wsel_tile;
                    wst_cnt   <= 16'd0;
                    sel_count <= sel_count + 16'd1;
                    wst       <= W_STRM;
                end
            end
            W_STRM: begin
                wb_valid <= 1'b1;
                if (wb_valid && wb_ready) begin
                    b_count <= b_count + 16'd1;
                    if (wst_cnt + 16'd1 == KSTEPS) begin
                        wb_valid <= 1'b0;
                        wst      <= W_REL;
                    end else begin
                        wst_cnt <= wst_cnt + 16'd1;
                    end
                end
            end
            W_REL: begin
                td_valid  <= 1'b1;
                td_status <= td_status_cfg;
                if (td_valid && td_ready) begin
                    td_valid <= 1'b0;
                    wst      <= W_IDLE;
                end
            end
            default: wst <= W_IDLE;
            endcase
        end
    end

    // zero weights, unit scale in every group
    genvar gj;
    generate
        for (gj = 0; gj < 8; gj = gj + 1) begin : g_wb
            assign wb_data[256*gj +: 256]        = 256'd0;
            assign wb_data[2048 + 32*gj +: 32]   = `NPU_E8M0_ONE;
        end
    endgenerate

    reg        wbjob_seen;
    reg [511:0] wbjob_last;

    assign wbjob_ready = 1'b1;

    always @(posedge clk_core or negedge rst_n) begin
        if (!rst_n) begin
            wbjob_seen    <= 1'b0;
            wbjob_last    <= 512'd0;
            wbdone_valid  <= 1'b0;
        end else begin
            if (wbjob_valid && wbjob_ready) begin
                wbjob_seen <= 1'b1;
                wbjob_last <= wbjob_cmd;
            end
            if (u_dut.state == 4'd6)      // M_WRITEBACK
                wbdone_valid <= 1'b1;
            if (wbdone_valid && wbdone_ready)
                wbdone_valid <= 1'b0;
        end
    end

    assign wbdone_status = `NPU_ST_OK;
    assign wbdone_tiles  = TILES;

    // ------------------------------------------------------------------
    // monitors
    // ------------------------------------------------------------------
    integer a_seen, acc_seen, ev_seen;
    reg [127:0] ev0, ev1, ev2;
    reg [511:0] cmd_good;

    always @(posedge clk_core or negedge rst_n) begin
        if (!rst_n) begin
            a_seen   <= 0;
            acc_seen <= 0;
            ev_seen  <= 0;
            ev0      <= 128'd0;
            ev1      <= 128'd0;
            ev2      <= 128'd0;
        end else begin
            if (ev_valid && ev_ready) begin
                if (ev_seen == 0) ev0 <= ev_data;
                if (ev_seen == 1) ev1 <= ev_data;
                if (ev_seen == 2) ev2 <= ev_data;
                if (ev_seen < 8) ev_seen <= ev_seen + 1;
            end
            if (acc_valid) begin
                acc_seen <= acc_seen + 1;
                if (acc_data !== 4096'd0) begin
                    fails = fails + 1;
                    $display("FAIL: acc_data not zero at pulse %0d", acc_seen);
                end
            end
        end
    end

    // A beat scoreboard, checked against the scratchpad image
    reg [1023:0] a_exp;
    reg [15:0]   a_val;
    integer am, ai, aw, asc;
    always @(posedge clk_core) begin
        if (a_seen < 64 && u_dut.a_wr) begin
            asc = a_seen % KSTEPS;      // step index; both tiles reuse A
            a_exp = 1024'd0;
            for (am = 0; am < 4; am = am + 1) begin
                for (ai = 0; ai < 16; ai = ai + 1) begin
                    if (am < MLEN) begin
                        a_val = 16'h3C00 + ((am*KVALID + asc*16 + ai) & 16'h0FFF);
                        a_exp[256*am + 16*ai +: 16] = a_val;
                    end
                end
            end
            if (u_dut.a_data_p !== a_exp) begin
                fails = fails + 1;
                $display("FAIL: A beat %0d (step %0d) mismatch", a_seen, asc);
            end
            a_seen = a_seen + 1;
        end
    end

    // ------------------------------------------------------------------
    // helpers
    // ------------------------------------------------------------------
    function [511:0] mk_matmul;
        input [7:0]  opcode;
        input [15:0] job;
        input [7:0]  slot;
        input [2:0]  mlen;
        input [3:0]  ngrp;
        input [1:0]  mode;
        input [15:0] tiles;
        input [15:0] ksteps;
        input [31:0] kvalid;
        input [31:0] nvalid;
        begin
            mk_matmul = 512'd0;
            mk_matmul[7:0]   = 8'h01;
            mk_matmul[15:8]  = opcode;
            mk_matmul[23:16] = 8'd0;
            mk_matmul[47:32] = job;
            mk_matmul[71:64] = SRCBUF;
            mk_matmul[87:80] = DSTBUF;
            mk_matmul[95:88] = 8'h08;          // dst fp16
            mk_matmul[103:96]= slot;
            mk_matmul[130:128] = mlen;
            mk_matmul[139:136] = ngrp;
            mk_matmul[145:144] = mode;
            mk_matmul[175:160] = tiles;
            mk_matmul[191:176] = ksteps;
            mk_matmul[223:192] = kvalid;
            mk_matmul[255:224] = nvalid;
            mk_matmul[287:256] = A_BASE;
            mk_matmul[319:288] = A_STRIDE;
            mk_matmul[351:320] = 32'd0;
            mk_matmul[383:352] = 32'd0;
            mk_matmul[415:384] = 32'd64;
        end
    endfunction

    task automatic send_task;
        input [511:0] cmd;
        begin
            @(negedge clk_core);
            task_cmd   = cmd;
            task_valid = 1'b1;
            @(posedge clk_core);        // adapter samples the task here
            #1;
            @(negedge clk_core);
            task_valid = 1'b0;
        end
    endtask

    task automatic wait_event;
        input integer idx;
        input [15:0] exp_status;
        input [15:0] exp_units;
        input [15:0] exp_job;
        input [7:0]  exp_slot;
        input [7:0]  exp_opcode;
        integer t;
        reg [127:0] ev;
        begin
            t = 0;
            while (ev_seen <= idx && t < 5000) begin
                @(negedge clk_core);
                t = t + 1;
            end
            if (ev_seen <= idx) begin
                fails = fails + 1;
                $display("FAIL: event %0d never arrived", idx);
            end else begin
                ev = (idx == 0) ? ev0 : (idx == 1) ? ev1 : ev2;
                if (`NPU_EV_STATUS(ev) !== exp_status) begin
                    fails = fails + 1;
                    $display("FAIL: event %0d status %0h exp %0h",
                             idx, `NPU_EV_STATUS(ev), exp_status);
                end
                if (`NPU_EV_UNITS(ev) !== exp_units) begin
                    fails = fails + 1;
                    $display("FAIL: event %0d units %0h exp %0h",
                             idx, `NPU_EV_UNITS(ev), exp_units);
                end
                if (`NPU_EV_JOB(ev) !== exp_job || `NPU_EV_SLOT(ev) !== exp_slot
                    || `NPU_EV_OPCODE(ev) !== exp_opcode) begin
                    fails = fails + 1;
                    $display("FAIL: event %0d id %0h/%0h/%0h",
                             idx, `NPU_EV_JOB(ev), `NPU_EV_SLOT(ev),
                             `NPU_EV_OPCODE(ev));
                end
            end
        end
    endtask

    task wait_idle;
        integer t;
        begin
            t = 0;
            while (u_dut.state !== 4'd0 && t < 5000) begin
                @(negedge clk_core);
                t = t + 1;
            end
            if (t >= 5000) begin
                fails = fails + 1;
                $display("FAIL: engine did not return to IDLE (state %0d)",
                         u_dut.state);
            end
        end
    endtask

    // ------------------------------------------------------------------
    // memory image
    // ------------------------------------------------------------------
    task set_byte;
        input integer addr;
        input [7:0]  v;
        begin
            u_mem.g_bank[0].u_ram.mem[addr >> 5][8*(addr & 31) +: 8] = v;
        end
    endtask

    // ------------------------------------------------------------------
    // main
    // ------------------------------------------------------------------
    initial begin
        fails      = 0;
        task_valid = 0;
        task_cmd   = 0;
        ev_ready   = 1;
        td_status_cfg = `NPU_ST_OK;

        // zero the A window, then load the activation image
        for (i = 0; i < 512; i = i + 1) set_byte(i, 8'h00);
        for (m = 0; m < MLEN; m = m + 1)
            for (s = 0; s < KSTEPS; s = s + 1)
                for (e = 0; e < 16; e = e + 1) begin
                    a = A_BASE + A_STRIDE*m + 2*(16*s + e);
                    set_byte(a,     (16'h3C00 + ((m*KVALID + s*16 + e) & 16'h0FFF)));
                    set_byte(a + 1, (16'h3C00 + ((m*KVALID + s*16 + e) & 16'h0FFF)) >> 8);
                end

        repeat (5) @(negedge clk_core);
        rst_n = 1'b1;
        repeat (3) @(negedge clk_core);

        // ---------------- A: invalid opcode ----------------
        $display("-- test A: invalid opcode");
        send_task(mk_matmul(8'h02, 16'h0007, 8'h01, MLEN, NGRP, 2'b00,
                            TILES, KSTEPS, KVALID, TILES*4*NGRP));
        wait_event(0, `NPU_ST_BAD_OPCODE, 16'd0, 16'h0007, 8'h01, 8'h02);
        if (wbjob_seen) begin
            fails = fails + 1;
            $display("FAIL: invalid task reserved a result slot");
        end

        // ---------------- B: valid W4 job ----------------
        $display("-- test B: W4 m=2 g=2 tiles=2 k_steps=3");
        a_seen = 0;
        cmd_good = mk_matmul(`NPU_OP_MATMUL, 16'h0011, 8'h02, MLEN, NGRP,
                             2'b00, TILES, KSTEPS, KVALID, TILES*4*NGRP);
        send_task(cmd_good);
        wait_event(1, `NPU_ST_OK, TILES, 16'h0011, 8'h02, `NPU_OP_MATMUL);
        wait_idle;
        $display("  selects=%0d B beats=%0d A beats=%0d acc pulses=%0d",
                 sel_count, b_count, a_seen, acc_seen);
        if (sel_count != TILES) begin
            fails = fails + 1;
            $display("FAIL: %0d selects, expected %0d", sel_count, TILES);
        end
        if (b_count != TILES*KSTEPS) begin
            fails = fails + 1;
            $display("FAIL: %0d B beats, expected %0d", b_count,
                     TILES*KSTEPS);
        end
        if (a_seen != TILES*KSTEPS) begin
            fails = fails + 1;
            $display("FAIL: %0d A beats, expected %0d", a_seen,
                     TILES*KSTEPS);
        end
        if (acc_seen != TILES) begin
            fails = fails + 1;
            $display("FAIL: %0d acc pulses, expected %0d", acc_seen, TILES);
        end
        if (!wbjob_seen || wbjob_last !== cmd_good) begin
            fails = fails + 1;
            $display("FAIL: wb job command mismatch");
        end
        if (fatal) begin
            fails = fails + 1;
            $display("FAIL: fatal asserted on a good job");
        end

        // ---------------- C: weight release error ----------------
        $display("-- test C: tile_done reports a release error");
        td_status_cfg = `NPU_ST_MEM_ACCESS;
        send_task(mk_matmul(`NPU_OP_MATMUL, 16'h0012, 8'h03, MLEN, NGRP, 2'b00,
                            TILES, KSTEPS, KVALID, TILES*4*NGRP));
        wait_event(2, `NPU_ST_MEM_ACCESS, TILES, 16'h0012, 8'h03,
                   `NPU_OP_MATMUL);
        repeat (5) @(negedge clk_core);
        if (!fatal) begin
            fails = fails + 1;
            $display("FAIL: fatal not asserted after a release error");
        end

        if (fails == 0) $display("PASS: tb_mpu_engine");
        else            $display("FAIL: tb_mpu_engine (%0d failures)", fails);
        $finish;
    end

    initial begin
        #40000;
        $display("FAIL: timeout");
        $display("  state=%0d a_seen=%0d sel=%0d b=%0d acc=%0d ev_seen=%0d",
                 u_dut.state, a_seen, sel_count, b_count, acc_seen, ev_seen);
        $finish;
    end

endmodule

`default_nettype wire

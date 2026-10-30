//=============================================================================
// tb_npu_top : integration smoke test of the peripheral
//
//   Two independent clocks (10 ns core, 7 ns io).  The sequence is the one the
//   platform performs: hold reset, release, wait for BOOT -> IDLE, write the
//   buffer map over the configuration port, switch to RUN, submit a VECTOR
//   task, read its completion record, stop and confirm DRAIN returns to IDLE.
//
//   Also covered: configuration is refused while RUN, an unknown opcode is
//   rejected through the same completion path, an SFU EXP2 task runs through
//   the real FP32 core (sfu_top), and the read-only registers report the map
//   and the state.
//=============================================================================
`timescale 1ns/1ps
`default_nettype none

`include "npu_defs.vh"

module tb_npu_top;

    localparam MEM_W = 256;
    localparam F_FP32 = 2'd2;

    reg clk_core, clk_io, reset_n;
    initial begin clk_core = 0; clk_io = 0; reset_n = 0; end
    always #5  clk_core = ~clk_core;
    always #7  clk_io   = ~clk_io;

    // ------------------------------------------------------------------
    // DUT
    // ------------------------------------------------------------------
    reg         platform_isolated;
    reg         run_req;
    reg         cmd_valid;
    wire        cmd_ready;
    reg  [511:0] cmd_data;

    wire        cpl_valid;
    reg         cpl_ready;
    wire [127:0] cpl_data;

    reg         cfg_valid, cfg_write;
    wire        cfg_ready;
    reg  [15:0] cfg_addr;
    reg  [63:0] cfg_wdata;
    wire        cfg_rsp_valid;
    reg         cfg_rsp_ready;
    wire [63:0] cfg_rdata;
    wire        cfg_error;

    wire        irq;
    wire [2:0]  top_state;
    wire        fatal;
    wire [15:0] fatal_code;

    wire        rd_req_valid, rd_req_ready;
    wire [63:0] rd_req_addr;
    wire [31:0] rd_req_bytes;
    wire [15:0] rd_req_tag;

    wire        wr_req_valid, wr_req_ready;
    wire [63:0] wr_req_addr;
    wire [31:0] wr_req_bytes;
    wire [15:0] wr_req_tag;
    wire        wr_valid, wr_ready;

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
        .rd_req_valid(rd_req_valid), .rd_req_ready(1'b1),
        .rd_req_addr(rd_req_addr), .rd_req_bytes(rd_req_bytes),
        .rd_req_tag(rd_req_tag),
        .rd_valid(1'b0), .rd_ready(), .rd_data({128{1'b0}}),
        .rd_keep(16'd0), .rd_last(1'b0), .rd_tag(16'd0), .rd_error(1'b0),
        .wr_req_valid(wr_req_valid), .wr_req_ready(1'b1),
        .wr_req_addr(wr_req_addr), .wr_req_bytes(wr_req_bytes),
        .wr_req_tag(wr_req_tag),
        .wr_valid(wr_valid), .wr_ready(1'b1), .wr_data(),
        .wr_keep(), .wr_last(), .wr_tag(),
        .wr_rsp_valid(1'b0), .wr_rsp_ready(), .wr_rsp_tag(16'd0),
        .wr_rsp_error(1'b0)
    );

    // ------------------------------------------------------------------
    // helpers
    // ------------------------------------------------------------------
    integer fails;
    integer i;

    // scratchpad probe: bank index must be constant in the hierarchy, so the
    // three banks the sequence uses are named explicitly
    task poke32;
        input integer bank;
        input integer addr;
        input [31:0] v;
        begin
            case (bank)
            0: u_dut.u_mem.g_bank[0].u_ram.mem[addr >> 5][8*(addr & 31) +: 32] = v;
            2: u_dut.u_mem.g_bank[2].u_ram.mem[addr >> 5][8*(addr & 31) +: 32] = v;
            default:
               u_dut.u_mem.g_bank[3].u_ram.mem[addr >> 5][8*(addr & 31) +: 32] = v;
            endcase
        end
    endtask

    function [31:0] peek32;
        input integer bank;
        input integer addr;
        reg [255:0] w;
        begin
            case (bank)
            0: w = u_dut.u_mem.g_bank[0].u_ram.mem[addr >> 5];
            2: w = u_dut.u_mem.g_bank[2].u_ram.mem[addr >> 5];
            default: w = u_dut.u_mem.g_bank[3].u_ram.mem[addr >> 5];
            endcase
            peek32 = w[8*(addr & 31) +: 32];
        end
    endfunction

    task chk32;
        input integer bank;
        input integer addr;
        input [31:0] exp;
        input [8*16-1:0] what;
        reg [31:0] got;
        begin
            got = peek32(bank, addr);
            if (got !== exp) begin
                fails = fails + 1;
                $display("FAIL %0s @%0d: %h exp %h", what, addr, got, exp);
            end
        end
    endtask

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

    task cfg_rd;
        input [15:0] a;
        input [63:0] exp;
        input [8*16-1:0] what;
        begin
            @(negedge clk_core);
            cfg_valid = 1'b1; cfg_write = 1'b0;
            cfg_addr  = a;    cfg_wdata = 64'd0;
            @(posedge clk_core);
            @(negedge clk_core);
            cfg_valid = 1'b0;
            while (!cfg_rsp_valid) @(negedge clk_core);
            if (cfg_error !== 1'b0) begin
                fails = fails + 1;
                $display("FAIL %0s: read error", what);
            end
            if (cfg_rdata !== exp) begin
                fails = fails + 1;
                $display("FAIL %0s: %h exp %h", what, cfg_rdata, exp);
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

    // ------------------------------------------------------------------
    // command builder for the smoke vectors
    // ------------------------------------------------------------------
    reg [511:0] c;

    task build_vec;
        input [7:0] op, sub, count;
        input [31:0] off0, off1, offd;
        begin
            c = 512'd0;
            c[7:0]     = 8'h01;
            c[15:8]    = op;
            c[23:16]   = sub;
            c[47:32]   = 16'd7;          // job
            c[71:64]   = 8'd2;           // src0 buffer (bank 2)
            c[79:72]   = 8'd3;           // src1 buffer (bank 3)
            c[87:80]   = 8'd0;           // dst buffer  (bank 0)
            c[89:88]   = F_FP32;
            c[91:90]   = F_FP32;
            c[93:92]   = F_FP32;
            c[95:94]   = F_FP32;
            c[103:96]  = 8'd1;           // event slot
            c[159:128] = {24'd0, count};
            c[191:160] = off0;
            c[223:192] = off1;
            c[255:224] = offd;
        end
    endtask

    // EVENT_RELEASE: free a completed slot so it can be used again
    task build_release;
        input [7:0] tslot;
        input [15:0] tjob;
        begin
            c = 512'd0;
            c[7:0]     = 8'h01;
            c[15:8]    = `NPU_OP_RELEASE;
            c[23:16]   = 8'd0;
            c[47:32]   = 16'd10;         // the release's own job id
            c[103:96]  = 8'hFF;          // EVENT_RELEASE allocates no slot
            c[159:128] = {8'd0, tjob, tslot};
        end
    endtask

    // LAYOUT command: TRANSPOSE2D, FP16, rows x cols = P0 x P1
    task build_layout;
        input [7:0] slot, sub, fs, b0, b2;
        input [31:0] p0, p1, p2, p3, p4, p5;
        begin
            c = 512'd0;
            c[7:0]     = 8'h01;
            c[15:8]    = `NPU_OP_LAYOUT;
            c[23:16]   = sub;
            c[47:32]   = 16'd9;          // job
            c[71:64]   = b0;
            c[79:72]   = 8'd0;           // src1 buffer must be zero
            c[87:80]   = b2;
            c[89:88]   = fs;
            c[91:90]   = 2'd0;
            c[93:92]   = fs;
            c[95:94]   = 2'd0;           // accumulator format must be zero
            c[103:96]  = slot;           // event slot
            c[159:128] = p0;
            c[191:160] = p1;
            c[223:192] = p2;
            c[255:224] = p3;
            c[287:256] = p4;
            c[319:288] = p5;
        end
    endtask

    // ------------------------------------------------------------------
    // sequence
    // ------------------------------------------------------------------
    localparam F1_0 = 32'h3F80_0000;
    localparam F2_0 = 32'h4000_0000;

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

        wait_state(3'd2, 50);                    // BOOT -> IDLE
        if (fatal !== 1'b0) begin
            fails = fails + 1;
            $display("FAIL: fatal in IDLE");
        end

        // ---- VERSION and default STATUS ----
        cfg_rd(16'h0000, 64'h0000_0000_0001_0000, "version");
        cfg_rd(16'h0008, {58'd0, 1'b1, 1'b1, 1'b0, 3'd2}, "status idle");

        // ---- buffers 0 / 2 / 3 on banks 0 / 2 / 3, base 0, 8 KiB each ----
        cfg_wr(16'h0100, 64'h0000_0000_0000_0001, 1'b0, "map buf0");
        cfg_wr(16'h0108, 64'd0, 1'b0, "base buf0");
        cfg_wr(16'h0110, 64'h0000_0000_0000_2000, 1'b0, "size buf0");
        cfg_wr(16'h0140, 64'h0000_0000_0000_0201, 1'b0, "map buf2");
        cfg_wr(16'h0148, 64'd0, 1'b0, "base buf2");
        cfg_wr(16'h0150, 64'h0000_0000_0000_2000, 1'b0, "size buf2");
        cfg_wr(16'h0160, 64'h0000_0000_0000_0301, 1'b0, "map buf3");
        cfg_wr(16'h0168, 64'd0, 1'b0, "base buf3");
        cfg_wr(16'h0170, 64'h0000_0000_0000_2000, 1'b0, "size buf3");
        cfg_rd(16'h0160, 64'h0000_0000_0000_0301, "map readback");
        cfg_rd(16'h0140, 64'h0000_0000_0000_0201, "map readback 2");
        cfg_rd(16'h0170, 64'h0000_0000_0000_2000, "size readback");

        // a map register with a reserved bit set is refused
        cfg_wr(16'h0160, 64'h0000_0000_0001_0301, 1'b1, "map dirty");
        // read-only word
        cfg_wr(16'h0000, 64'd0, 1'b1, "ro write");

        // ---- RUN ----
        run_req = 1'b1;
        wait_state(3'd3, 20);

        // configuration is closed in RUN
        cfg_wr(16'h0168, 64'd4, 1'b1, "cfg in run");

        // ---- VECTOR ADD, count 2 ----
        poke32(2, 32'h100, F1_0);
        poke32(2, 32'h104, F2_0);
        poke32(3, 32'h120, F1_0);
        poke32(3, 32'h124, F2_0);
        build_vec(`NPU_OP_VECTOR, 8'd0, 8'd2, 32'h100, 32'h120, 32'h140);
        send_cmd(c);
        get_cpl("vector add");
        if (cpl[47:32] !== `NPU_ST_OK) begin
            fails = fails + 1;
            $display("FAIL add: status %h detail %h", cpl[47:32], cpl[63:48]);
        end
        if (cpl[95:64] !== 32'd2) begin
            fails = fails + 1;
            $display("FAIL add: units %0d", cpl[95:64]);
        end
        if (cpl[15:0] !== 16'd7 || cpl[23:16] !== 8'd1 ||
            cpl[31:24] !== `NPU_OP_VECTOR) begin
            fails = fails + 1;
            $display("FAIL add: identity job=%0d slot=%0d op=%h",
                     cpl[15:0], cpl[23:16], cpl[31:24]);
        end
        chk32(0, 32'h140, F2_0, "add[0]");
        chk32(0, 32'h144, 32'h4080_0000, "add[1]");   // 4.0

        // ---- LAYOUT (TRANSPOSE2D) shares the VPU client ----
        //     2x2 FP16 in buffer 2 -> buffer 0, stride 8 bytes
        poke32(2, 32'h180, 32'h1002_1001);
        poke32(2, 32'h188, 32'h1004_1003);
        // slot 1 is occupied by the VECTOR job, so release it first
        build_release(8'd1, 16'd7);
        send_cmd(c);
        get_cpl("release");
        if (cpl[47:32] !== `NPU_ST_OK) begin
            fails = fails + 1;
            $display("FAIL release: status %h", cpl[47:32]);
        end

        // LAYOUT (TRANSPOSE2D) reuses the freed slot
        build_layout(8'd1, 8'd0, 2'd0, 8'd2, 8'd0,
                     32'd2, 32'd2, 32'h180, 32'h1C0, 32'h008, 32'h008);
        send_cmd(c);
        get_cpl("layout");
        if (cpl[47:32] !== `NPU_ST_OK) begin
            fails = fails + 1;
            $display("FAIL layout: status %h detail %h", cpl[47:32], cpl[63:48]);
        end
        if (cpl[95:64] !== 32'd4) begin
            fails = fails + 1;
            $display("FAIL layout: units %0d", cpl[95:64]);
        end
        chk32(0, 32'h1C0, 32'h1003_1001, "layout[0]");
        chk32(0, 32'h1C8, 32'h1004_1002, "layout[1]");

        // ---- the VPU datapath is usable again after a LAYOUT task ----
        build_vec(`NPU_OP_VECTOR, 8'd0, 8'd2, 32'h100, 32'h120, 32'h140);
        c[103:96] = 8'd2;            // a fresh event slot
        c[47:32]  = 16'd8;
        send_cmd(c);
        get_cpl("vector after layout");
        if (cpl[47:32] !== `NPU_ST_OK) begin
            fails = fails + 1;
            $display("FAIL add2: status %h", cpl[47:32]);
        end
        chk32(0, 32'h140, F2_0, "add2[0]");
        chk32(0, 32'h144, 32'h4080_0000, "add2[1]");

        // ---- SFU : EXP2 through the real FP32 core ----
        // 2^0, 2^1, 2^2, 2^-2 in buffer 2 -> buffer 0
        poke32(2, 32'h200, 32'h0000_0000);
        poke32(2, 32'h204, 32'h3F80_0000);
        poke32(2, 32'h208, 32'h4000_0000);
        poke32(2, 32'h20C, 32'hC000_0000);
        // build_layout gives the SFU the field layout it wants: src1 buffer 0,
        // accumulator bits 0, dst format = src format
        build_layout(8'd3, 8'd0, F_FP32, 8'd2, 8'd0,
                     32'd4, 32'h200, 32'd0, 32'h240, 32'd0, 32'd0);
        c[15:8]  = `NPU_OP_SFU;
        c[47:32] = 16'd11;           // a fresh job id
        send_cmd(c);
        get_cpl("sfu exp2");
        if (cpl[47:32] !== `NPU_ST_OK) begin
            fails = fails + 1;
            $display("FAIL sfu: status %h detail %h", cpl[47:32], cpl[63:48]);
        end
        if (cpl[95:64] !== 32'd4) begin
            fails = fails + 1;
            $display("FAIL sfu: units %0d", cpl[95:64]);
        end
        chk32(0, 32'h240, 32'h3F80_0000, "sfu exp2[0]");
        chk32(0, 32'h244, 32'h4000_0000, "sfu exp2[1]");
        chk32(0, 32'h248, 32'h4080_0000, "sfu exp2[2]");
        chk32(0, 32'h24C, 32'h3E80_0000, "sfu exp2[3]");   // 2^-2 = 0.25

        // ---- unknown opcode is rejected through the same path ----
        build_vec(8'h55, 8'd0, 8'd2, 32'h100, 32'h120, 32'h140);
        send_cmd(c);
        get_cpl("bad opcode");
        if (cpl[47:32] !== `NPU_ST_BAD_OPCODE) begin
            fails = fails + 1;
            $display("FAIL badop: status %h", cpl[47:32]);
        end
        if (fatal !== 1'b0) begin
            fails = fails + 1;
            $display("FAIL: fatal raised by a static rejection");
        end

        // ---- stop and drain ----
        run_req = 1'b0;
        wait_state(3'd4, 20);
        wait_state(3'd2, 400);

        cfg_rd(16'h0008, {58'd0, 1'b1, 1'b1, 1'b0, 3'd2}, "status drained");

        if (fatal !== 1'b0) begin
            fails = fails + 1;
            $display("FAIL: fatal at end (%h)", fatal_code);
        end
        if (irq !== 1'b0) begin
            fails = fails + 1;
            $display("FAIL: irq stuck high");
        end

        if (fails == 0) $display("PASS: tb_npu_top");
        else            $display("FAIL: tb_npu_top (%0d)", fails);
        $finish;
    end

    initial begin
        #400000;
        $display("FAIL: tb_npu_top (timeout)");
        $finish;
    end

endmodule

`default_nettype wire

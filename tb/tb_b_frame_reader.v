//=============================================================================
// tb_b_frame_reader : the MPU's activation-B source against a memory stub
//
//   The reader turns a PACK_W16 frame in the scratchpad into the same 2048-bit
//   beats the weight reader produces, so nothing downstream changes.  What has
//   to be right here is the address sequence and the padding:
//
//     group j, step s  ->  base + j*stride + 32*s,  tile t owns groups
//                          t*n_grp .. t*n_grp+n_grp-1 of one flat frame
//
//   Cases:
//     n_grp=3, k_steps=4, tiles=2   (a tile boundary, and 5 of 8 slots unused)
//     n_grp=8, k_steps=1, tiles=1   (every slot used, single beat)
//   The stub answers every read with an address pattern, so a slot fed from the
//   wrong group, a stale slot, a misplaced step or a wrong tile offset all show
//   up word by word.
//=============================================================================
`timescale 1ns/1ps
`default_nettype none

`include "npu_defs.vh"

module tb_b_frame_reader;

    localparam DW    = 256;
    localparam B     = 32;
    localparam REQ_W = `NPU_REQ_W(DW, B);
    localparam RSP_W = `NPU_RSP_W(DW);

    reg clk, rst_n;
    initial begin clk = 0; rst_n = 0; end
    always #5 clk = ~clk;

    // ------------------------------------------------------------------
    // DUT
    // ------------------------------------------------------------------
    reg          cfg_valid;
    wire         cfg_ready;
    reg  [511:0] cfg_cmd;

    wire         rq_valid, rq_ready;
    wire [REQ_W-1:0] rq;
    wire         rs_valid, rs_ready;
    wire [RSP_W-1:0] rs;

    wire         b_valid;
    reg          b_ready;
    wire [2047:0] b_data;
    wire         b_done;
    wire [15:0]  error;

    b_frame_reader #(.MEM_DATA_W(DW)) dut (
        .clk_core(clk), .rst_core_n(rst_n),
        .cfg_valid(cfg_valid), .cfg_ready(cfg_ready), .cfg_cmd(cfg_cmd),
        .mem_req_valid(rq_valid), .mem_req_ready(rq_ready), .mem_req(rq),
        .mem_rsp_valid(rs_valid), .mem_rsp_ready(rs_ready), .mem_rsp(rs),
        .b_valid(b_valid), .b_ready(b_ready), .b_data(b_data),
        .b_done(b_done), .error(error)
    );

    // ------------------------------------------------------------------
    // memory stub: one request in flight, two-cycle response, every word
    // returns its own byte address
    // ------------------------------------------------------------------
    reg          st_busy;
    reg [31:0]   st_addr;
    reg [2:0]    st_dly;
    reg          st_rsp_valid;
    reg [DW-1:0] st_data;
    integer      sw;

    assign rq_ready = ~st_busy;
    assign rs_valid = st_rsp_valid;
    assign rs = {`NPU_MEMR_OK, 1'b0, 8'h00, st_data};

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            st_busy      <= 1'b0;
            st_addr      <= 32'd0;
            st_dly       <= 3'd0;
            st_rsp_valid <= 1'b0;
            st_data      <= {DW{1'b0}};
        end else begin
            if (st_rsp_valid && rs_ready) st_rsp_valid <= 1'b0;

            if (rq_valid && rq_ready && !st_busy) begin
                st_busy <= 1'b1;
                st_addr <= rq[40:9];
                st_dly  <= 3'd2;
            end else if (st_busy) begin
                if (st_dly != 3'd0) begin
                    st_dly <= st_dly - 3'd1;
                end else begin
                    st_busy      <= 1'b0;
                    st_rsp_valid <= 1'b1;
                    for (sw = 0; sw < DW/32; sw = sw + 1)
                        st_data[32*sw +: 32] <= st_addr + 4*sw;
                end
            end
        end
    end

    // ------------------------------------------------------------------
    // beat checking
    // ------------------------------------------------------------------
    integer beats, errors;
    integer t_cur, s_cur, g, w;
    reg [31:0] got_w, exp_w;

    integer cfg_ngrp, cfg_ksteps, cfg_tiles;
    reg [31:0] cfg_base, cfg_stride;

    always @(posedge clk) begin
        if (b_valid && b_ready) begin
            t_cur = beats / cfg_ksteps;
            s_cur = beats % cfg_ksteps;
            for (g = 0; g < 8; g = g + 1) begin
                for (w = 0; w < 8; w = w + 1) begin
                    got_w = b_data[256*g + 32*w +: 32];
                    if (g >= cfg_ngrp)
                        exp_w = 32'd0;
                    else
                        exp_w = cfg_base + (t_cur*cfg_ngrp + g)*cfg_stride +
                                s_cur*32 + 4*w;
                    if (got_w !== exp_w) begin
                        errors = errors + 1;
                        if (errors <= 8)
                            $display("  beat %0d (t%0d s%0d) slot %0d word %0d: got %08x exp %08x",
                                     beats, t_cur, s_cur, g, w, got_w, exp_w);
                    end
                end
            end
            beats = beats + 1;
        end
    end

    // ------------------------------------------------------------------
    // config: hold the request until the reader takes it (it may still be
    // finishing the previous job, where cfg_ready is low)
    // ------------------------------------------------------------------
    task arm;
        input [31:0] base;
        input [31:0] stride;
        input integer ngrp;
        input integer ksteps;
        input integer tiles;
        begin
            cfg_base   = base;
            cfg_stride = stride;
            cfg_ngrp   = ngrp;
            cfg_ksteps = ksteps;
            cfg_tiles  = tiles;
            cfg_cmd = 512'd0;
            cfg_cmd[79:72]   = 8'd2;             // buffer 2
            cfg_cmd[139:136] = ngrp[3:0];
            cfg_cmd[175:160] = tiles[15:0];
            cfg_cmd[191:176] = ksteps[15:0];
            cfg_cmd[351:320] = base;             // P6, B_base
            cfg_cmd[447:416] = stride;           // P9, B_group_stride
            // raise the request away from a clock edge: doing it in the same
            // time step as a posedge lets the DUT capture it before the task
            // ever sees cfg_ready, and the task then waits for a ready that
            // has already gone by.
            @(negedge clk);
            cfg_valid = 1'b1;
            while (!cfg_ready) @(negedge clk);
            @(posedge clk);                      // captured on this edge
            #1 cfg_valid = 1'b0;
        end
    endtask

    // ------------------------------------------------------------------
    // run
    // ------------------------------------------------------------------
    initial begin
        errors    = 0;
        beats     = 0;
        cfg_valid = 1'b0;
        cfg_cmd   = 512'd0;
        b_ready   = 1'b1;
        repeat (3) @(posedge clk);
        rst_n <= 1'b1;
        repeat (2) @(posedge clk);

        // ---- case 1: two tiles, 3 groups of 4 k-steps -------------------
        arm(32'h0000_0100, 32'd128, 3, 4, 2);
        fork
            begin : hold_beat
                wait (beats == 3);
                @(posedge clk);
                b_ready <= 1'b0;
                repeat (5) @(posedge clk);
                b_ready <= 1'b1;
            end
            begin : wait_done
                while (!b_done) @(posedge clk);
            end
        join
        if (beats != cfg_tiles*cfg_ksteps) begin
            $display("  case 1: %0d beats, expected %0d", beats,
                     cfg_tiles*cfg_ksteps);
            errors = errors + 1;
        end
        if (error !== `NPU_ST_OK) begin
            $display("  case 1: error %04x", error);
            errors = errors + 1;
        end

        // ---- case 2: one beat, all eight slots -------------------------
        beats = 0;
        arm(32'h0000_2000, 32'd1024, 8, 1, 1);
        while (!b_done) @(posedge clk);
        if (beats != 1) begin
            $display("  case 2: %0d beats, expected 1", beats);
            errors = errors + 1;
        end
        if (error !== `NPU_ST_OK) begin
            $display("  case 2: error %04x", error);
            errors = errors + 1;
        end

        repeat (2) @(posedge clk);
        if (errors == 0)
            $display("PASS: tb_b_frame_reader");
        else
            $display("FAIL: tb_b_frame_reader (%0d errors)", errors);
        $finish;
    end

    initial begin
        #200000;
        $display("FAIL: tb_b_frame_reader timeout");
        $finish;
    end

endmodule

`default_nettype wire

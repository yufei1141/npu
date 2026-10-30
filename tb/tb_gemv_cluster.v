//=============================================================================
// tb_gemv_cluster : numeric check of the W4 gemv array
//
//   The stream testbench (tb_mpu_engine) feeds zero weights, so the multiply
//   path has no numeric coverage.  This one drives real int4 data and real
//   E8M0 scales into gemv_cluster and checks the fp32 result against the
//   value the encoding implies.
//
//   Encoding under test (matches scripts/hw_pack.py):
//     b_din[256*n +: 256]   group n: 16 K rows x 4 columns x 4 bits, row
//                           major; row i occupies bits [16i +: 16] with the
//                           four column nibbles at 12, 8, 4, 0
//     b_din[2048+32*n+: 32] group n: one E8M0 byte per PE (K row 4p..4p+3)
//     a_din[256*m +: 256]   row m: 16 fp16 values
//
//   Case 1: every weight nibble = 4, every scale = 125 (2^-2), A = 1.0
//           -> each column = 16 * 4 * 0.25 = 16.0
//   Case 2: A = 1.0, weights 1 in K row 0 only -> column = 0.25
//           (isolates the K-row addressing inside a PE)
//=============================================================================
`timescale 1ns/1ps
`default_nettype none

module tb_gemv_cluster;

    reg clk = 0;
    reg rst_n = 0;
    always #5 clk = ~clk;

    reg  [1:0]    mode;
    reg  [2:0]    m_len;
    reg  [3:0]    n_grp;
    reg           beat_en, first, last;
    reg  [1023:0] a_din;
    reg  [2303:0] b_din;
    wire [4095:0] acc_out;
    wire          acc_valid;

    integer errors = 0;
    integer i;

    gemv_cluster u_dut (
        .clk(clk), .rst_n(rst_n),
        .mode(mode), .m_len(m_len), .n_grp(n_grp),
        .beat_en(beat_en), .first(first), .last(last),
        .a_din(a_din), .b_din(b_din),
        .acc_out(acc_out), .acc_valid(acc_valid)
    );

    // fp16 1.0 = 0x3C00
    localparam [15:0] ONE16 = 16'h3C00;

    // iverilog has no $bitstoshortreal, and $bitstoreal needs a double
    function real fp32_val;
        input [31:0] b;
        integer e, i;
        real m, r;
        begin
            e = b[30:23];
            m = b[22:0] * (1.0 / 8388608.0);
            if (b[30:0] == 31'd0) begin
                fp32_val = 0.0;
            end else begin
                r = (e == 0) ? m : (1.0 + m);
                for (i = 0; i < (e - 127); i = i + 1) r = r * 2.0;
                for (i = 0; i < (127 - e); i = i + 1) r = r / 2.0;
                fp32_val = b[31] ? -r : r;
            end
        end
    endfunction

    // ------------------------------------------------------------------
    // build one B beat: nib[g][row] is the 4-bit weight of column c in
    // group g, K row (4*pe + e).  scale[g][pe] is the E8M0 byte.
    // ------------------------------------------------------------------
    task automatic set_b;
        input integer fill;          // nibble value for every entry
        input integer only_row0;     // 1: only K row 0 nonzero
        integer g, r, c, pe, e;
        reg [3:0] nib;
        begin
            b_din = 2304'd0;
            for (g = 0; g < 8; g = g + 1) begin
                for (r = 0; r < 16; r = r + 1) begin
                    for (c = 0; c < 4; c = c + 1) begin
                        nib = (only_row0 && r != 0) ? 4'd0 : fill[3:0];
                        b_din[256*g + 16*r + 4*(3 - c) +: 4] = nib;
                    end
                    b_din[2048 + 32*g + 8*(r / 4) +: 8] = 8'd125;   // 2^-2
                end
            end
        end
    endtask

    task automatic set_a;
        integer m, k;
        begin
            a_din = 1024'd0;
            for (m = 0; m < 4; m = m + 1)
                for (k = 0; k < 16; k = k + 1)
                    a_din[256*m + 16*k +: 16] = ONE16;
        end
    endtask

    task automatic run_beat;
        begin
            @(negedge clk);
            beat_en = 1'b1; first = 1'b1; last = 1'b1;
            @(posedge clk);
            @(negedge clk);
            beat_en = 1'b0; first = 1'b0; last = 1'b0;
        end
    endtask

    task automatic check_row0;
        input [8*24-1:0] name;
        input integer    exp_i;
        integer c, bad;
        real exp_r, got;
        begin
            exp_r = exp_i / 1000.0;
            bad = 0;
            for (c = 0; c < 32; c = c + 1) begin
                got = fp32_val(acc_out[32*c +: 32]);
                if (got < exp_r - 0.001 || got > exp_r + 0.001) begin
                    if (bad < 4)
                        $display("  row0 col%0d = %f expected %f", c, got, exp_r);
                    bad = bad + 1;
                end
            end
            if (bad) begin
                $display("FAIL %0s: %0d/32 columns wrong", name, bad);
                errors = errors + 1;
            end else
                $display("PASS %0s: all 32 columns = %f", name, exp_r);
        end
    endtask

    initial begin
        mode = 2'b00; m_len = 3'd1; n_grp = 4'd8;
        beat_en = 0; first = 0; last = 0; a_din = 0; b_din = 0;
        repeat (4) @(negedge clk);
        rst_n = 1;
        repeat (2) @(negedge clk);

        // ---- case 1: every weight 4, scale 2^-2, A = 1 -> 16.0 ----
        set_a; set_b(4, 0);
        run_beat;
        wait (acc_valid);
        #1;
        check_row0("all-ones", 16000);
        // rows 2 and 3 are inactive: they must read back as zero
        if (acc_out[4095:2048] !== 2048'd0) begin
            $display("FAIL all-ones: inactive rows not cleared");
            errors = errors + 1;
        end

        // ---- case 2: weight only in K row 0 -> 4 * 2^-2 = 1.0 ----
        @(negedge clk);
        set_a; set_b(4, 1);
        run_beat;
        wait (acc_valid);
        #1;
        check_row0("row0-only", 1000);

        // ---- case 3: two beats accumulate (first=1 then 0) ----
        // the first pulse arrives 8 cycles after the beat, so wait for it
        // before starting the second beat, then wait for the second pulse.
        @(negedge clk);
        set_a; set_b(4, 0);
        run_beat;
        wait (acc_valid);
        @(posedge clk);
        @(negedge clk);
        beat_en = 1'b1; first = 1'b0; last = 1'b1;
        @(posedge clk);
        @(negedge clk);
        beat_en = 1'b0; last = 1'b0;
        wait (acc_valid);
        #1;
        check_row0("accumulate", 32000);

        if (errors == 0) $display("RESULT: PASS");
        else             $display("RESULT: FAIL (%0d errors)", errors);
        $finish;
    end

    initial begin
        #200000;
        $display("RESULT: FAIL (timeout)");
        $finish;
    end

endmodule

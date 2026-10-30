//=============================================================================
// tb_fp32_alu : directed cases plus a double precision cross-check
//
//   The reference widens both operands to double (exact), evaluates there and
//   rounds back to single with RNE (d2f).  Double rounding can only differ
//   from single rounding when the exact result sits on a float midpoint,
//   which random operands do not hit.  Directed cases cover the special
//   values, FTZ, overflow and the tie behaviour.
//=============================================================================
`timescale 1ns/1ps
`default_nettype none

module tb_fp32_alu;

    localparam OP_ADD = 3'd0;
    localparam OP_SUB = 3'd1;
    localparam OP_MUL = 3'd2;
    localparam OP_MAX = 3'd3;
    localparam OP_GT  = 3'd4;

    reg  [31:0] a, b;
    reg  [2:0]  op;
    wire [31:0] y;
    wire        pred;
    wire [7:0]  flags;

    fp32_alu u_dut (.a(a), .b(b), .op(op), .y(y), .pred(pred), .flags(flags));

    integer fails, n;

    // ---- float -> double (exact; only normals and zeros are used here) ----
    function [63:0] f2d;
        input [31:0] f;
        reg [10:0] e64;
        begin
            if (f[30:23] == 8'd0) begin
                f2d = {f[31], 63'd0};
            end else begin
                e64 = {3'd0, f[30:23]} + 11'd896;   // -127 + 1023
                f2d = {f[31], e64, f[22:0], 29'd0};
            end
        end
    endfunction

    // ---- double -> float with RNE; subnormal results flush to zero ----
    function [31:0] d2f;
        input [63:0] d;
        reg        sgn;
        reg [10:0] e;
        reg [51:0] m;
        reg signed [11:0] e32;
        reg [24:0] mant;
        reg        rup;
        begin
            sgn = d[63];
            e   = d[62:52];
            m   = d[51:0];
            if (e == 11'h7FF) begin
                d2f = {sgn, 8'hFF, (m != 0) ? 23'h400000 : 23'd0};
            end else if (e == 11'd0 && m == 52'd0) begin
                d2f = {sgn, 31'd0};
            end else begin
                e32 = $signed({1'b0, e}) - 12'sd1023 + 12'sd127;
                if (e32 <= 0) begin
                    d2f = {sgn, 31'd0};
                end else if (e32 >= 255) begin
                    d2f = {sgn, 8'hFF, 23'd0};
                end else begin
                    rup  = m[28] && ((|m[27:0]) || m[29]);
                    mant = {1'b0, m[51:29]} + {24'd0, rup};
                    if (mant[23]) begin
                        e32 = e32 + 1;
                        mant = {1'd0, mant[23:1]};
                    end
                    if (e32 >= 255)
                        d2f = {sgn, 8'hFF, 23'd0};
                    else
                        d2f = {sgn, e32[7:0], mant[22:0]};
                end
            end
        end
    endfunction

    task chk;
        input [31:0] va, vb;
        input [2:0]  o;
        input [31:0] exp_y;
        input [7:0]  exp_f;
        input [8*24-1:0] what;
        begin
            a = va; b = vb; op = o;
            #1;
            if (y !== exp_y) begin
                fails = fails + 1;
                $display("FAIL %0s: got %h exp %h", what, y, exp_y);
            end
            if (exp_f !== 8'hxx && flags !== exp_f) begin
                fails = fails + 1;
                $display("FAIL %0s: flags %h exp %h", what, flags, exp_f);
            end
        end
    endtask

    reg [31:0] ra, rb, rex;
    integer o, k;

    initial begin
        fails = 0;
        a = 0; b = 0; op = 0;
        $display("DBG start"); $fflush;

        // ---- directed ----
        chk(32'h3F80_0000, 32'h3F80_0000, OP_ADD, 32'h4000_0000, 8'h00, "1+1");
        chk(32'h3FC0_0000, 32'h4010_0000, OP_ADD, 32'h4070_0000, 8'h00, "1.5+2.25");
        chk(32'h3F80_0000, 32'h3F80_0000, OP_SUB, 32'h0000_0000, 8'h00, "1-1");
        chk(32'h0000_0000, 32'h8000_0000, OP_ADD, 32'h0000_0000, 8'h00, "+0+-0");
        chk(32'h8000_0000, 32'h8000_0000, OP_ADD, 32'h8000_0000, 8'h00, "-0+-0");
        chk(32'h3F80_0000, 32'h8000_0000, OP_SUB, 32'h3F80_0000, 8'h00, "1--0");
        // 1.0 + 2^-24 exactly halfway -> ties to even (stays 1.0)
        chk(32'h3F80_0000, 32'h3380_0000, OP_ADD, 32'h3F80_0000, 8'h08, "1+2^-24");
        chk(32'h3F80_0000, 32'h3380_0001, OP_ADD, 32'h3F80_0001, 8'h08, "1+2^-24+eps");
        // cancellation needing a big normalisation shift
        chk(32'h3F80_0001, 32'h3F80_0000, OP_SUB, 32'h3400_0000, 8'h00, "cancel");
        // overflow
        chk(32'h7F7F_FFFF, 32'h7F7F_FFFF, OP_ADD, 32'h7F80_0000, 8'h0A, "ovf add");
        chk(32'h7F7F_FFFF, 32'h4000_0000, OP_MUL, 32'h7F80_0000, 8'h0A, "ovf mul");
        // invalid
        chk(32'h7FC0_0000, 32'h3F80_0000, OP_ADD, 32'h7FC0_0000, 8'h01, "nan+1");
        chk(32'h7F80_0000, 32'h7F80_0000, OP_SUB, 32'h7FC0_0000, 8'h01, "inf-inf");
        chk(32'h7F80_0000, 32'h0000_0000, OP_MUL, 32'h7FC0_0000, 8'h01, "inf*0");
        chk(32'h7F80_0000, 32'h3F80_0000, OP_MUL, 32'h7F80_0000, 8'h00, "inf*1");
        // zero results keep the sign
        chk(32'h3F80_0000, 32'h0000_0000, OP_MUL, 32'h0000_0000, 8'h00, "1*+0");
        chk(32'h8000_0000, 32'h3F00_0000, OP_MUL, 32'h8000_0000, 8'h00, "-0*.5");
        // subnormal operand is flushed
        chk(32'h0000_0001, 32'h3F80_0000, OP_ADD, 32'h3F80_0000, 8'h0C, "subn+1");
        // exact products / sums inside the normal range
        chk(32'h3FC0_0000, 32'h4000_0000, OP_MUL, 32'h4040_0000, 8'h00, "1.5*2");
        chk(32'h42C8_0000, 32'h3DCC_CCCD, OP_MUL, 32'h4120_0000, 8'h08, "100*0.1");
        // compare / max
        chk(32'h3F80_0000, 32'h4000_0000, OP_MAX, 32'h4000_0000, 8'h00, "max");
        chk(32'hC000_0000, 32'h3F80_0000, OP_MAX, 32'h3F80_0000, 8'h00, "max neg");
        chk(32'h0000_0000, 32'h8000_0000, OP_MAX, 32'h0000_0000, 8'h00, "max +0-0");
        chk(32'hC000_0000, 32'hBF80_0000, OP_MAX, 32'hBF80_0000, 8'h00, "max -2,-1");
        chk(32'h4000_0000, 32'h3F80_0000, OP_GT, 32'h0000_0001, 8'h00, "gt 2>1");
        chk(32'hC000_0000, 32'hBF80_0000, OP_GT, 32'h0000_0000, 8'h00, "gt -2>-1");
        chk(32'h3F80_0000, 32'h7FC0_0000, OP_GT, 32'h0000_0000, 8'h01, "gt nan");

        $display("DBG directed done fails=%0d", fails); $fflush;
        // ---- random cross-check ----
        n = 0;
        for (k = 0; k < 2000; k = k + 1) begin
            ra = $random;
            ra[30:23] = 8'd100 + ({$random} % 8'd70);
            rb = $random;
            rb[30:23] = 8'd100 + ({$random} % 8'd70);
            for (o = 0; o < 3; o = o + 1) begin
                case (o)
                0: rex = d2f($realtobits($bitstoreal(f2d(ra)) + $bitstoreal(f2d(rb))));
                1: rex = d2f($realtobits($bitstoreal(f2d(ra)) - $bitstoreal(f2d(rb))));
                2: rex = d2f($realtobits($bitstoreal(f2d(ra)) * $bitstoreal(f2d(rb))));
                default: rex = 32'd0;
                endcase
                a = ra; b = rb; op = o[2:0];
                #1;
                if (y !== rex) begin
                    n = n + 1;
                    if (n < 6)
                        $display("  rand op=%0d a=%h b=%h got %h exp %h",
                                 o, ra, rb, y, rex);
                end
            end
        end
        if (n != 0) begin
            fails = fails + 1;
            $display("FAIL: %0d random mismatches", n);
        end

        if (fails == 0) $display("PASS: tb_fp32_alu");
        else            $display("FAIL: tb_fp32_alu (%0d)", fails);
        $finish;
    end

endmodule

`default_nettype wire

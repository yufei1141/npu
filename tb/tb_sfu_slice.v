//=============================================================================
// tb_sfu_slice : functional and numeric acceptance for the SFU datapath
//
//   Every op is driven through the slice interface and compared against a
//   double precision reference ($pow/$ln/$exp/$sqrt/$sin/$cos).  The fp32
//   reference is exact to double, so the only error is the datapath's own;
//   the ULP scale is taken from the DUT result bits.
//
//   Directed cases cover the special values, the FTZ behaviour, the SIN/COS
//   |x| <= 2^20 limit, SINCOS two-beat write-back, BF16/FP16 output rounding
//   and the `mode` Newton-step selection.
//=============================================================================
`timescale 1ns/1ps
`default_nettype none

module tb_sfu_slice;

    localparam OP_EXP2 = 4'd0, OP_LOG2 = 4'd1, OP_RCP  = 4'd2,
               OP_RSQ  = 4'd3, OP_SQRT = 4'd4, OP_SIN  = 4'd5,
               OP_COS  = 4'd6, OP_TANH = 4'd7, OP_SINC = 4'd8;

    localparam [31:0] QNAN = 32'h7FC0_0000, PINF = 32'h7F80_0000,
                      NINF = 32'hFF80_0000;

    reg          clk = 0, rst_n = 0;
    reg          in_valid = 0;
    wire         in_ready;
    reg  [3:0]   op = 0;
    reg  [1:0]   fmt_in = 0, fmt_out = 0, mode = 0;
    reg  [127:0] din = 0;
    wire         out_valid;
    reg          out_ready = 1;
    wire [127:0] dout;
    wire [7:0]   status;

    sfu_slice #(.ELEMS(4), .DATA_W(32)) dut (
        .clk(clk), .rst_n(rst_n),
        .in_valid(in_valid), .in_ready(in_ready),
        .op(op), .fmt_in(fmt_in), .fmt_out(fmt_out), .mode(mode),
        .din(din), .out_valid(out_valid), .out_ready(out_ready),
        .dout(dout), .status(status));

    always #5 clk = ~clk;

    integer fails = 0;
    integer checks = 0;
    reg [127:0] r0, r1;
    reg [7:0]   st;
    real        worst [0:8];
    integer     wcount [0:8];

    //-------------------------------------------------------------------------
    // float <-> double helpers
    //-------------------------------------------------------------------------
    function [63:0] f2d;
        input [31:0] f;
        reg [10:0] e64;
        begin
            if (f[30:23] == 8'd0) f2d = {f[31], 63'd0};
            else begin
                e64 = {3'd0, f[30:23]} + 11'd896;
                f2d = {f[31], e64, f[22:0], 29'd0};
            end
        end
    endfunction

    function real toreal;
        input [31:0] f;
        begin
            toreal = $bitstoreal(f2d(f));
        end
    endfunction

    function [31:0] d2f;
        input [63:0] d;
        reg        sgn;
        reg [10:0] e;
        reg [51:0] m;
        reg signed [11:0] e32;
        reg [24:0] mant;
        reg        rup;
        begin
            sgn = d[63]; e = d[62:52]; m = d[51:0];
            if (e == 11'h7FF) d2f = {sgn, 8'hFF, (m != 0) ? 23'h400000 : 23'd0};
            else if (e == 11'd0 && m == 52'd0) d2f = {sgn, 31'd0};
            else if (e == 11'd0) d2f = {sgn, 31'd0};
            else begin
                e32 = $signed({1'b0, e}) - 12'sd1023;
                if (e32 > 12'sd127) d2f = {sgn, 8'hFF, 23'd0};
                else if (e32 < -12'sd126) d2f = {sgn, 31'd0};
                else begin
                    mant = {1'b1, m[51:29]};
                    rup  = m[28] && ((|m[27:0]) || mant[0]);
                    mant = mant + rup;
                    if (mant[24]) begin mant = mant >> 1; e32 = e32 + 12'sd1; end
                    if (e32 > 12'sd127) d2f = {sgn, 8'hFF, 23'd0};
                    else d2f = {sgn, (e32 + 12'sd127), mant[22:0]};
                end
            end
        end
    endfunction

    function real ulp32;
        input [31:0] f;
        integer e;
        begin
            e = f[30:23];
            if (e == 0) ulp32 = 1.0 * (2.0 ** -149.0);
            else ulp32 = 2.0 ** ((e - 127) - 23);
        end
    endfunction

    //-------------------------------------------------------------------------
    // transaction driver
    //-------------------------------------------------------------------------
    task do_op;
        input [3:0]   o;
        input [127:0] d;
        input         two;
        begin
            @(negedge clk);
            op = o; din = d; in_valid = 1'b1;
            @(negedge clk);
            in_valid = 1'b0;
            while (!out_valid) @(negedge clk);
            r0 = dout; st = status;
            if (two) begin
                @(negedge clk);
                r1 = dout;
            end
            @(negedge clk);
            while (out_valid) @(negedge clk);
        end
    endtask

    // |got - want| in ulps, for the directed checks whose exact result is
    // not guaranteed without an FMA
    function within;
        input [31:0] got, want;
        input integer n;
        reg [31:0] a, b;
        begin
            a = got[31] ? (~got + 32'd1) : got;
            b = want[31] ? (~want + 32'd1) : want;
            within = ((got[31] == want[31]) && ((a > b ? a - b : b - a) <= n));
        end
    endfunction

    function [31:0] lane;
        input [127:0] v;
        input integer i;
        begin
            lane = v[i*32 +: 32];
        end
    endfunction

    //-------------------------------------------------------------------------
    // reference
    //-------------------------------------------------------------------------
    function real refval;
        input [3:0] o;
        input real  x;
        real z;
        begin
            case (o)
            OP_EXP2: refval = $pow(2.0, x);
            OP_LOG2: refval = $ln(x) / 0.69314718055994530942;
            OP_RCP:  refval = 1.0 / x;
            OP_RSQ:  refval = 1.0 / $sqrt(x);
            OP_SQRT: refval = $sqrt(x);
            OP_SIN:  refval = $sin(x);
            OP_COS:  refval = $cos(x);
            OP_TANH: begin
                z = $exp(2.0 * x);
                refval = (z - 1.0) / (z + 1.0);
            end
            default: refval = 0.0;
            endcase
        end
    endfunction

    task check_lane;
        input [3:0]   o;
        input [31:0]  got;
        input real    ref;
        input real    tol;
        input [255:0] tag;
        real err, u;
        begin
            // references outside the fp32 normal range are not comparable:
            // the datapath flushes subnormals (FTZ) and saturates to inf
            if (ref != 0.0 &&
                (ref < 1.17549435e-38 || ref > 3.40282346e38)) begin
                got = got;
            end else begin
            checks = checks + 1;
            if (got[30:23] == 8'hFF) begin
                $display("FAIL %0s: result is inf/nan for ref %g", tag, ref);
                fails = fails + 1;
            end else begin
                u   = ulp32(got);
                err = (toreal(got) - ref) / u;
                if (err < 0.0) err = -err;
                if (err > worst[o]) worst[o] = err;
                wcount[o] = wcount[o] + 1;
                if (err > tol && (ref != 0.0 || toreal(got) != 0.0)) begin
                    $display("FAIL %0s: got %08h (%.9g) ref %.9g  %.2f ulp",
                             tag, got, toreal(got), ref, err);
                    fails = fails + 1;
                end
            end
            end
        end
    endtask

    //-------------------------------------------------------------------------
    // random operands
    //-------------------------------------------------------------------------
    function [31:0] rf;
        input integer emin, emax, pos;
        reg [7:0] e;
        reg [31:0] mv, rnd;
        begin
            rnd = $random;
            mv  = $random;
            e = emin[7:0] + ((rnd & 32'h7FFF_FFFF) % (emax - emin + 1));
            rf = {pos ? 1'b0 : rnd[31], e, mv[22:0]};
        end
    endfunction

    integer k, j;
    reg [31:0] vec [0:3];
    real tol;
    real xr, rr;

    //-------------------------------------------------------------------------
    // run n random vectors of one op over the given exponent window
    //-------------------------------------------------------------------------
    task rand_op;
        input [3:0]   o;
        input integer n, emin, emax, pos;
        input real    t;
        begin
            tol = t;
            for (k = 0; k < n; k = k + 1) begin
                for (j = 0; j < 4; j = j + 1) vec[j] = rf(emin, emax, pos);
                do_op(o, {vec[3], vec[2], vec[1], vec[0]}, 1'b0);
                for (j = 0; j < 4; j = j + 1) begin
                    xr = toreal(vec[j]);
                    rr = refval(o, xr);
                    if (rr == rr) check_lane(o, lane(r0, j), rr, t, "rand");
                end
                // overflow / underflow are legitimate here: the random
                // exponent windows reach results beyond the fp32 range and
                // FTZ flags them.  Only invalid and div0 would be bugs.
                if (st[2] || st[3]) begin
                    $display("FAIL: status %02h on a normal input (%04h) in %h %h %h %h",
                             st, o, vec[0], vec[1], vec[2], vec[3]);
                    fails = fails + 1;
                end
            end
        end
    endtask

    //-------------------------------------------------------------------------
    integer i;
    reg [31:0] v;
    real       x;

    initial begin
        for (i = 0; i < 9; i = i + 1) begin
            worst[i] = 0.0;
            wcount[i] = 0;
        end

        repeat (4) @(posedge clk);
        rst_n = 1;
        repeat (2) @(posedge clk);

        //---------------------------------------------------------------
        // random numeric sweeps
        //---------------------------------------------------------------
        rand_op(OP_EXP2, 1500, 110, 130, 0, 2.0);
        rand_op(OP_LOG2, 1500, 1, 254, 1, 4.0);
        rand_op(OP_RCP,  1500, 1, 254, 0, 4.0);
        rand_op(OP_RSQ,  1500, 1, 254, 1, 4.0);
        rand_op(OP_SQRT, 1500, 1, 254, 1, 4.0);
        rand_op(OP_SIN,  1500, 100, 130, 0, 4.0);
        rand_op(OP_COS,  1500, 100, 130, 0, 4.0);
        rand_op(OP_TANH, 1500, 118, 135, 0, 6.0);

        //---------------------------------------------------------------
        // directed: specials and boundaries
        //---------------------------------------------------------------
        // exp2
        do_op(OP_EXP2, {32'hBF80_0000, QNAN, PINF, 32'd0}, 0);
        if (lane(r0,0) !== 32'h3F80_0000 || lane(r0,1) !== PINF ||
            lane(r0,2) !== QNAN || lane(r0,3) !== 32'h3F00_0000) begin
            $display("FAIL exp2 specials %h %h %h %h", lane(r0,0), lane(r0,1),
                     lane(r0,2), lane(r0,3));
            fails = fails + 1;
        end
        if (st[2] !== 1'b1) begin
            $display("FAIL exp2: NaN did not set invalid"); fails = fails + 1;
        end
        // exp2 overflow / underflow
        do_op(OP_EXP2, {32'h42C8_0000, 32'hC316_0000, 32'h4317_0000, 32'd0}, 0);
        if (lane(r0,1) !== PINF || !st[0]) begin
            $display("FAIL exp2 overflow %h st=%h", lane(r0,1), st); fails = fails + 1;
        end
        if (lane(r0,2) !== 32'd0 || !st[1]) begin
            $display("FAIL exp2 underflow %h st=%h", lane(r0,2), st); fails = fails + 1;
        end

        // log2 : log2(1)=0, log2(2)=1, log2(0.5)=-1, log2(0)=-inf+div0,
        //        log2(-1)=nan+invalid, log2(inf)=inf
        do_op(OP_LOG2, {32'h0000_0000, 32'h3F00_0000, 32'h4000_0000, 32'h3F80_0000}, 0);
        if (lane(r0,0) !== 32'd0 || lane(r0,1) !== 32'h3F80_0000 ||
            lane(r0,2) !== 32'hBF80_0000 || lane(r0,3) !== NINF) begin
            $display("FAIL log2 exact %h %h %h %h", lane(r0,0), lane(r0,1),
                     lane(r0,2), lane(r0,3));
            fails = fails + 1;
        end
        if (st[3] !== 1'b1) begin
            $display("FAIL log2: log2(0) did not set div0"); fails = fails + 1;
        end
        do_op(OP_LOG2, {32'h7F7F_FFFF, 32'h0080_0000, PINF, 32'hBF80_0000}, 0);
        if (lane(r0,0) !== QNAN || !st[2]) begin
            $display("FAIL log2(-1) %h st=%h", lane(r0,0), st); fails = fails + 1;
        end
        if (lane(r0,1) !== PINF) begin
            $display("FAIL log2(inf) %h", lane(r0,1)); fails = fails + 1;
        end
        if (lane(r0,3) !== 32'h4300_0000) begin    // log2(FLT_MAX) ~ 128
            $display("FAIL log2(max) %h", lane(r0,3)); fails = fails + 1;
        end

        // rcp
        do_op(OP_RCP, {32'h7F80_0000, 32'h0000_0000, 32'hC000_0000, 32'h3F80_0000}, 0);
        if (!within(lane(r0,0), 32'h3F80_0000, 1) ||
            !within(lane(r0,1), 32'hBF00_0000, 1) ||
            lane(r0,2) !== PINF || lane(r0,3) !== 32'd0) begin
            $display("FAIL rcp specials %h %h %h %h", lane(r0,0), lane(r0,1),
                     lane(r0,2), lane(r0,3));
            fails = fails + 1;
        end
        if (st[3] !== 1'b1) begin
            $display("FAIL rcp: 1/0 did not set div0"); fails = fails + 1;
        end

        // sqrt / rsqrt
        do_op(OP_SQRT, {PINF, 32'hBF80_0000, 32'd0, 32'h4080_0000}, 0);
        if (lane(r0,0) !== 32'h4000_0000 || lane(r0,1) !== 32'd0 ||
            lane(r0,2) !== QNAN || lane(r0,3) !== PINF) begin
            $display("FAIL sqrt %h %h %h %h", lane(r0,0), lane(r0,1),
                     lane(r0,2), lane(r0,3));
            fails = fails + 1;
        end
        if (st[2] !== 1'b1) begin
            $display("FAIL sqrt(-1) did not set invalid"); fails = fails + 1;
        end
        do_op(OP_RSQ, {PINF, 32'hBF80_0000, 32'd0, 32'h4080_0000}, 0);
        if (lane(r0,0) !== 32'h3F00_0000 || lane(r0,1) !== PINF ||
            lane(r0,2) !== QNAN || lane(r0,3) !== 32'd0) begin
            $display("FAIL rsqrt %h %h %h %h", lane(r0,0), lane(r0,1),
                     lane(r0,2), lane(r0,3));
            fails = fails + 1;
        end

        // sin / cos exact points
        do_op(OP_SIN, {32'h4000_0000, 32'h3F80_0000, 32'h3FC9_0FDB, 32'd0}, 0);
        if (lane(r0,0) !== 32'd0 || lane(r0,1) !== 32'h3F80_0000 ||
            lane(r0,2) !== 32'h3F57_6AA4) begin
            $display("FAIL sin(0)=%h sin(pi/2)=%h sin(1)=%h", lane(r0,0),
                     lane(r0,1), lane(r0,2));
            fails = fails + 1;
        end
        do_op(OP_COS, {32'h4049_0FDB, 32'h3F80_0000, 32'h3FC9_0FDB, 32'd0}, 0);
        if (lane(r0,0) !== 32'h3F80_0000 || lane(r0,2) !== 32'h3F0A_5140 ||
            lane(r0,3) !== 32'hBF80_0000) begin
            $display("FAIL cos(0)=%h cos(1)=%h cos(pi)=%h", lane(r0,0),
                     lane(r0,2), lane(r0,3));
            fails = fails + 1;
        end
        if (!within(lane(r0,1), 32'hB33B_BD2E, 4) && lane(r0,1) !== 32'd0) begin
            $display("FAIL cos(pi/2)=%h", lane(r0,1)); fails = fails + 1;
        end

        // sin argument limit: 2^20 is accepted, just above is invalid
        do_op(OP_SIN, {32'h7F80_0000, 32'hFB80_0000, 32'h4980_0001, 32'h4980_0000}, 0);
        v = lane(r0,0);
        if (v[30:23] == 8'hFF || lane(r0,1) !== QNAN ||
            lane(r0,2) !== QNAN || lane(r0,3) !== QNAN) begin
            $display("FAIL sin range limit %h %h %h %h", lane(r0,0), lane(r0,1),
                     lane(r0,2), lane(r0,3));
            fails = fails + 1;
        end

        // tanh saturation
        do_op(OP_TANH, {32'h4049_0FDB, 32'hC120_0000, 32'h4120_0000, 32'd0}, 0);
        if (lane(r0,0) !== 32'd0 || lane(r0,1) !== 32'h3F80_0000 ||
            lane(r0,2) !== 32'hBF80_0000) begin
            $display("FAIL tanh sat %h %h %h", lane(r0,0), lane(r0,1), lane(r0,2));
            fails = fails + 1;
        end
        if (!within(lane(r0,3), 32'h3F7F_0BB0, 2)) begin   // tanh(pi) = 0.9962721
            $display("FAIL tanh(pi) %h", lane(r0,3)); fails = fails + 1;
        end

        //---------------------------------------------------------------
        // SINCOS : two write-back beats, sin then cos
        //---------------------------------------------------------------
        do_op(OP_SINC, {32'h4049_0FDB, 32'h3F80_0000, 32'h3FC9_0FDB, 32'd0}, 1'b1);
        if (lane(r0,1) !== 32'h3F80_0000 || lane(r1,0) !== 32'h3F80_0000 ||
            lane(r0,0) !== 32'd0 || lane(r0,2) !== 32'h3F57_6AA4 ||
            lane(r1,2) !== 32'h3F0A_5140 || lane(r1,3) !== 32'hBF80_0000) begin
            $display("FAIL sincos sin=%h %h %h cos=%h %h %h",
                     lane(r0,0), lane(r0,1), lane(r0,2),
                     lane(r1,0), lane(r1,2), lane(r1,3));
            fails = fails + 1;
        end

        //---------------------------------------------------------------
        // output formats : BF16 and FP16 rounding of a known result
        //---------------------------------------------------------------
        fmt_out = 2'd1;                       // BF16
        do_op(OP_RCP, {32'd0, 32'd0, 32'd0, 32'h4000_0000}, 1'b0);
        if (lane(r0,0) !== 32'h0000_3F00) begin
            $display("FAIL bf16 out %h", lane(r0,0)); fails = fails + 1;
        end
        fmt_out = 2'd2;                       // FP16
        do_op(OP_RCP, {32'd0, 32'd0, 32'd0, 32'h4000_0000}, 1'b0);
        if (lane(r0,0) !== 32'h0000_3800) begin
            $display("FAIL fp16 out %h", lane(r0,0)); fails = fails + 1;
        end
        // FP16 input : 1.5 -> rcp = 0.6666..  (0x3E00 = 1.5)
        fmt_in = 2'd2; fmt_out = 2'd0;
        do_op(OP_RCP, {32'd0, 32'd0, 32'd0, 32'h0000_3E00}, 1'b0);
        if (lane(r0,0) < 32'h3F2A_AAA7 || lane(r0,0) > 32'h3F2A_AAAF) begin
            $display("FAIL fp16 in %h (want ~3f2aaaab)", lane(r0,0));
            fails = fails + 1;
        end
        fmt_in = 2'd0; fmt_out = 2'd0;

        //---------------------------------------------------------------
        // accurate mode : one extra Newton step
        //---------------------------------------------------------------
        mode = 2'd1;
        rand_op(OP_RCP, 100, 1, 254, 0, 4.0);
        mode = 2'd0;

        //---------------------------------------------------------------
        $display("");
        $display("  op    worst   n");
        $display("  EXP2 %6.2f  %0d", worst[0], wcount[0]);
        $display("  LOG2 %6.2f  %0d", worst[1], wcount[1]);
        $display("  RCP  %6.2f  %0d", worst[2], wcount[2]);
        $display("  RSQ  %6.2f  %0d", worst[3], wcount[3]);
        $display("  SQRT %6.2f  %0d", worst[4], wcount[4]);
        $display("  SIN  %6.2f  %0d", worst[5], wcount[5]);
        $display("  COS  %6.2f  %0d", worst[6], wcount[6]);
        $display("  TANH %6.2f  %0d", worst[7], wcount[7]);
        $display("  checks %0d, failures %0d", checks, fails);
        if (fails == 0) $display("PASS: tb_sfu_slice");
        else            $display("FAIL: tb_sfu_slice (%0d)", fails);
        $finish;
    end

    initial begin
        #5000000;
        $display("FAIL: tb_sfu_slice timeout");
        $finish;
    end

endmodule
`default_nettype wire

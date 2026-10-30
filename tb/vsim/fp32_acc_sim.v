//=============================================================================
// fp32_acc_sim : Verilator-only stand-in for tensorcore/rtl/fp32_acc.v
//
//   The tensorcore original accumulates with a delayed assignment to an array
//   inside a for loop:
//
//       always @(posedge clk or negedge rst_n)
//           for (i = 0; i < N; i = i + 1)
//               acc[i] <= acc_en ? fadd(acc[i], y[32*i +: 32]) : y[...];
//
//   The simulator cannot schedule that (BLKLOOPINIT, "Unsupported: Delayed
//   assignment to array inside for loops"), and the lint suppression that
//   makes it build leaves the assignment silently dropped: acc never moves
//   off its reset value.  That is invisible end to end because everything
//   upstream is correct -- the PE emits 4.0, the reduce emits 16.0, and only
//   the fp32 accumulator stays 0.
//
//   tensorcore/ is not modified.  This file is an elaboration-identical
//   replacement used only by scripts/verilate.sh: fadd() is copied verbatim
//   (bit-exact arithmetic), and the only change is that the per-lane update
//   is written as one always block per lane, so no array is assigned inside
//   a loop.  iverilog keeps compiling the original and stays the reference
//   simulator.
//
//   If fp32_acc.v is ever rewritten to be Verilator-clean, delete this file
//   and the SIM_SHIM mapping in scripts/verilate.sh.
//=============================================================================
`timescale 1ns/1ps
`default_nettype none

module fp32_acc #(
    parameter ROWS  = 4,
    parameter COLS  = 32
)(
    input  wire                    clk,
    input  wire                    rst_n,
    input  wire                    valid,
    input  wire                    acc_en,    // 0: load y, 1: acc + y
    input  wire [ROWS*COLS*32-1:0] y,
    output wire [ROWS*COLS*32-1:0] acc_out
);

    localparam N = ROWS*COLS;

    // ------------------------------------------------------------------
    // single-cycle fp32 add (RNE) -- verbatim from tensorcore/rtl/fp32_acc.v
    // ------------------------------------------------------------------
    function [31:0] fadd;
        input [31:0] a;
        input [31:0] b;
        reg [7:0]  ea, eb, d, sh;
        reg [23:0] ma, mb, mnt;
        reg        sa, sb, sg;
        reg [26:0] xa, xb;
        reg [27:0] sm, nrm;
        reg        stk;
        reg signed [11:0] e;
        reg [5:0]  p;
        reg [24:0] mi;
        reg        inc;
        integer k;
        begin
            ea = a[30:23]; sa = a[31];
            eb = b[30:23]; sb = b[31];

            if ((ea == 8'd0) && (a[22:0] == 23'd0)) begin
                fadd = b;                       // 0 + b
            end else if ((eb == 8'd0) && (b[22:0] == 23'd0)) begin
                fadd = a;                       // a + 0
            end else begin
                ma = {1'b1, a[22:0]};
                mb = {1'b1, b[22:0]};

                // align the smaller exponent operand, keep guard bits + sticky
                if (ea >= eb) begin
                    e  = $signed({4'd0, ea});
                    d  = ea - eb;
                    xa = {ma, 3'b000};
                    sg = sa;
                    if (d > 8'd26) begin
                        xb = 27'd1;
                    end else if (d == 8'd0) begin
                        xb = {mb, 3'b000};
                    end else begin
                        xb = {mb, 3'b000} >> d;
                        if (|({mb, 3'b000} & ((27'd1 << d) - 27'd1))) xb[0] = 1'b1;
                    end
                end else begin
                    e  = $signed({4'd0, eb});
                    d  = eb - ea;
                    xa = {mb, 3'b000};
                    sg = sb;
                    if (d > 8'd26) begin
                        xb = 27'd1;
                    end else if (d == 8'd0) begin
                        xb = {ma, 3'b000};
                    end else begin
                        xb = {ma, 3'b000} >> d;
                        if (|({ma, 3'b000} & ((27'd1 << d) - 27'd1))) xb[0] = 1'b1;
                    end
                end

                if (sa == sb) begin
                    sm = {1'b0, xa} + {1'b0, xb};
                end else if (xa >= xb) begin
                    sm = {1'b0, xa} - {1'b0, xb};
                end else begin
                    sm = {1'b0, xb} - {1'b0, xa};
                    sg = ~sg;
                end

                if (sm == 28'd0) begin
                    fadd = 32'd0;
                end else begin
                    p = 6'd0;
                    for (k = 0; k < 28; k = k + 1)
                        if (sm[k]) p = k[5:0];

                    stk = 1'b0;
                    if (p > 6'd26) begin
                        sh  = p - 6'd26;
                        stk = |(sm & ((28'd1 << sh) - 28'd1));
                        nrm = sm >> sh;
                        e   = e + $signed({6'd0, sh});
                    end else if (p < 6'd26) begin
                        nrm = sm << (6'd26 - p);
                        e   = e - $signed({6'd0, (6'd26 - p)});
                    end else begin
                        nrm = sm;
                    end

                    nrm[0] = nrm[0] | stk;          // fold sticky into S

                    mnt = nrm[26:3];
                    inc = nrm[2] && (nrm[1] || nrm[0] || mnt[0]);
                    mi  = {1'b0, mnt} + {24'd0, inc};
                    if (mi[24]) begin
                        mnt = mi[24:1];
                        e   = e + 12'sd1;
                    end else begin
                        mnt = mi[23:0];
                    end

                    if (e >= 12'sd255)   fadd = {sg, 8'hFF, 23'd0};
                    else if (e < 12'sd1) fadd = {sg, 31'd0};
                    else                 fadd = {sg, e[7:0], mnt[22:0]};
                end
            end
        end
    endfunction

    // ------------------------------------------------------------------
    // accumulator : one always block per lane (no array write in a loop)
    // ------------------------------------------------------------------
    reg [31:0] acc [0:N-1];

    genvar gi, gj;
    generate
        for (gi = 0; gi < N; gi = gi + 1) begin : g_acc
            always @(posedge clk or negedge rst_n) begin
                if (!rst_n)
                    acc[gi] <= 32'd0;
                else if (valid)
                    acc[gi] <= acc_en ? fadd(acc[gi], y[32*gi +: 32])
                                      : y[32*gi +: 32];
            end
        end
        for (gj = 0; gj < N; gj = gj + 1) begin : g_out
            assign acc_out[32*gj +: 32] = acc[gj];
        end
    endgenerate

endmodule

`default_nettype wire

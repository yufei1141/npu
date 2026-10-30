//=============================================================================
// fp32_alu : FP32 add / subtract / multiply / max / greater-than
//
//   Combinational core used by the VPU lanes, following the numeric contract
//   of docs/npu_integration/12_format_and_numeric.md:
//
//   - finite normals and +/-0 are the working range
//   - subnormal operands are flushed to a signed zero (underflow|inexact)
//   - a subnormal result is flushed to a signed zero (underflow|inexact)
//   - RNE with guard / round / sticky, carry out of the mantissa bumps the
//     exponent, a result beyond the largest finite value becomes infinity
//     with overflow|inexact
//   - NaN/Inf operands set invalid; the result is a quiet NaN, or infinity
//     when the operation is well defined on infinities
//   - MAX keeps the earlier operand on equality (including +0 / -0), GT
//     clears its predicate on invalid
//
//   The registered ready/valid wrapper lives in lane_alu (vector_engine).
//
//   flags: [0] invalid, [1] overflow, [2] underflow, [3] inexact
//=============================================================================
`timescale 1ns/1ps
`default_nettype none

module fp32_alu (
    input  wire [31:0] a,
    input  wire [31:0] b,
    input  wire [2:0]  op,
    output reg  [31:0] y,
    output reg         pred,
    output reg  [7:0]  flags
);

    localparam OP_ADD = 3'd0;
    localparam OP_SUB = 3'd1;
    localparam OP_MUL = 3'd2;
    localparam OP_MAX = 3'd3;
    localparam OP_GT  = 3'd4;

    localparam [31:0] QNAN = 32'h7FC0_0000;

    // ------------------------------------------------------------------
    // operand classification (shared)
    // ------------------------------------------------------------------
    wire        sa = a[31];
    wire        sb = b[31];
    wire [7:0]  ea = a[30:23];
    wire [7:0]  eb = b[30:23];
    wire [22:0] fa = a[22:0];
    wire [22:0] fb = b[22:0];

    wire nan_a  = (ea == 8'hFF) && (fa != 23'd0);
    wire nan_b  = (eb == 8'hFF) && (fb != 23'd0);
    wire inf_a  = (ea == 8'hFF) && (fa == 23'd0);
    wire inf_b  = (eb == 8'hFF) && (fb == 23'd0);
    wire zero_a = (ea == 8'd0) && (fa == 23'd0);
    wire zero_b = (eb == 8'd0) && (fb == 23'd0);
    wire sub_a  = (ea == 8'd0) && (fa != 23'd0);
    wire sub_b  = (eb == 8'd0) && (fb != 23'd0);

    wire sb_eff = sb ^ (op == OP_SUB);

    // flushed view of the operands
    wire [7:0]  ea_f = sub_a ? 8'd0 : ea;
    wire [7:0]  eb_f = sub_b ? 8'd0 : eb;
    wire [22:0] fa_f = sub_a ? 23'd0 : fa;
    wire [22:0] fb_f = sub_b ? 23'd0 : fb;
    wire        za_f = (ea_f == 8'd0) && (fa_f == 23'd0);
    wire        zb_f = (eb_f == 8'd0) && (fb_f == 23'd0);

    wire [23:0] ma = za_f ? 24'd0 : {1'b1, fa_f};
    wire [23:0] mb = zb_f ? 24'd0 : {1'b1, fb_f};

    // ------------------------------------------------------------------
    // compare / max
    // ------------------------------------------------------------------
    reg cmp_max_sel;        // 1 -> choose a
    reg cmp_flags;
    reg [31:0] max_y;
    reg        gt_pred;

    always @* begin
        // magnitude / sign ordering for MAX and GT
        cmp_flags = (sub_a || sub_b) ? 8'h0C : 8'h00;
        if (nan_a || nan_b) begin
            cmp_max_sel = 1'b1;
            cmp_flags   = cmp_flags | 8'h01;
            gt_pred     = 1'b0;
        end else if (za_f && zb_f) begin
            // +0 == -0; keep the earlier operand, never report greater
            cmp_max_sel = 1'b1;
            gt_pred     = 1'b0;
        end else begin
            if (sa ^ sb) begin
                cmp_max_sel = !sa;              // positive wins
                gt_pred     = (sa == 1'b0);
            end else if (ea_f != eb_f) begin
                cmp_max_sel = (ea_f > eb_f) ^ sa;
                gt_pred     = ((ea_f > eb_f) ^ sa);
            end else if (fa_f != fb_f) begin
                cmp_max_sel = (fa_f > fb_f) ^ sa;
                gt_pred     = ((fa_f > fb_f) ^ sa);
            end else begin
                cmp_max_sel = 1'b1;             // equal: keep a
                gt_pred     = 1'b0;
            end
        end
        max_y = cmp_max_sel ? (sub_a ? {sa, 31'd0} : a)
                            : (sub_b ? {sb, 31'd0} : b);
        if (nan_a || nan_b) max_y = QNAN;
    end

    // ------------------------------------------------------------------
    // add / subtract
    // ------------------------------------------------------------------
    reg [31:0] add_y;
    reg [7:0]  add_flags;
    reg [26:0] sum;
    reg [26:0] big_e, sml_e;
    reg [26:0] norm;
    reg [8:0]  d;
    reg        s_big;
    reg        sticky;
    reg [7:0]  e_big;
    reg signed [10:0] e_n;
    reg [9:0]  nsh;
    reg        rup_a, rup_m;
    reg [24:0] mant_a, mant_m;
    integer    i;

    always @* begin
        add_y     = 32'd0;
        add_flags = (sub_a || sub_b) ? 8'h0C : 8'h00;

        if (nan_a || nan_b) begin
            add_y     = QNAN;
            add_flags = add_flags | 8'h01;
        end else if (inf_a && inf_b) begin
            if (sa != sb_eff) begin
                add_y     = QNAN;
                add_flags = add_flags | 8'h01;
            end else begin
                add_y = {sa, 8'hFF, 23'd0};
            end
        end else if (inf_a) begin
            add_y = {sa, 8'hFF, 23'd0};
        end else if (inf_b) begin
            add_y = {sb_eff, 8'hFF, 23'd0};
        end else if (za_f && zb_f) begin
            add_y = {sa & sb_eff, 31'd0};       // -0 only if both are -0
        end else if (za_f) begin
            add_y = {sb_eff, eb_f, fb_f};
        end else if (zb_f) begin
            add_y = {sa, ea_f, fa_f};
        end else begin
            // larger magnitude first
            if (ea_f > eb_f || (ea_f == eb_f && ma >= mb)) begin
                e_big = ea_f;
                s_big = sa;
                big_e = {1'b0, ma, 2'b00};
                sml_e = {1'b0, mb, 2'b00};
                d     = {1'b0, ea_f} - {1'b0, eb_f};
            end else begin
                e_big = eb_f;
                s_big = sb_eff;
                big_e = {1'b0, mb, 2'b00};
                sml_e = {1'b0, ma, 2'b00};
                d     = {1'b0, eb_f} - {1'b0, ea_f};
            end

            // Bits shifted below the two guard bits are not folded into the
            // operand: for a subtraction that would turn the borrow into a
            // lost fraction.  They are kept as a separate sticky flag and
            // merged with the round bits after normalisation.  sum is then
            // the floor of the exact result: for an addition the dropped
            // fraction only lowers it, for a subtraction the truncated
            // subtrahend makes it one too large, which is stepped back here.
            sticky = |(sml_e & ~(27'h7FF_FFFF << d));
            sml_e  = sml_e >> d;

            e_n = {2'b00, e_big};
            if (sa == sb_eff) begin
                sum = big_e + sml_e;
            end else begin
                sum = big_e - sml_e - {26'd0, sticky};
            end

            if (sum == 27'd0) begin
                add_y = 32'd0;                  // exact cancellation is +0
            end else begin
                // normalise so that the leading one sits at bit 25
                norm = sum;
                if (norm[26]) begin
                    if (norm[0]) sticky = 1'b1; // shifted out of the round bits
                    norm = norm >> 1;
                    e_n  = e_n + 1;
                end else begin
                    for (i = 0; i < 26; i = i + 1) begin
                        if (!norm[25]) begin
                            norm = norm << 1;
                            e_n  = e_n - 1;
                        end
                    end
                end

                // round to nearest, ties to even
                rup_a  = norm[1] && (norm[0] || sticky || norm[2]);
                mant_a = {1'b0, norm[25:2]} + {24'd0, rup_a};
                if (mant_a[24]) begin
                    mant_a = mant_a >> 1;
                    e_n  = e_n + 1;
                end
                if (norm[1] || norm[0] || sticky)
                    add_flags = add_flags | 8'h08;

                if (e_n <= 0) begin
                    add_y     = {s_big, 31'd0}; // FTZ
                    add_flags = add_flags | 8'h0C;
                end else if (e_n >= 255) begin
                    add_y     = {s_big, 8'hFF, 23'd0};
                    add_flags = add_flags | 8'h0A;
                end else begin
                    add_y = {s_big, e_n[7:0], mant_a[22:0]};
                end
            end
        end
    end

    // ------------------------------------------------------------------
    // multiply
    // ------------------------------------------------------------------
    reg [31:0] mul_y;
    reg [7:0]  mul_flags;
    reg [47:0] prod;
    reg [47:0] pnorm;
    reg signed [11:0] e_m;

    always @* begin
        mul_y     = 32'd0;
        mul_flags = (sub_a || sub_b) ? 8'h0C : 8'h00;

        if (nan_a || nan_b) begin
            mul_y     = QNAN;
            mul_flags = mul_flags | 8'h01;
        end else if ((inf_a && zb_f) || (inf_b && za_f)) begin
            mul_y     = QNAN;
            mul_flags = mul_flags | 8'h01;
        end else if (inf_a || inf_b) begin
            mul_y = {sa ^ sb, 8'hFF, 23'd0};
        end else if (za_f || zb_f) begin
            mul_y = {sa ^ sb, 31'd0};
        end else begin
            prod   = ma * mb;                   // 1.xxx * 1.xxx
            e_m    = {3'b000, ea_f} + {3'b000, eb_f} - 12'd127;
            if (prod[47]) begin
                pnorm = prod;
                e_m   = e_m + 1;
            end else begin
                pnorm = prod << 1;
            end
            rup_m = pnorm[23] && ((|pnorm[22:0]) || pnorm[24]);
            mant_m = {1'b0, pnorm[46:24]} + {24'd0, rup_m};
            if (mant_m[24]) begin
                mant_m = mant_m >> 1;
                e_m  = e_m + 1;
            end
            if (pnorm[23] || (|pnorm[22:0])) mul_flags = mul_flags | 8'h08;

            if (e_m <= 0) begin
                mul_y     = {sa ^ sb, 31'd0};
                mul_flags = mul_flags | 8'h0C;
            end else if (e_m >= 255) begin
                mul_y     = {sa ^ sb, 8'hFF, 23'd0};
                mul_flags = mul_flags | 8'h0A;
            end else begin
                mul_y = {sa ^ sb, e_m[7:0], mant_m[22:0]};
            end
        end
    end

    // ------------------------------------------------------------------
    // output select
    // ------------------------------------------------------------------
    always @* begin
        pred = 1'b0;
        case (op)
        OP_ADD: begin y = add_y; flags = add_flags; end
        OP_SUB: begin y = add_y; flags = add_flags; end
        OP_MUL: begin y = mul_y; flags = mul_flags; end
        OP_MAX: begin y = max_y; flags = cmp_flags; end
        OP_GT:  begin pred = gt_pred;
                        y    = gt_pred ? 32'd1 : 32'd0;
                        flags = cmp_flags; end
        default: begin y = QNAN; flags = 8'h01; end
        endcase
    end

endmodule

`default_nettype wire

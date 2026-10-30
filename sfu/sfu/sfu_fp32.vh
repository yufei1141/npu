//=============================================================================
// sfu_fp32.vh : FP32 combinational helpers for the SFU datapath
//
//   Included inside the body of sfu_slice.  Everything here is a Verilog-2001
//   function, i.e. purely combinational logic once synthesised.
//
//   Arithmetic contract
//   -------------------
//   * IEEE-754 binary32, round-to-nearest-even, one rounding per operation.
//   * fp_add  : 34-bit aligned datapath (9 guard bits) with a carry-aware
//               sticky bit.  Subtraction uses the "true result lies below the
//               computed one" rounding rule, so add and sub are both exact RNE.
//   * fp_mul  : 48-bit product, RNE.
//   * FTZ     : subnormal inputs and results are flushed to zero.  This is a
//               documented deviation from the FTZ=0 default in
//               sfu/arch/microarch.md.
//   * NaN     : every NaN produced here is the quiet default 0x7FC00000.
//
//   No FMA is built.  fp_add and fp_mul round separately, which the accuracy
//   budget in sfu_slice.v accounts for.
//=============================================================================
`ifndef SFU_FP32_VH
`define SFU_FP32_VH

localparam [31:0] SF_QNAN = 32'h7FC0_0000;
localparam [31:0] SF_PINF = 32'h7F80_0000;
localparam [31:0] SF_ONE  = 32'h3F80_0000;
localparam [31:0] SF_TWO  = 32'h4000_0000;
localparam [31:0] SF_HALF = 32'h3F00_0000;
localparam [31:0] SF_ONE5 = 32'h3FC0_0000;
localparam [31:0] SF_NEG1 = 32'hBF80_0000;

//-----------------------------------------------------------------------------
// predicates
//-----------------------------------------------------------------------------
function sf_is_nan;
    input [31:0] a;
    begin
        sf_is_nan = ((a & 32'h7F80_0000) == 32'h7F80_0000) &&
                    ((a & 32'h007F_FFFF) != 32'd0);
    end
endfunction

function sf_is_inf;
    input [31:0] a;
    begin
        sf_is_inf = ((a & 32'h7FFF_FFFF) == 32'h7F80_0000);
    end
endfunction

function sf_is_zero;
    input [31:0] a;
    begin
        sf_is_zero = ((a & 32'h7FFF_FFFF) == 32'd0);
    end
endfunction

function sf_is_sub;
    input [31:0] a;
    begin
        sf_is_sub = (a[30:23] == 8'd0) && (a[22:0] != 23'd0);
    end
endfunction

function [31:0] sf_neg;
    input [31:0] a;
    begin
        sf_neg = a ^ 32'h8000_0000;
    end
endfunction

function [31:0] sf_abs;
    input [31:0] a;
    begin
        sf_abs = a & 32'h7FFF_FFFF;
    end
endfunction

//-----------------------------------------------------------------------------
// fp32 add / subtract, RNE, FTZ
//-----------------------------------------------------------------------------
function [31:0] sf_add;
    input [31:0] a;
    input [31:0] b;
    reg [33:0] mx_s, mn_s, acc;
    reg [23:0] mx, mn, frac, tm;
    reg [7:0]  ex, en, d, te;
    reg [5:0]  lz;
    reg signed [10:0] ers;
    reg        sx, sn, sticky, guard, up, ts;
    integer    j;
    begin
        sx = a[31];
        sn = b[31];
        if (sf_is_nan(a) || sf_is_nan(b)) begin
            sf_add = SF_QNAN;
        end else if (sf_is_inf(a) && sf_is_inf(b)) begin
            sf_add = (a[31] == b[31]) ? a : SF_QNAN;
        end else if (sf_is_inf(a)) begin
            sf_add = a;
        end else if (sf_is_inf(b)) begin
            sf_add = b;
        end else if (sf_is_sub(a) || sf_is_sub(b)) begin
            // FTZ on the inputs
            if (sf_is_sub(a) && sf_is_sub(b))   sf_add = {sx & sn, 31'd0};
            else if (sf_is_sub(a))              sf_add = b;
            else                                sf_add = a;
        end else if (sf_is_zero(a) && sf_is_zero(b)) begin
            sf_add = {sx & sn, 31'd0};
        end else if (sf_is_zero(a)) begin
            sf_add = b;
        end else if (sf_is_zero(b)) begin
            sf_add = a;
        end else begin
            ex = a[30:23]; mx = {1'b1, a[22:0]};
            en = b[30:23]; mn = {1'b1, b[22:0]};
            if ((ex < en) || ((ex == en) && (mx < mn))) begin
                te = ex; ex = en; en = te;
                tm = mx; mx = mn; mn = tm;
                ts = sx; sx = sn; sn = ts;
            end
            d = ex - en;
            mx_s   = {10'd0, mx} << 9;
            sticky = 1'b0;
            if (d == 8'd0) begin
                mn_s = {10'd0, mn} << 9;
            end else if (d >= 8'd34) begin
                mn_s   = 34'd0;
                sticky = 1'b1;
            end else begin
                mn_s   = ({10'd0, mn} << 9) >> d;
                sticky = |(({10'd0, mn} << 9) & ((34'd1 << d) - 34'd1));
            end
            acc = (sx == sn) ? (mx_s + mn_s) : (mx_s - mn_s);
            if (acc == 34'd0) begin
                sf_add = {sx, 31'd0};
            end else begin
                if (acc[33]) begin
                    sticky = sticky | acc[0];
                    acc    = acc >> 1;
                    ers    = {2'b00, ex} + 11'sd1;
                end else begin
                    lz = 6'd0;
                    for (j = 0; j < 32; j = j + 1)
                        if (!acc[32]) begin
                            acc = acc << 1;
                            lz  = lz + 6'd1;
                        end
                    ers = {2'b00, ex} - {5'd0, lz};
                end
                frac  = {1'b0, acc[31:9]};
                guard = acc[8];
                // addition: the truncated tail is a true remainder (round up
                //           when anything was dropped)
                // subtraction: the truncation made acc larger than the exact
                //           value, so an all-zero tail means "exactly at .5"
                if (sx == sn)
                    up = guard & (sticky | (|acc[7:0]) | frac[0]);
                else
                    up = guard & ((|acc[7:0]) | (~sticky & frac[0]));
                frac = frac + {23'd0, up};
                if (frac[23]) begin
                    frac = 24'd0;
                    ers  = ers + 11'sd1;
                end
                if (ers >= 11'sd255)     sf_add = {sx, 8'hFF, 23'd0};
                else if (ers <= 11'sd0)  sf_add = {sx, 31'd0};
                else                     sf_add = {sx, ers[7:0], frac[22:0]};
            end
        end
    end
endfunction

//-----------------------------------------------------------------------------
// fp32 multiply, RNE, FTZ
//-----------------------------------------------------------------------------
function [31:0] sf_mul;
    input [31:0] a;
    input [31:0] b;
    reg [47:0] p;
    reg [23:0] ma, mb, frac;
    reg        guard, st, up, sr;
    reg signed [10:0] es;
    begin
        sr = a[31] ^ b[31];
        if (sf_is_nan(a) || sf_is_nan(b)) begin
            sf_mul = SF_QNAN;
        end else if ((sf_is_inf(a) && sf_is_zero(b)) ||
                     (sf_is_inf(b) && sf_is_zero(a))) begin
            sf_mul = SF_QNAN;
        end else if (sf_is_inf(a) || sf_is_inf(b)) begin
            sf_mul = {sr, 8'hFF, 23'd0};
        end else if (sf_is_zero(a) || sf_is_zero(b)) begin
            sf_mul = {sr, 31'd0};
        end else if (sf_is_sub(a) || sf_is_sub(b)) begin
            sf_mul = {sr, 31'd0};                       // FTZ
        end else begin
            ma = {1'b1, a[22:0]};
            mb = {1'b1, b[22:0]};
            p  = ma * mb;
            if (p[47]) begin
                frac  = {1'b0, p[46:24]};
                guard = p[23];
                st    = |p[22:0];
                es    = $signed({1'b0, a[30:23]}) +
                        $signed({1'b0, b[30:23]}) - 11'sd126;
            end else begin
                frac  = {1'b0, p[45:23]};
                guard = p[22];
                st    = |p[21:0];
                es    = $signed({1'b0, a[30:23]}) +
                        $signed({1'b0, b[30:23]}) - 11'sd127;
            end
            up   = guard & (st | frac[0]);
            frac = frac + {23'd0, up};
            if (frac[23]) begin
                frac = 24'd0;
                es   = es + 11'sd1;
            end
            if (es >= 11'sd255)     sf_mul = {sr, 8'hFF, 23'd0};
            else if (es <= 11'sd0)  sf_mul = {sr, 31'd0};
            else                    sf_mul = {sr, es[7:0], frac[22:0]};
        end
    end
endfunction

//-----------------------------------------------------------------------------
// conversions
//-----------------------------------------------------------------------------
// |x| < 2^24 -> signed integer, round to nearest even
function [31:0] sf_irne;
    input [31:0] x;
    reg signed [10:0] es;
    reg [23:0] m;
    reg [4:0]  sh;
    reg        g, st;
    reg [31:0] mag;
    begin
        es  = $signed({1'b0, x[30:23]}) - 11'sd127;
        m   = {1'b1, x[22:0]};
        mag = 32'd0;
        if (!(sf_is_nan(x) || sf_is_inf(x) || sf_is_zero(x) || sf_is_sub(x))) begin
            if (es >= 11'sd23) begin
                mag = {8'd0, m} << (es[4:0] - 5'd23);
            end else if (es >= 11'sd0) begin
                sh  = 5'd23 - es[4:0];
                g   = (m >> (sh - 5'd1)) & 1'b1;
                st  = |(m & ((24'd1 << (sh - 5'd1)) - 24'd1));
                mag = m >> sh;
                if (g && (st || mag[0])) mag = mag + 32'd1;
            end else if (es == -11'sd1) begin
                // x in [0.5,1): 0.5 itself rounds to 0 (even), anything above to 1
                mag = (m > 24'h80_0000) ? 32'd1 : 32'd0;
            end
        end
        sf_irne = x[31] ? (~mag + 32'd1) : mag;
    end
endfunction

// signed integer -> fp32, exact for |i| < 2^24
function [31:0] sf_fint;
    input [31:0] i;
    reg [31:0] u;
    reg [4:0]  e;
    reg [23:0] m;
    integer    j;
    begin
        if (i == 32'd0) begin
            sf_fint = 32'd0;
        end else begin
            u = i[31] ? (~i + 32'd1) : i;
            e = 5'd0;
            for (j = 0; j < 32; j = j + 1)
                if (u[j]) e = j[4:0];
            if (e >= 5'd23) m = u >> (e - 5'd23);
            else            m = u << (5'd23 - e);
            sf_fint = {i[31], (e + 8'd127), m[22:0]};
        end
    end
endfunction

// a * 2^k with clamping; k is a signed integer
function [31:0] sf_ldexp;
    input [31:0] a;
    input integer k;
    reg signed [12:0] e;
    begin
        if (sf_is_zero(a) || sf_is_inf(a) || sf_is_nan(a)) begin
            sf_ldexp = a;
        end else begin
            e = $signed({1'b0, a[30:23]}) + k;
            if (e >= 13'sd255)     sf_ldexp = {a[31], 8'hFF, 23'd0};
            else if (e <= 13'sd0)  sf_ldexp = {a[31], 31'd0};
            else                   sf_ldexp = {a[31], e[7:0], a[22:0]};
        end
    end
endfunction

// 2^k - 1, correctly rounded for |k| <= 24 (used by the expm1 correction)
function [31:0] sf_pow2m1;
    input integer k;
    integer    m;
    reg [23:0] f24;
    begin
        if (k == 0) begin
            sf_pow2m1 = 32'd0;
        end else if (k > 0) begin
            sf_pow2m1 = sf_fint((32'd1 << k) - 32'd1);
        end else begin
            m = -k;
            if (m >= 25) begin
                sf_pow2m1 = 32'hBF80_0000;
            end else begin
                f24 = 24'h80_0000 - (24'd1 << (24 - m));
                sf_pow2m1 = 32'h8000_0000 | (32'd126 << 23) | {9'd0, f24[22:0]};
            end
        end
    end
endfunction

// storage format -> fp32 (fmt_in: 0 FP32, 1 BF16, 2 FP16; 16-bit raw in [15:0])
function [31:0] sf_widen;
    input [31:0] raw;
    input [1:0]  fmt;
    reg [15:0] h;
    reg [9:0]  f10;
    reg [7:0]  e;
    begin
        h = raw[15:0];
        if (fmt == 2'd0) begin
            sf_widen = raw;
        end else if (fmt == 2'd2) begin
            e   = {3'd0, h[14:10]};
            f10 = h[9:0];
            if (e == 8'd0) begin
                sf_widen = {h[15], 31'd0};                     // FTZ
            end else if (e == 8'h1F) begin
                sf_widen = (f10 != 10'd0) ? SF_QNAN : {h[15], 8'hFF, 23'd0};
            end else begin
                sf_widen = {h[15], e + 8'd112, f10, 13'd0};
            end
        end else begin
            e = h[14:7];
            if (e == 8'd0) begin
                sf_widen = {h[15], 31'd0};                     // FTZ
            end else if (e == 8'hFF) begin
                sf_widen = (h[6:0] != 7'd0) ? SF_QNAN : {h[15], 8'hFF, 23'd0};
            end else begin
                sf_widen = {h[15], e, h[6:0], 16'd0};
            end
        end
    end
endfunction

// fp32 -> storage format, RNE (fmt_out: 0 FP32, 1 BF16, 2 FP16)
function [31:0] sf_narrow;
    input [31:0] x;
    input [1:0]  fmt;
    reg [15:0] h;
    reg [9:0]  f10;
    reg [23:0] m;
    reg [10:0] f11;
    reg        g, st, up;
    reg signed [10:0] es, e16;
    reg [5:0]  sh;
    begin
        if (fmt == 2'd0) begin
            sf_narrow = x;
        end else if (sf_is_nan(x)) begin
            sf_narrow = (fmt == 2'd2) ? {16'd0, 16'h7E00} : {16'd0, 16'h7FC0};
        end else if (sf_is_inf(x)) begin
            sf_narrow = (fmt == 2'd2) ? {16'd0, x[31], 15'h7C00}
                                      : {16'd0, x[31], 15'h7F80};
        end else if (sf_is_zero(x) || sf_is_sub(x)) begin
            sf_narrow = {16'd0, x[31], 15'd0};                 // FTZ
        end else if (fmt == 2'd1) begin
            // BF16: round the 16 dropped bits; a carry into the exponent
            // produces Inf naturally
            g  = x[15];
            st = |x[14:0];
            h  = x[31:16] + {15'd0, (g & (st | x[16]))};
            sf_narrow = {16'd0, h};
        end else begin
            es = $signed({1'b0, x[30:23]}) - 11'sd127;
            m  = {1'b1, x[22:0]};
            if (es > 11'sd15) begin
                sf_narrow = {16'd0, x[31], 15'h7C00};           // overflow -> Inf
            end else if (es >= -11'sd14) begin
                f10 = x[22:13];
                g   = x[12];
                st  = |x[11:0];
                up  = g & (st | f10[0]);
                f11 = {1'b0, f10} + {10'd0, up};    // 11 bits: keep the carry
                e16 = es + 11'sd15;
                if (f11[10]) begin                  // rounding carried out
                    f11 = 11'd0;
                    e16 = e16 + 11'sd1;
                end
                h   = {x[31], e16[4:0], f11[9:0]};
                sf_narrow = {16'd0, h};
            end else begin
                // subnormal: k = round(m * 2^(es+1))
                sh = 6'd0 - es[5:0] - 6'd1;
                if (sh > 6'd24) begin
                    sf_narrow = {16'd0, x[31], 15'd0};
                end else begin
                    up = 1'b0;
                    if (sh != 6'd0) begin
                        g  = (m >> (sh[4:0] - 5'd1)) & 1'b1;
                        st = (sh >= 6'd2) ?
                             (|(m & ((24'd1 << (sh[4:0] - 5'd1)) - 24'd1))) : 1'b0;
                        up = g & (st | ((m >> sh[4:0]) & 1'b1));
                    end
                    f10 = (m >> sh[4:0]) + up;
                    h   = {x[31], 5'd0, f10};
                    sf_narrow = {16'd0, h};
                end
            end
        end
    end
endfunction

`endif

`default_nettype none
// FP32 add with round-to-nearest-even, 3-stage pipeline.
//   stage 1: unpack / swap / align / sticky
//   stage 2: add-sub / overflow fix / leading-one detect
//   stage 3: normalize / round / pack
// Latency = 3 cycles. Normal numbers and +/-0 only; denormals are treated as
// normals, NaN/Inf inputs are unsupported. Result exponent <= 0 flushes to +0,
// exponent >= 255 saturates to Inf.
module fp32_add (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        valid_i,
    input  wire [31:0] a,
    input  wire [31:0] b,
    output reg         valid_o,
    output reg  [31:0] y
);
    // ------------------------------------------------------------------
    // stage 1: unpack, select the larger operand, align the smaller one
    // ------------------------------------------------------------------
    reg        u_sa, u_sb;
    reg [7:0]  u_ea, u_eb;
    reg [22:0] u_fa, u_fb;
    reg [23:0] u_ma, u_mb;
    reg [23:0] u_m_large, u_m_small;
    reg        u_s_large, u_s_small;
    reg [7:0]  u_e_large, u_e_small, u_diff;
    // 28 bits: one carry bit above the 24-bit mantissa plus 3 guard bits,
    // so the operand width matches the 28-bit sum (no implicit zero-extend).
    reg [27:0] u_m_ext_large, u_m_ext_small, u_m_shift;
    reg        u_sticky;
    reg        u_a_zero, u_b_zero;
    reg [1:0]  u_zero_sel;          // 0 = normal, 1 = result is b, 2 = result is a
    reg [31:0] u_pass;
    integer i;

    always @* begin
        u_sa = a[31]; u_ea = a[30:23]; u_fa = a[22:0];
        u_sb = b[31]; u_eb = b[30:23]; u_fb = b[22:0];
        u_ma = {1'b1, u_fa};
        u_mb = {1'b1, u_fb};
        u_a_zero = (u_ea == 8'd0) && (u_fa == 23'd0);
        u_b_zero = (u_eb == 8'd0) && (u_fb == 23'd0);

        // The +/-0 shortcut returns the other operand verbatim. a_zero has
        // priority, matching the previous combinational implementation.
        u_zero_sel = 2'd0;
        u_pass     = 32'd0;
        if (u_a_zero) begin
            u_zero_sel = 2'd1;
            u_pass     = b;
        end else if (u_b_zero) begin
            u_zero_sel = 2'd2;
            u_pass     = a;
        end

        // Select larger exponent (and larger mantissa on a tie).
        if (u_ea > u_eb) begin
            u_e_large = u_ea; u_e_small = u_eb;
            u_m_large = u_ma; u_m_small = u_mb;
            u_s_large = u_sa; u_s_small = u_sb;
        end else if (u_eb > u_ea) begin
            u_e_large = u_eb; u_e_small = u_ea;
            u_m_large = u_mb; u_m_small = u_ma;
            u_s_large = u_sb; u_s_small = u_sa;
        end else begin
            u_e_large = u_ea; u_e_small = u_eb;
            if (u_ma >= u_mb) begin
                u_m_large = u_ma; u_m_small = u_mb;
                u_s_large = u_sa; u_s_small = u_sb;
            end else begin
                u_m_large = u_mb; u_m_small = u_ma;
                u_s_large = u_sb; u_s_small = u_sa;
            end
        end

        u_diff        = u_e_large - u_e_small;
        u_m_ext_large = {1'b0, u_m_large, 3'b000};
        u_m_ext_small = {1'b0, u_m_small, 3'b000};
        u_m_shift     = u_m_ext_small >> u_diff;

        u_sticky = 1'b0;
        for (i = 0; i < 28; i = i + 1)
            if (i < u_diff) u_sticky = u_sticky | u_m_ext_small[i];
    end

    reg        s1_valid;
    reg        s1_sign;
    reg [7:0]  s1_exp;
    reg [27:0] s1_m_large, s1_m_shift;
    reg        s1_sticky, s1_same;
    reg [1:0]  s1_zero_sel;
    reg [31:0] s1_pass;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s1_valid    <= 1'b0;
            s1_sign     <= 1'b0;
            s1_exp      <= 8'd0;
            s1_m_large  <= 28'd0;
            s1_m_shift  <= 28'd0;
            s1_sticky   <= 1'b0;
            s1_same     <= 1'b0;
            s1_zero_sel <= 2'd0;
            s1_pass     <= 32'd0;
        end else begin
            s1_valid    <= valid_i;
            s1_sign     <= u_s_large;
            s1_exp      <= u_e_large;
            s1_m_large  <= u_m_ext_large;
            s1_m_shift  <= u_m_shift;
            s1_sticky   <= u_sticky;
            s1_same     <= (u_s_large == u_s_small);
            s1_zero_sel <= u_zero_sel;
            s1_pass     <= u_pass;
        end
    end

    // ------------------------------------------------------------------
    // stage 2: add or subtract, fix the carry, find the leading one
    // ------------------------------------------------------------------
    reg [27:0] v_b, v_sum;
    reg        v_cin;
    reg signed [15:0] v_exp;
    reg        v_sticky;
    reg        v_zero;
    reg [4:0]  v_lshift;

    // 28-bit carry-select adder: four 7-bit groups, each high group computes
    // both carry-in variants in parallel. This keeps the carry chain off the
    // critical path (was ~55 levels with a plain 28-bit add).
    // r0..r2 keep their carry out; the top group's carry (bit 28) cannot be
    // set because both operands have bit 27 clear, so r3 is 7 bits only.
    reg [7:0] r0, r1a, r1b, r1, r2a, r2b, r2;
    reg [6:0] r3a, r3b, r3;
    reg       c1, c2, c3;

    always @* begin
        // A + addend + cin covers add, subtract and subtract-with-borrow.
        if (s1_same) begin
            v_b   = s1_m_shift;
            v_cin = 1'b0;
        end else if (s1_sticky) begin
            v_b   = ~s1_m_shift;      // A - B - 1
            v_cin = 1'b0;
        end else begin
            v_b   = ~s1_m_shift;      // A - B
            v_cin = 1'b1;
        end

        r0  = {1'b0, s1_m_large[6:0]}   + {1'b0, v_b[6:0]}   + {7'd0, v_cin};
        c1  = r0[7];

        r1a = {1'b0, s1_m_large[13:7]}  + {1'b0, v_b[13:7]};
        r1b = r1a + 8'd1;
        r1  = c1 ? r1b : r1a;
        c2  = r1[7];

        r2a = {1'b0, s1_m_large[20:14]} + {1'b0, v_b[20:14]};
        r2b = r2a + 8'd1;
        r2  = c2 ? r2b : r2a;
        c3  = r2[7];

        r3a = s1_m_large[27:21] + v_b[27:21];
        r3b = r3a + 7'd1;
        r3  = c3 ? r3b : r3a;

        v_sum    = {r3[6:0], r2[6:0], r1[6:0], r0[6:0]};
        v_sticky = s1_sticky;
        v_exp    = {8'b0, s1_exp};

        // Carry out of the 28-bit sum: shift right one and bump exponent.
        if (v_sum[27]) begin
            v_sticky = v_sticky | v_sum[0];
            v_sum    = {1'b0, v_sum[27:1]};
            v_exp    = v_exp + 16'sd1;
        end

        v_zero = (v_sum == 28'd0);

        // Normalization shift (26 - msb), selected directly by a 5-level
        // binary search. Assigned as a constant per leaf so no 5-bit
        // subtract is needed after the leading-one search. All range-OR
        // terms are computed in parallel from v_sum, keeping the depth
        // near 10 instead of one level per bit. If v_sum is zero the value
        // is don't-care because stage 3 forces a zero result.
        v_lshift = 5'd0;
        if (|v_sum[26:13]) begin
            if (|v_sum[26:20]) begin
                if (|v_sum[26:23]) begin
                    if (|v_sum[26:25])
                        v_lshift = v_sum[26] ? 5'd0 : 5'd1;
                    else
                        v_lshift = v_sum[24] ? 5'd2 : 5'd3;
                end else begin
                    if (|v_sum[22:21])
                        v_lshift = v_sum[22] ? 5'd4 : 5'd5;
                    else
                        v_lshift = v_sum[20] ? 5'd6 : 5'd7;
                end
            end else begin
                if (|v_sum[19:16]) begin
                    if (|v_sum[19:18])
                        v_lshift = v_sum[19] ? 5'd7 : 5'd8;
                    else
                        v_lshift = v_sum[17] ? 5'd9 : 5'd10;
                end else begin
                    if (|v_sum[15:14])
                        v_lshift = v_sum[15] ? 5'd11 : 5'd12;
                    else
                        v_lshift = 5'd13;
                end
            end
        end else begin
            if (|v_sum[12:6]) begin
                if (|v_sum[12:9]) begin
                    if (|v_sum[12:11])
                        v_lshift = v_sum[12] ? 5'd14 : 5'd15;
                    else
                        v_lshift = v_sum[10] ? 5'd16 : 5'd17;
                end else begin
                    if (|v_sum[8:7])
                        v_lshift = v_sum[8] ? 5'd18 : 5'd19;
                    else
                        v_lshift = 5'd20;
                end
            end else begin
                if (|v_sum[5:3]) begin
                    if (|v_sum[5:4])
                        v_lshift = v_sum[5] ? 5'd21 : 5'd22;
                    else
                        v_lshift = 5'd23;
                end else begin
                    if (|v_sum[2:1])
                        v_lshift = v_sum[2] ? 5'd24 : 5'd25;
                    else
                        v_lshift = 5'd26;
                end
            end
        end
    end

    reg        s2_valid, s2_sign, s2_zero, s2_sticky;
    reg signed [15:0] s2_exp;
    reg [26:0] s2_sum;          // bit 27 is cleared by the overflow fix
    reg [4:0]  s2_lshift;
    reg [1:0]  s2_zero_sel;
    reg [31:0] s2_pass;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s2_valid    <= 1'b0;
            s2_sign     <= 1'b0;
            s2_zero     <= 1'b1;
            s2_sticky   <= 1'b0;
            s2_exp      <= 16'sd0;
            s2_sum      <= 27'd0;
            s2_lshift   <= 5'd0;
            s2_zero_sel <= 2'd0;
            s2_pass     <= 32'd0;
        end else begin
            s2_valid    <= s1_valid;
            s2_sign     <= s1_sign;
            s2_zero     <= v_zero;
            s2_sticky   <= v_sticky;
            s2_exp      <= v_exp;
            s2_sum      <= v_sum[26:0];
            s2_lshift   <= v_lshift;
            s2_zero_sel <= s1_zero_sel;
            s2_pass     <= s1_pass;
        end
    end

    // ------------------------------------------------------------------
    // stage 3: normalize, round to nearest even, pack
    // ------------------------------------------------------------------
    reg [26:0] w_norm;          // bit 27 is always zero after normalization
    reg [23:0] w_mant;
    reg        w_guard, w_round, w_sticky, w_up;
    reg [24:0] w_mant_r;
    reg signed [15:0] w_exp;
    reg [31:0] w_y;

    always @* begin
        w_norm   = s2_sum << s2_lshift;
        w_mant   = w_norm[26:3];
        w_guard  = w_norm[2];
        w_round  = w_norm[1];
        w_sticky = s2_sticky | w_norm[0];
        w_up     = w_guard && (w_round || w_sticky || w_mant[0]);

        w_mant_r = {1'b0, w_mant} + {24'b0, w_up};
        w_exp    = s2_exp - {11'b0, s2_lshift};

        if (w_mant_r[24]) begin
            w_mant_r = w_mant_r >> 1;
            w_exp    = w_exp + 16'sd1;
        end

        if (s2_zero_sel != 2'd0)
            w_y = s2_pass;
        else if (s2_zero)
            w_y = 32'h0000_0000;
        else if (w_exp <= 16'sd0)
            w_y = 32'h0000_0000;
        else if (w_exp >= 16'sd255)
            w_y = {s2_sign, 8'hFF, 23'h000000};
        else
            w_y = {s2_sign, w_exp[7:0], w_mant_r[22:0]};
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_o <= 1'b0;
            y       <= 32'd0;
        end else begin
            valid_o <= s2_valid;
            y       <= w_y;
        end
    end
endmodule
`default_nettype wire

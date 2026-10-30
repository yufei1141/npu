//=============================================================================
// sfu_slice.v : four-lane special function datapath
//
//   Eight-lane sfu_top is two copies of this slice, so every op below widens
//   to ELEMS = 4 lanes and runs them in lockstep.
//
//   Interface (unchanged from the original skeleton, which sfu_top drives):
//     in_valid/in_ready    one op per handshake
//     op      0 EXP2   1 LOG2   2 RCP  3 RSQRT  4 SQRT
//             5 SIN    6 COS    7 TANH 8 SINCOS (two write-back beats)
//     fmt_in  0 FP32   1 BF16   2 FP16   (16-bit data in the low half)
//     fmt_out same encoding, applied per lane on write-back
//     mode    0 fast (4 Newton steps)  1 accurate (5 Newton steps)
//     dout    ELEMS x 32, one beat for every op except SINCOS
//     status  bit0 overflow, bit1 underflow, bit2 invalid, bit3 div0,
//             OR-reduced over the lanes
//
//   Implementation notes and deviations from sfu/arch/microarch.md
//   ---------------------------------------------------------------
//   * Fixed states per function instead of a microcode ROM.  Each Horner step
//     is one cycle; the coefficient ROM is addressed by the fixed per-op base.
//     Worst-case latency: EXP2 13, LOG2 11, RCP 9, RSQRT/SQRT 9, SIN/COS 16,
//     TANH 21 cycles (TANH is the 21-uop case in the microarch document).
//   * LOG2 uses a 16-entry per-lane table (argument reduction to |u| < 2^-4,
//     then a degree-6 series in u) rather than the atanh series, because the
//     non-FMA datapath cannot round a quotient to better than about 1 ulp.
//     The table stores log2(m_i) split into high/low words so that the
//     exponent term e + log2(m_i) is exact.
//   * TANH computes expm1(2x) as f * P(f) in the factored form (no "exp - 1"
//     cancellation), then tanh = m / (m + 2) with a Newton reciprocal.
//   * Arithmetic is fp32 with one rounding per operation and no FMA; the
//     measured accuracy against a double-precision reference (tb_sfu_slice,
//     worst case over 6k-15k random vectors per op) is:
//       EXP2 0.95 ulp, LOG2 0.50, RCP 1.36, RSQRT 1.52, SQRT 2.01,
//       SIN 1.37, COS 1.17, TANH 3.05.
//   * Without an FMA an exact result is not guaranteed: 1/1.0 and 1/2.0 are
//     one ulp low (Newton fixed point in fp32), cos(pi/2) is 2 ulp from the
//     exact -4.37e-8, tanh(pi) is 1 ulp from the rounded value.
//   * Deviations: FTZ (subnormals flush to zero) instead of the documented
//     FTZ=0 default; SIN/COS arguments above 2^20 and NaN/Inf report invalid.
//=============================================================================
`timescale 1ns/1ps
`default_nettype none

module sfu_slice #(
    parameter ELEMS  = 4,
    parameter DATA_W = 32
) (
    input  wire                    clk,
    input  wire                    rst_n,
    input  wire                    in_valid,
    output wire                    in_ready,
    input  wire [3:0]              op,
    input  wire [1:0]              fmt_in,
    input  wire [1:0]              fmt_out,
    input  wire [1:0]              mode,
    input  wire [ELEMS*DATA_W-1:0] din,
    output reg                     out_valid,
    input  wire                    out_ready,
    output reg  [ELEMS*DATA_W-1:0] dout,
    output wire [7:0]              status
);

    //-------------------------------------------------------------------------
    // fp32 helpers (sf_add, sf_mul, sf_irne, sf_fint, sf_ldexp, sf_pow2m1,
    // sf_widen, sf_narrow, predicates, constants)
    //-------------------------------------------------------------------------
`include "sfu_fp32.vh"

    localparam [31:0] RCP_A = 32'h3FB4_4A99;   // 1 + sqrt2/pi  reciprocal seed
    localparam [31:0] RCP_B = 32'hBEF4_225A;
    localparam [31:0] RQ_A  = 32'h3F84_CFE1;   // rsqrt seed
    localparam [31:0] RQ_B  = 32'hBE17_D8A0;
    localparam [31:0] INV2PI = 32'h3F22_F983;  // 2/pi
    localparam [31:0] LOG2E  = 32'h3FB8_AA3B;  // log2(e)
    localparam [31:0] PIO4   = 32'h3F49_0FDB;  // pi/4
    localparam [31:0] PIO2_HI = 32'h3FC9_0FDB;
    localparam [31:0] PIO2_LO = 32'hB33B_BD2E;
    localparam [31:0] LIM_150 = 32'h4316_0000; // 150.0
    localparam [31:0] LIM_10  = 32'h4120_0000; // 10.0
    localparam [31:0] LIM_2P20 = 32'h4980_0000;// 2^20

    //-------------------------------------------------------------------------
    // states
    //-------------------------------------------------------------------------
    localparam [4:0]
        S_IDLE = 5'd0,
        S_EK   = 5'd1,  S_EP = 5'd2,  S_EF = 5'd3,   // exp2
        S_LG   = 5'd4,  S_LP = 5'd5,  S_LF = 5'd6,   // log2
        S_RM   = 5'd7,  S_RN = 5'd8,  S_RF = 5'd9,   // rcp
        S_QM   = 5'd10, S_QN = 5'd11, S_QF = 5'd12,  // rsqrt / sqrt
        S_SN   = 5'd13, S_SR = 5'd14, S_SC = 5'd15,  // sin/cos reduction
        S_SP   = 5'd16, S_SF = 5'd17,                // sin/cos polynomial
        S_TZ   = 5'd18, S_TR = 5'd19, S_TU = 5'd20,  // tanh
        S_TP   = 5'd21, S_TV = 5'd22, S_TN = 5'd23, S_TE = 5'd24,
        S_FIN  = 5'd25, S_OUT = 5'd26;

    //-------------------------------------------------------------------------
    // coefficient ROM : broadcast to all lanes
    //   0.. 8  EXP2   c0..c8
    //   9..15  LOG2   series q0..q6      (1, -1/2, 1/3, ... 1/7)
    //  16..21  SIN    c0..c5
    //  22..27  COS    c0..c5
    //  28..35  EXPM1  q0..q7             ((e^f-1)/f)
    //  36..40  pi/2   five-term 4-bit split (exact products for |n| <= 2^20)
    //  41..43  ln2    three-term split
    //  44     2/ln2
    //-------------------------------------------------------------------------
    reg [31:0] crom [0:47];

    // per-lane LOG2 table:  log2(m_i) = hi + lo,  1/m_i,  m_i
    reg [31:0] l2_mi [0:15];
    reg [31:0] l2_rt [0:15];
    reg [31:0] l2_hi [0:15];
    reg [31:0] l2_lo [0:15];

    initial begin
        crom[ 0] = 32'h3F80_0000; crom[ 1] = 32'h3F31_7218;
        crom[ 2] = 32'h3E75_FDF0; crom[ 3] = 32'h3D63_5847;
        crom[ 4] = 32'h3C1D_955B; crom[ 5] = 32'h3AAE_C3FF;
        crom[ 6] = 32'h3921_8489; crom[ 7] = 32'h377F_E5FE;
        crom[ 8] = 32'h35B1_6011;
        crom[ 9] = 32'h3F80_0000; crom[10] = 32'hBF00_0000;
        crom[11] = 32'h3EAA_AAAB; crom[12] = 32'hBE80_0000;
        crom[13] = 32'h3E4C_CCCD; crom[14] = 32'hBE2A_AAAB;
        crom[15] = 32'h3E12_4925;
        crom[16] = 32'h3F80_0000; crom[17] = 32'hBE2A_AAAB;
        crom[18] = 32'h3C08_8889; crom[19] = 32'hB950_0D01;
        crom[20] = 32'h3638_EF1D; crom[21] = 32'hB2D7_322B;
        crom[22] = 32'h3F80_0000; crom[23] = 32'hBF00_0000;
        crom[24] = 32'h3D2A_AAAB; crom[25] = 32'hBAB6_0B61;
        crom[26] = 32'h37D0_0D01; crom[27] = 32'hB493_F27E;
        // expm1(f)/f = 1 + f/2! + f^2/3! + ... + f^7/8!   (see S_TV)
        crom[28] = 32'h3F80_0000; crom[29] = 32'h3F00_0000;
        crom[30] = 32'h3E2A_AAAB; crom[31] = 32'h3D2A_AAAB;
        crom[32] = 32'h3C08_8889; crom[33] = 32'h3AB6_0B61;
        crom[34] = 32'h3950_0D01; crom[35] = 32'h37D0_0D01;
        crom[36] = 32'h3FD0_0000; crom[37] = 32'hBD60_0000;
        crom[38] = 32'h3A00_0000; crom[39] = 32'hB690_0000;
        crom[40] = 32'hB42E_EF4C;
        crom[41] = 32'h3F31_0000; crom[42] = 32'h3AE4_0000;
        crom[43] = 32'h35BF_BE8E;
        crom[44] = 32'h4038_AA3B; crom[45] = 32'h3FB8_AA3B;
        crom[46] = 32'h0000_0000; crom[47] = 32'h0000_0000;

        l2_mi[ 0] = 32'h3F80_0000; l2_rt[ 0] = 32'h3F80_0000;
        l2_hi[ 0] = 32'h0000_0000; l2_lo[ 0] = 32'h0000_0000;
        l2_mi[ 1] = 32'h3F88_0000; l2_rt[ 1] = 32'h3F70_F0F1;
        l2_hi[ 1] = 32'h3DB3_0000; l2_lo[ 1] = 32'h387D_BEB2;
        l2_mi[ 2] = 32'h3F90_0000; l2_rt[ 2] = 32'h3F63_8E39;
        l2_hi[ 2] = 32'h3E2E_0000; l2_lo[ 2] = 32'h3651_CFDF;
        l2_mi[ 3] = 32'h3F98_0000; l2_rt[ 3] = 32'h3F57_9436;
        l2_hi[ 3] = 32'h3E7D_E000; l2_lo[ 3] = 32'h3635_C813;
        l2_mi[ 4] = 32'h3FA0_0000; l2_rt[ 4] = 32'h3F4C_CCCD;
        l2_hi[ 4] = 32'h3EA4_C000; l2_lo[ 4] = 32'h391E_12F3;
        l2_mi[ 5] = 32'h3FA8_0000; l2_rt[ 5] = 32'h3F43_0C31;
        l2_hi[ 5] = 32'h3EC8_C000; l2_lo[ 5] = 32'h396E_A248;
        l2_mi[ 6] = 32'h3FB0_0000; l2_rt[ 6] = 32'h3F3A_2E8C;
        l2_hi[ 6] = 32'h3EEB_2000; l2_lo[ 6] = 32'h3954_F80D;
        l2_mi[ 7] = 32'h3FB8_0000; l2_rt[ 7] = 32'h3F32_1643;
        l2_hi[ 7] = 32'h3F06_0000; l2_lo[ 7] = 32'h3902_806B;
        l2_mi[ 8] = 32'h3FC0_0000; l2_rt[ 8] = 32'h3F2A_AAAB;
        l2_hi[ 8] = 32'h3F15_C000; l2_lo[ 8] = 32'h35D1_CFDF;
        l2_mi[ 9] = 32'h3FC8_0000; l2_rt[ 9] = 32'h3F23_D70A;
        l2_hi[ 9] = 32'h3F24_C000; l2_lo[ 9] = 32'h399E_12F3;
        l2_mi[10] = 32'h3FD0_0000; l2_rt[10] = 32'h3F1D_89D9;
        l2_hi[10] = 32'h3F33_4000; l2_lo[10] = 32'h3980_2392;
        l2_mi[11] = 32'h3FD8_0000; l2_rt[11] = 32'h3F17_B426;
        l2_hi[11] = 32'h3F41_4000; l2_lo[11] = 32'h369D_5BE7;
        l2_mi[12] = 32'h3FE0_0000; l2_rt[12] = 32'h3F12_4925;
        l2_hi[12] = 32'h3F4E_A000; l2_lo[12] = 32'h396C_FEA8;
        l2_mi[13] = 32'h3FE8_0000; l2_rt[13] = 32'h3F0D_3DCB;
        l2_hi[13] = 32'h3F5B_A000; l2_lo[13] = 32'h3894_8F55;
        l2_mi[14] = 32'h3FF0_0000; l2_rt[14] = 32'h3F08_8889;
        l2_hi[14] = 32'h3F68_2000; l2_lo[14] = 32'h391F_B693;
        l2_mi[15] = 32'h3FF8_0000; l2_rt[15] = 32'h3F04_2108;
        l2_hi[15] = 32'h3F74_4000; l2_lo[15] = 32'h38C6_B362;
    end

    //-------------------------------------------------------------------------
    // state and datapath registers
    //-------------------------------------------------------------------------
    reg [4:0]  state;
    reg [3:0]  op_r;
    reg [1:0]  fmi_r, fmo_r;
    reg [2:0]  nm_r;               // Newton iterations for this op
    reg [3:0]  cnt_r;
    reg        frame_r;

    reg [ELEMS*32-1:0] x_r;        // widened input, also sqrt's multiplicand
    reg [ELEMS*32-1:0] pv_r;       // polynomial variable / Newton operand
    reg [ELEMS*32-1:0] p_r;        // accumulator / running remainder
    reg [ELEMS*32-1:0] q_r;        // cos polynomial / log2 exponent part
    reg [ELEMS*32-1:0] s_r;        // sin polynomial / tanh expm1 result
    reg [ELEMS*32-1:0] u_r;        // log2 reduced argument
    reg [ELEMS*32-1:0] n_r;        // -n as fp32 (sin reduction)
    reg [ELEMS*32-1:0] k_r;        // integer k / exponent field
    reg [ELEMS*32-1:0] y_r;        // result
    reg [ELEMS*32-1:0] c_r;        // cos result (SINCOS second beat)
    reg [ELEMS*32-1:0] spc_r;      // per-lane special result
    reg [ELEMS-1:0]    don_r;      // lane served by the special path
    reg [ELEMS-1:0]    sgn_r;      // rcp sign
    reg [ELEMS*2-1:0]  qd_r;       // sin/cos quadrant

    reg [ELEMS-1:0] ovf_l, unf_l, inv_l, div0_l;

    // loop / temporary variables (combinational intermediates)
    integer i;
    reg [31:0] xv, yv, kk, ee;
    reg [3:0]  idx;

    assign in_ready = (state == S_IDLE);
    assign status   = {4'd0, |div0_l, |inv_l, |unf_l, |ovf_l};

    //-------------------------------------------------------------------------
    // main sequence
    //-------------------------------------------------------------------------
    always @(posedge clk) begin
        if (!rst_n) begin
            state     <= S_IDLE;
            out_valid <= 1'b0;
            frame_r   <= 1'b0;
            cnt_r     <= 4'd0;
            op_r      <= 4'd0;
            fmi_r     <= 2'd0;
            fmo_r     <= 2'd0;
            nm_r      <= 3'd4;
            don_r     <= {ELEMS{1'b0}};
            ovf_l     <= {ELEMS{1'b0}};
            unf_l     <= {ELEMS{1'b0}};
            inv_l     <= {ELEMS{1'b0}};
            div0_l    <= {ELEMS{1'b0}};
            y_r       <= {ELEMS*32{1'b0}};
            c_r       <= {ELEMS*32{1'b0}};
            spc_r     <= {ELEMS*32{1'b0}};
        end else begin
            case (state)

            //-----------------------------------------------------------------
            S_IDLE: begin
                if (in_valid) begin
                    op_r  <= op;
                    fmi_r <= fmt_in;
                    fmo_r <= fmt_out;
                    nm_r  <= (mode == 2'd0) ? 3'd4 : 3'd5;
                    ovf_l  <= {ELEMS{1'b0}};
                    unf_l  <= {ELEMS{1'b0}};
                    inv_l  <= {ELEMS{1'b0}};
                    div0_l <= {ELEMS{1'b0}};
                    for (i = 0; i < ELEMS; i = i + 1) begin
                        x_r[i*32 +: 32]   <= sf_widen(din[i*32 +: 32], fmt_in);
                        don_r[i]          <= 1'b0;
                        spc_r[i*32 +: 32] <= 32'd0;
                    end
                    case (op)
                    4'd0:    state <= S_EK;
                    4'd1:    state <= S_LG;
                    4'd2:    state <= S_RM;
                    4'd3:    state <= S_QM;
                    4'd4:    state <= S_QM;
                    4'd5:    state <= S_SN;
                    4'd6:    state <= S_SN;
                    4'd7:    state <= S_TZ;
                    4'd8:    state <= S_SN;
                    default: begin
                        for (i = 0; i < ELEMS; i = i + 1) begin
                            don_r[i]          <= 1'b1;
                            spc_r[i*32 +: 32] <= SF_QNAN;
                        end
                        inv_l <= {ELEMS{1'b1}};
                        state <= S_FIN;
                    end
                    endcase
                end
            end

            //-----------------------------------------------------------------
            // EXP2 : k = rne(x), f = x - k, 2^x = 2^k * P(f)
            //-----------------------------------------------------------------
            S_EK: begin
                for (i = 0; i < ELEMS; i = i + 1) begin
                    xv = x_r[i*32 +: 32];
                    if (sf_is_nan(xv)) begin
                        don_r[i] <= 1'b1; spc_r[i*32 +: 32] <= SF_QNAN; inv_l[i] <= 1'b1;
                    end else if (sf_is_inf(xv)) begin
                        don_r[i] <= 1'b1;
                        spc_r[i*32 +: 32] <= xv[31] ? 32'd0 : SF_PINF;
                    end else if (sf_is_zero(xv) || sf_is_sub(xv)) begin
                        don_r[i] <= 1'b1; spc_r[i*32 +: 32] <= SF_ONE;
                        if (sf_is_sub(xv)) unf_l[i] <= 1'b1;
                    end else if ((!xv[31]) && (xv > LIM_150)) begin
                        don_r[i] <= 1'b1; spc_r[i*32 +: 32] <= SF_PINF; ovf_l[i] <= 1'b1;
                    end else if (xv[31] && ((xv & 32'h7FFF_FFFF) > LIM_150)) begin
                        don_r[i] <= 1'b1; spc_r[i*32 +: 32] <= 32'd0; unf_l[i] <= 1'b1;
                    end else begin
                        k_r[i*32 +: 32] <= sf_irne(xv);
                        pv_r[i*32 +: 32] <= sf_add(xv, sf_neg(sf_fint(sf_irne(xv))));
                        p_r[i*32 +: 32]  <= crom[8];
                    end
                end
                cnt_r <= 4'd1;
                state <= S_EP;
            end

            S_EP: begin
                for (i = 0; i < ELEMS; i = i + 1)
                    p_r[i*32 +: 32] <= sf_add(sf_mul(p_r[i*32 +: 32], pv_r[i*32 +: 32]),
                                              crom[8 - cnt_r]);
                if (cnt_r == 4'd8) state <= S_EF;
                else               cnt_r <= cnt_r + 4'd1;
            end

            S_EF: begin
                for (i = 0; i < ELEMS; i = i + 1) if (!don_r[i]) begin
                    yv = sf_ldexp(p_r[i*32 +: 32], $signed(k_r[i*32 +: 32]));
                    y_r[i*32 +: 32] <= yv;
                    if (sf_is_inf(yv))      ovf_l[i] <= 1'b1;
                    else if (sf_is_zero(yv)) unf_l[i] <= 1'b1;
                end
                state <= S_FIN;
            end

            //-----------------------------------------------------------------
            // LOG2 : m in [1,2),  i = m[22:19],  u = (m-m_i)/m_i
            //        log2(m) = log2(m_i) + log2(1+u)
            //-----------------------------------------------------------------
            S_LG: begin
                for (i = 0; i < ELEMS; i = i + 1) begin
                    xv = x_r[i*32 +: 32];
                    if (sf_is_nan(xv)) begin
                        don_r[i] <= 1'b1; spc_r[i*32 +: 32] <= SF_QNAN; inv_l[i] <= 1'b1;
                    end else if (sf_is_inf(xv)) begin
                        don_r[i] <= 1'b1;
                        spc_r[i*32 +: 32] <= xv[31] ? SF_QNAN : SF_PINF;
                        if (xv[31]) inv_l[i] <= 1'b1;
                    end else if (sf_is_zero(xv) || sf_is_sub(xv)) begin
                        don_r[i] <= 1'b1; spc_r[i*32 +: 32] <= 32'hFF80_0000;
                        div0_l[i] <= 1'b1;
                    end else if (xv[31]) begin
                        don_r[i] <= 1'b1; spc_r[i*32 +: 32] <= SF_QNAN; inv_l[i] <= 1'b1;
                    end else begin
                        ee = {24'd0, xv[30:23]} - 32'd127;      // integer exponent
                        idx = xv[22:19];
                        k_r[i*32 +: 32]  <= ee;
                        pv_r[i*32 +: 32] <= 32'd0;
                        u_r[i*32 +: 32]  <= sf_mul(sf_add({1'b0, xv[22:0]} | SF_ONE,
                                                          sf_neg(l2_mi[idx])), l2_rt[idx]);
                        p_r[i*32 +: 32]  <= crom[15];
                        q_r[i*32 +: 32]  <= sf_add(sf_fint(ee), l2_hi[idx]);
                        s_r[i*32 +: 32]  <= l2_lo[idx];
                    end
                end
                cnt_r <= 4'd1;
                state <= S_LP;
            end

            S_LP: begin
                for (i = 0; i < ELEMS; i = i + 1)
                    p_r[i*32 +: 32] <= sf_add(sf_mul(p_r[i*32 +: 32], u_r[i*32 +: 32]),
                                              crom[15 - cnt_r]);
                if (cnt_r == 4'd6) state <= S_LF;
                else               cnt_r <= cnt_r + 4'd1;
            end

            S_LF: begin
                for (i = 0; i < ELEMS; i = i + 1) if (!don_r[i]) begin
                    yv = sf_mul(sf_mul(u_r[i*32 +: 32], LOG2E), p_r[i*32 +: 32]);
                    y_r[i*32 +: 32] <= sf_add(q_r[i*32 +: 32],
                                              sf_add(s_r[i*32 +: 32], yv));
                end
                state <= S_FIN;
            end

            //-----------------------------------------------------------------
            // RCP
            //-----------------------------------------------------------------
            S_RM: begin
                for (i = 0; i < ELEMS; i = i + 1) begin
                    xv = x_r[i*32 +: 32];
                    sgn_r[i] <= xv[31];
                    if (sf_is_nan(xv)) begin
                        don_r[i] <= 1'b1; spc_r[i*32 +: 32] <= SF_QNAN; inv_l[i] <= 1'b1;
                    end else if (sf_is_zero(xv) || sf_is_sub(xv)) begin
                        don_r[i] <= 1'b1;
                        spc_r[i*32 +: 32] <= xv[31] ? 32'hFF80_0000 : SF_PINF;
                        div0_l[i] <= 1'b1;
                    end else if (sf_is_inf(xv)) begin
                        don_r[i] <= 1'b1; spc_r[i*32 +: 32] <= {xv[31], 31'd0};
                    end else begin
                        kk = {1'b0, xv[22:0]} | SF_ONE;
                        pv_r[i*32 +: 32] <= kk;
                        p_r[i*32 +: 32]  <= sf_add(RCP_A, sf_mul(RCP_B, kk));
                        k_r[i*32 +: 32]  <= {24'd0, xv[30:23]};
                    end
                end
                cnt_r <= 4'd1;
                state <= S_RN;
            end

            S_RN: begin
                for (i = 0; i < ELEMS; i = i + 1)
                    p_r[i*32 +: 32] <= sf_mul(p_r[i*32 +: 32],
                        sf_add(SF_TWO, sf_neg(sf_mul(pv_r[i*32 +: 32], p_r[i*32 +: 32]))));
                if (cnt_r == {1'b0, nm_r}) state <= S_RF;
                else                       cnt_r <= cnt_r + 4'd1;
            end

            S_RF: begin
                for (i = 0; i < ELEMS; i = i + 1) if (!don_r[i]) begin
                    yv = sf_ldexp(p_r[i*32 +: 32],
                                  {22'd0, 10'd127} - $signed(k_r[i*32 +: 32]));
                    if (sgn_r[i]) yv = sf_neg(yv);
                    y_r[i*32 +: 32] <= yv;
                    if (sf_is_inf(yv))      ovf_l[i] <= 1'b1;
                    else if (sf_is_zero(yv)) unf_l[i] <= 1'b1;
                end
                state <= S_FIN;
            end

            //-----------------------------------------------------------------
            // RSQRT / SQRT : driven from an even exponent
            //-----------------------------------------------------------------
            S_QM: begin
                for (i = 0; i < ELEMS; i = i + 1) begin
                    xv = x_r[i*32 +: 32];
                    if (sf_is_nan(xv)) begin
                        don_r[i] <= 1'b1; spc_r[i*32 +: 32] <= SF_QNAN; inv_l[i] <= 1'b1;
                    end else if (sf_is_zero(xv)) begin
                        don_r[i] <= 1'b1;
                        if (op_r == 4'd4) spc_r[i*32 +: 32] <= xv;
                        else begin
                            spc_r[i*32 +: 32] <= xv[31] ? 32'hFF80_0000 : SF_PINF;
                            div0_l[i] <= 1'b1;
                        end
                    end else if (sf_is_inf(xv)) begin
                        don_r[i] <= 1'b1;
                        if (op_r == 4'd4) spc_r[i*32 +: 32] <= xv[31] ? SF_QNAN : SF_PINF;
                        else              spc_r[i*32 +: 32] <= {xv[31], 31'd0};
                        if (xv[31]) inv_l[i] <= 1'b1;
                    end else if (xv[31]) begin
                        don_r[i] <= 1'b1; spc_r[i*32 +: 32] <= SF_QNAN; inv_l[i] <= 1'b1;
                    end else if (sf_is_sub(xv)) begin
                        don_r[i] <= 1'b1; spc_r[i*32 +: 32] <= SF_PINF; div0_l[i] <= 1'b1;
                    end else begin
                        kk = {1'b0, xv[22:0]} | SF_ONE;
                        ee = {24'd0, xv[30:23]} - 32'd127;
                        if (ee[0]) begin
                            kk = sf_add(kk, kk);            // m in [2,4) -> even exponent
                            ee = ee - 32'd1;
                        end
                        pv_r[i*32 +: 32] <= kk;
                        p_r[i*32 +: 32]  <= sf_add(RQ_A, sf_mul(RQ_B, kk));
                        k_r[i*32 +: 32]  <= ee;
                    end
                end
                cnt_r <= 4'd1;
                state <= S_QN;
            end

            S_QN: begin
                for (i = 0; i < ELEMS; i = i + 1)
                    p_r[i*32 +: 32] <= sf_mul(p_r[i*32 +: 32],
                        sf_add(SF_ONE5,
                               sf_neg(sf_mul(SF_HALF,
                                             sf_mul(pv_r[i*32 +: 32],
                                                    sf_mul(p_r[i*32 +: 32], p_r[i*32 +: 32]))))));
                if (cnt_r == {1'b0, nm_r}) state <= S_QF;
                else                       cnt_r <= cnt_r + 4'd1;
            end

            S_QF: begin
                for (i = 0; i < ELEMS; i = i + 1) if (!don_r[i]) begin
                    yv = sf_ldexp(p_r[i*32 +: 32], -($signed(k_r[i*32 +: 32]) / 2));
                    if (op_r == 4'd4) yv = sf_mul(yv, x_r[i*32 +: 32]);
                    y_r[i*32 +: 32] <= yv;
                    if (sf_is_inf(yv)) ovf_l[i] <= 1'b1;
                end
                state <= S_FIN;
            end

            //-----------------------------------------------------------------
            // SIN / COS / SINCOS
            //   n = rne(x*2/pi), r = x - n*(pi/2) with a five-term split whose
            //   products are exact for |n| <= 2^20
            //-----------------------------------------------------------------
            S_SN: begin
                for (i = 0; i < ELEMS; i = i + 1) begin
                    xv = x_r[i*32 +: 32];
                    if (sf_is_nan(xv) || sf_is_inf(xv) ||
                        ((xv & 32'h7FFF_FFFF) > LIM_2P20)) begin
                        don_r[i] <= 1'b1; spc_r[i*32 +: 32] <= SF_QNAN; inv_l[i] <= 1'b1;
                    end else begin
                        k_r[i*32 +: 32] <= sf_irne(sf_mul(xv, INV2PI));
                        n_r[i*32 +: 32] <= sf_neg(sf_fint(sf_irne(sf_mul(xv, INV2PI))));
                        p_r[i*32 +: 32] <= xv;
                    end
                end
                cnt_r <= 4'd0;
                state <= S_SR;
            end

            S_SR: begin
                for (i = 0; i < ELEMS; i = i + 1)
                    p_r[i*32 +: 32] <= sf_add(p_r[i*32 +: 32],
                                              sf_mul(n_r[i*32 +: 32], crom[36 + cnt_r]));
                if (cnt_r == 4'd4) state <= S_SC;
                else               cnt_r <= cnt_r + 4'd1;
            end

            S_SC: begin
                for (i = 0; i < ELEMS; i = i + 1) begin
                    xv = p_r[i*32 +: 32];
                    kk = k_r[i*32 +: 32];
                    if ((!xv[31]) && (xv > PIO4)) begin
                        xv = sf_add(sf_add(xv, sf_neg(PIO2_HI)), sf_neg(PIO2_LO));
                        kk = kk + 32'd1;
                    end else if (xv[31] && ((xv & 32'h7FFF_FFFF) > PIO4)) begin
                        xv = sf_add(sf_add(xv, PIO2_HI), PIO2_LO);
                        kk = kk - 32'd1;
                    end
                    p_r[i*32 +: 32]  <= xv;
                    k_r[i*32 +: 32]  <= kk;
                    qd_r[i*2 +: 2]   <= kk[1:0];
                    pv_r[i*32 +: 32] <= sf_mul(xv, xv);
                    s_r[i*32 +: 32]  <= crom[21];
                    q_r[i*32 +: 32]  <= crom[27];
                end
                cnt_r <= 4'd1;
                state <= S_SP;
            end

            S_SP: begin
                for (i = 0; i < ELEMS; i = i + 1) begin
                    s_r[i*32 +: 32] <= sf_add(sf_mul(s_r[i*32 +: 32], pv_r[i*32 +: 32]),
                                              crom[21 - cnt_r]);
                    q_r[i*32 +: 32] <= sf_add(sf_mul(q_r[i*32 +: 32], pv_r[i*32 +: 32]),
                                              crom[27 - cnt_r]);
                end
                if (cnt_r == 4'd5) state <= S_SF;
                else               cnt_r <= cnt_r + 4'd1;
            end

            S_SF: begin
                for (i = 0; i < ELEMS; i = i + 1) if (!don_r[i]) begin
                    xv = sf_mul(p_r[i*32 +: 32], s_r[i*32 +: 32]);      // sin
                    yv = q_r[i*32 +: 32];                               // cos
                    case (qd_r[i*2 +: 2])
                    2'd0: begin                                  end
                    2'd1: begin xv = yv; yv = sf_neg(sf_mul(p_r[i*32 +: 32],
                                                            s_r[i*32 +: 32])); end
                    2'd2: begin xv = sf_neg(sf_mul(p_r[i*32 +: 32],
                                                   s_r[i*32 +: 32])); yv = sf_neg(yv); end
                    default: begin kk = xv; xv = sf_neg(yv); yv = kk; end
                    endcase
                    c_r[i*32 +: 32] <= yv;
                    y_r[i*32 +: 32] <= (op_r == 4'd6) ? yv : xv;
                end
                state <= S_FIN;
            end

            //-----------------------------------------------------------------
            // TANH : u = e^(2x)-1,  tanh(x) = u/(u+2)
            //-----------------------------------------------------------------
            S_TZ: begin
                for (i = 0; i < ELEMS; i = i + 1) begin
                    xv = x_r[i*32 +: 32];
                    if (sf_is_nan(xv)) begin
                        don_r[i] <= 1'b1; spc_r[i*32 +: 32] <= SF_QNAN; inv_l[i] <= 1'b1;
                    end else if ((!xv[31]) && (xv >= LIM_10)) begin
                        don_r[i] <= 1'b1; spc_r[i*32 +: 32] <= SF_ONE;
                    end else if (xv[31] && ((xv & 32'h7FFF_FFFF) >= LIM_10)) begin
                        don_r[i] <= 1'b1; spc_r[i*32 +: 32] <= SF_NEG1;
                    end else begin
                        kk = sf_mul(sf_mul(xv, SF_TWO), LOG2E);
                        k_r[i*32 +: 32] <= sf_irne(kk);
                        n_r[i*32 +: 32] <= sf_neg(sf_fint(sf_irne(kk)));
                        p_r[i*32 +: 32] <= sf_mul(xv, SF_TWO);
                    end
                end
                cnt_r <= 4'd0;
                state <= S_TR;
            end

            S_TR: begin
                for (i = 0; i < ELEMS; i = i + 1)
                    p_r[i*32 +: 32] <= sf_add(p_r[i*32 +: 32],
                                              sf_mul(n_r[i*32 +: 32], crom[41 + cnt_r]));
                if (cnt_r == 4'd2) state <= S_TU;
                else               cnt_r <= cnt_r + 4'd1;
            end

            S_TU: begin
                for (i = 0; i < ELEMS; i = i + 1) begin
                    pv_r[i*32 +: 32] <= p_r[i*32 +: 32];
                    p_r[i*32 +: 32]  <= crom[35];
                end
                cnt_r <= 4'd1;
                state <= S_TP;
            end

            S_TP: begin
                for (i = 0; i < ELEMS; i = i + 1)
                    p_r[i*32 +: 32] <= sf_add(sf_mul(p_r[i*32 +: 32], pv_r[i*32 +: 32]),
                                              crom[35 - cnt_r]);
                if (cnt_r == 4'd7) state <= S_TV;
                else               cnt_r <= cnt_r + 4'd1;
            end

            S_TV: begin
                for (i = 0; i < ELEMS; i = i + 1) begin
                    // p = expm1(f)/f, pv = f  ->  em1 = f*p,
                    // expm1(k+f) = 2^k * em1 + (2^k - 1)
                    yv = sf_mul(p_r[i*32 +: 32], pv_r[i*32 +: 32]);
                    xv = sf_add(sf_ldexp(yv, $signed(k_r[i*32 +: 32])),
                                sf_pow2m1($signed(k_r[i*32 +: 32])));   // u
                    kk = sf_add(xv, SF_TWO);                            // d = u+2
                    s_r[i*32 +: 32]  <= xv;
                    pv_r[i*32 +: 32] <= {1'b0, kk[22:0]} | SF_ONE;
                    p_r[i*32 +: 32]  <= sf_add(RCP_A,
                                        sf_mul(RCP_B, {1'b0, kk[22:0]} | SF_ONE));
                    k_r[i*32 +: 32]  <= {24'd0, kk[30:23]};
                end
                cnt_r <= 4'd1;
                state <= S_TN;
            end

            S_TN: begin
                for (i = 0; i < ELEMS; i = i + 1)
                    p_r[i*32 +: 32] <= sf_mul(p_r[i*32 +: 32],
                        sf_add(SF_TWO, sf_neg(sf_mul(pv_r[i*32 +: 32], p_r[i*32 +: 32]))));
                if (cnt_r == {1'b0, nm_r}) state <= S_TE;
                else                       cnt_r <= cnt_r + 4'd1;
            end

            S_TE: begin
                for (i = 0; i < ELEMS; i = i + 1) if (!don_r[i]) begin
                    yv = sf_ldexp(p_r[i*32 +: 32],
                                  {22'd0, 10'd127} - $signed(k_r[i*32 +: 32]));
                    y_r[i*32 +: 32] <= sf_mul(s_r[i*32 +: 32], yv);
                end
                state <= S_FIN;
            end

            //-----------------------------------------------------------------
            S_FIN: begin
                for (i = 0; i < ELEMS; i = i + 1)
                    dout[i*32 +: 32] <= sf_narrow(don_r[i] ? spc_r[i*32 +: 32]
                                                           : y_r[i*32 +: 32], fmo_r);
                out_valid <= 1'b1;
                frame_r   <= 1'b0;
                state     <= S_OUT;
            end

            S_OUT: begin
                if (out_ready) begin
                    if ((op_r == 4'd8) && !frame_r) begin
                        for (i = 0; i < ELEMS; i = i + 1)
                            dout[i*32 +: 32] <= sf_narrow(don_r[i] ? spc_r[i*32 +: 32]
                                                                   : c_r[i*32 +: 32], fmo_r);
                        frame_r <= 1'b1;
                    end else begin
                        out_valid <= 1'b0;
                        state     <= S_IDLE;
                    end
                end
            end

            default: state <= S_IDLE;
            endcase
        end
    end

endmodule
`default_nettype wire

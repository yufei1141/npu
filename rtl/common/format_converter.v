//=============================================================================
// format_converter : one element in flight, FP16/BF16/FP32 -> FP16/BF16/FP32
//
//   Follows docs/npu_integration/12_format_and_numeric.md sections 1..3.
//
//   - normal ranges and +/-0 are the working range, rounding is RNE with
//     guard/round/sticky (a BF16 narrowing is not a truncation)
//   - subnormal inputs and subnormal results are flushed to a signed zero and
//     reported as underflow|inexact (fixed FTZ profile of the peripheral)
//   - NaN/Inf input sets invalid; a narrowing overflow sets overflow
//   - the payload is latched, converted in the next cycle and held in HOLD
//     until accepted, so the handshake is backpressure-stable
//
//   fp_flags: [0] invalid, [1] overflow, [2] underflow, [3] inexact
//=============================================================================
//=============================================================================
// 中文说明
//
//   FP16 / BF16 / FP32 之间的单元素格式转换器，一次只处理一个元素。实现
//   依据 docs/npu_integration/12_format_and_numeric.md 第 1..3 节。
//
//   数值行为（这是本模块最需要说清楚的部分）：
//     - 正常工作区间只覆盖规格数与 ±0；舍入统一用 RNE（就近舍入、偶数优先），
//       并带 guard/round/sticky 三位，因此 BF16 收窄**不是**简单截断，而是
//       一次真正的舍入；
//     - 次正规输入和次正规结果一律冲刷（flush）为带符号的零，同时报
//       underflow|inexact——这是本外设固定的 FTZ（flush-to-zero）档位，不是
//       可选配置；
//     - 输入 NaN/Inf 置 invalid；收窄时上溢置 overflow；
//     - fp_flags 位定义：[0] invalid、[1] overflow、[2] underflow、[3] inexact。
//
//   微时序：
//     - 载荷先锁存，下一拍完成转换，再进入 HOLD 状态一直保持到被接收，因此
//       握手对反压是稳定的：上游随时可以停，数据不会丢也不会变形；
//     - 一次只有一个元素在途，没有内部吞吐流水，转换率约等于握手速率。
//=============================================================================
`timescale 1ns/1ps
`default_nettype none

module format_converter (
    input  wire        clk,
    input  wire        rst_n,

    input  wire        in_valid,
    output wire        in_ready,
    input  wire [31:0] in_bits,
    input  wire [1:0]  src_fmt,
    input  wire [1:0]  dst_fmt,
    input  wire [7:0]  in_tag,

    output wire        out_valid,
    input  wire        out_ready,
    output wire [31:0] out_bits,
    output wire [7:0]  out_tag,
    output wire [7:0]  fp_flags
);

    localparam F_FP16 = 2'd0;
    localparam F_BF16 = 2'd1;
    localparam F_FP32 = 2'd2;

    reg        busy;
    reg [31:0] in_r;
    reg [1:0]  src_r, dst_r;
    reg [7:0]  tag_r;

    reg [31:0] out_r;
    reg [7:0]  flags_r;
    reg        valid_r;

    assign in_ready  = !busy && !valid_r;
    assign out_valid = valid_r;
    assign out_bits  = out_r;
    assign out_tag   = tag_r;
    assign fp_flags  = flags_r;

    // ------------------------------------------------------------------
    // combinational conversion (from the latched payload)
    // ------------------------------------------------------------------
    reg [31:0] mid;        // source expanded to FP32
    reg [31:0] cvt;
    reg [7:0]  flg;

    reg        sgn;
    reg [7:0]  e32;
    reg [22:0] f32;
    reg [4:0]  e16;
    reg [7:0]  eb;
    reg        guard;
    reg [10:0] mant16;
    reg [7:0]  mantb;
    reg        rup;

    always @* begin
        mid = 32'd0;
        flg = 8'd0;

        // ---------------- source -> FP32 ----------------
        case (src_r)
        F_FP16: begin
            if (in_r[14:10] == 5'd0) begin
                mid = {in_r[15], 31'd0};
                if (in_r[9:0] != 10'd0)
                    flg = flg | 8'h0C;          // underflow | inexact
            end else if (in_r[14:10] == 5'h1F) begin
                mid = {in_r[15], 8'hFF, in_r[9:0] != 10'd0,
                       22'd0};
                flg = flg | 8'h01;              // invalid
            end else begin
                mid = {in_r[15], (in_r[14:10] + 8'd112), in_r[9:0], 13'd0};
            end
        end
        F_BF16: begin
            if (in_r[14:7] == 8'd0) begin
                mid = {in_r[15], 31'd0};
                if (in_r[6:0] != 7'd0)
                    flg = flg | 8'h0C;
            end else if (in_r[14:7] == 8'hFF) begin
                mid = {in_r[15], 8'hFF, in_r[6:0] != 7'd0, 22'd0};
                flg = flg | 8'h01;
            end else begin
                mid = {in_r[15], in_r[14:7], in_r[6:0], 16'd0};
            end
        end
        F_FP32: mid = in_r;
        default: flg = flg | 8'h01;
        endcase

        // ---------------- FP32 -> destination ----------------
        sgn = mid[31];
        e32 = mid[30:23];
        f32 = mid[22:0];
        cvt = 32'd0;

        case (dst_r)
        F_FP32: cvt = mid;

        F_FP16: begin
            if (e32 == 8'hFF) begin
                cvt = {sgn, 5'h1F, (f32 != 23'd0) ? 10'h200 : 10'd0};
                flg = flg | 8'h01;
            end else if (e32 == 8'd0) begin
                cvt = {sgn, 15'd0};
                if (f32 != 23'd0) flg = flg | 8'h0C;
            end else if (e32 > 8'd142) begin
                cvt = {sgn, 5'h1F, 10'd0};
                flg = flg | 8'h0A;              // overflow | inexact
            end else if (e32 < 8'd113) begin
                cvt = {sgn, 15'd0};
                flg = flg | 8'h0C;              // FTZ subnormal result
            end else begin
                e16   = e32 - 8'd112;
                guard = f32[12];
                rup   = guard && ((|f32[11:0]) || f32[13]);
                mant16 = {1'b0, f32[22:13]} + {10'd0, rup};
                if (mant16[10]) begin
                    e16 = e16 + 5'd1;
                    if (e16 >= 5'd31) begin
                        cvt = {sgn, 5'h1F, 10'd0};
                        flg = flg | 8'h0A;
                    end else begin
                        cvt = {sgn, e16, 10'd0};
                    end
                end else begin
                    cvt = {sgn, e16, mant16[9:0]};
                end
                if (guard || (|f32[11:0]))
                    flg = flg | 8'h08;          // inexact
            end
        end

        F_BF16: begin
            if (e32 == 8'hFF) begin
                cvt = {sgn, 8'hFF, (f32 != 23'd0) ? 7'h40 : 7'd0};
                flg = flg | 8'h01;
            end else if (e32 == 8'd0) begin
                cvt = {sgn, 15'd0};
                if (f32 != 23'd0) flg = flg | 8'h0C;
            end else begin
                guard  = f32[15];
                rup    = guard && ((|f32[14:0]) || f32[16]);
                mantb  = {1'b0, f32[22:16]} + {7'd0, rup};
                if (mantb[7]) begin
                    eb = e32 + 8'd1;
                    if (eb == 8'hFF) begin
                        cvt = {sgn, 8'hFF, 7'd0};
                        flg = flg | 8'h0A;
                    end else begin
                        cvt = {sgn, eb, 7'd0};
                    end
                end else begin
                    cvt = {sgn, e32, mantb[6:0]};
                end
                if (guard || (|f32[14:0]))
                    flg = flg | 8'h08;
            end
        end

        default: begin
            cvt = 32'd0;
            flg = flg | 8'h01;
        end
        endcase
    end

    // ------------------------------------------------------------------
    // handshake
    // ------------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            busy    <= 1'b0;
            in_r    <= 32'd0;
            src_r   <= 2'd0;
            dst_r   <= 2'd0;
            tag_r   <= 8'd0;
            out_r   <= 32'd0;
            flags_r <= 8'd0;
            valid_r <= 1'b0;
        end else begin
            if (in_valid && in_ready) begin
                in_r  <= in_bits;
                src_r <= src_fmt;
                dst_r <= dst_fmt;
                tag_r <= in_tag;
                busy  <= 1'b1;
            end else if (busy) begin
                out_r   <= cvt;
                flags_r <= flg;
                valid_r <= 1'b1;
                busy    <= 1'b0;
            end

            if (valid_r && out_ready)
                valid_r <= 1'b0;
        end
    end

endmodule

`default_nettype wire

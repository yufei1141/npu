//=============================================================================
// tb_format_converter : RNE / FTZ / flag checks for one-element conversions
//=============================================================================
`timescale 1ns/1ps
`default_nettype none

module tb_format_converter;

    localparam F_FP16 = 2'd0;
    localparam F_BF16 = 2'd1;
    localparam F_FP32 = 2'd2;

    reg clk, rst_n;
    initial begin clk = 0; rst_n = 0; end
    always #5 clk = ~clk;

    reg         in_valid;
    wire        in_ready;
    reg  [31:0] in_bits;
    reg  [1:0]  src_fmt, dst_fmt;
    wire        out_valid;
    reg         out_ready;
    wire [31:0] out_bits;
    wire [7:0]  out_tag;
    wire [7:0]  fp_flags;

    format_converter u_dut (
        .clk(clk), .rst_n(rst_n),
        .in_valid(in_valid), .in_ready(in_ready), .in_bits(in_bits),
        .src_fmt(src_fmt), .dst_fmt(dst_fmt), .in_tag(8'hA5),
        .out_valid(out_valid), .out_ready(out_ready),
        .out_bits(out_bits), .out_tag(out_tag), .fp_flags(fp_flags)
    );

    integer fails;

    task automatic conv;
        input [31:0] bits;
        input [1:0]  sf, df;
        input [31:0] exp_bits;
        input [7:0]  exp_flags;
        input [8*24-1:0] what;
        begin
            @(negedge clk);
            if (in_ready !== 1'b1) begin
                fails = fails + 1;
                $display("FAIL: %0s not ready at start", what);
            end
            in_bits = bits; src_fmt = sf; dst_fmt = df;
            in_valid = 1'b1;
            @(posedge clk);
            #1;
            @(negedge clk);
            in_valid = 1'b0;
            out_ready = 1'b1;
            while (out_valid !== 1'b1) @(negedge clk);
            if (out_bits !== exp_bits) begin
                fails = fails + 1;
                $display("FAIL: %0s got %h exp %h", what, out_bits, exp_bits);
            end
            if (fp_flags !== exp_flags) begin
                fails = fails + 1;
                $display("FAIL: %0s flags %h exp %h", what, fp_flags,
                         exp_flags);
            end
            if (out_tag !== 8'hA5) begin
                fails = fails + 1;
                $display("FAIL: %0s tag %h", what, out_tag);
            end
            @(negedge clk);
            out_ready = 1'b0;
        end
    endtask

    initial begin
        fails = 0;
        in_valid = 0; out_ready = 0;
        in_bits = 0; src_fmt = 0; dst_fmt = 0;

        repeat (3) @(negedge clk);
        rst_n = 1'b1;
        repeat (2) @(negedge clk);

        // ---- FP32 -> FP16 ----
        conv(32'h3F80_0000, F_FP32, F_FP16, 32'h0000_3C00, 8'h00, "f32->f16 1.0");
        conv(32'hC020_0000, F_FP32, F_FP16, 32'h0000_C100, 8'h00, "f32->f16 -2.5");
        conv(32'hBF20_0000, F_FP32, F_FP16, 32'h0000_B900, 8'h00, "f32->f16 -0.625");
        conv(32'h477F_E000, F_FP32, F_FP16, 32'h0000_7BFF, 8'h00, "f32->f16 max");
        conv(32'h477F_F000, F_FP32, F_FP16, 32'h0000_7C00, 8'h0A, "f32->f16 ovf");
        conv(32'h3880_0000, F_FP32, F_FP16, 32'h0000_0400, 8'h00, "f32->f16 minnorm");
        conv(32'h3200_0000, F_FP32, F_FP16, 32'h0000_0000, 8'h0C, "f32->f16 subn");
        // 1.0 + 2^-11 with lsb 0: tie rounds down
        conv(32'h3F80_1000, F_FP32, F_FP16, 32'h0000_3C00, 8'h08, "f32->f16 tie down");
        // 1.0 + 2^-11 + 2^-23: rounds up
        conv(32'h3F80_1001, F_FP32, F_FP16, 32'h0000_3C01, 8'h08, "f32->f16 tie up");
        // 1.0 + 2^-10 (lsb 1, exact): mantissa bit 0 set
        conv(32'h3F80_2000, F_FP32, F_FP16, 32'h0000_3C01, 8'h00, "f32->f16 exact");

        // ---- FP16 -> FP32 ----
        conv(32'h0000_3C00, F_FP16, F_FP32, 32'h3F80_0000, 8'h00, "f16->f32 1.0");
        conv(32'h0000_8000, F_FP16, F_FP32, 32'h8000_0000, 8'h00, "f16->f32 -0");
        conv(32'h0000_7C00, F_FP16, F_FP32, 32'h7F80_0000, 8'h01, "f16->f32 inf");
        conv(32'h0000_0001, F_FP16, F_FP32, 32'h0000_0000, 8'h0C, "f16->f32 subn");

        // ---- BF16 ----
        conv(32'h0000_3F80, F_BF16, F_FP32, 32'h3F80_0000, 8'h00, "bf16->f32 1.0");
        conv(32'h3F80_8000, F_FP32, F_BF16, 32'h0000_3F80, 8'h08, "f32->bf16 tie down");
        conv(32'h3F81_8000, F_FP32, F_BF16, 32'h0000_3F82, 8'h08, "f32->bf16 tie up");
        conv(32'h3F80_C000, F_FP32, F_BF16, 32'h0000_3F81, 8'h08, "f32->bf16 up");

        // ---- FP32 -> FP32 ----
        conv(32'hDEAD_BEEF, F_FP32, F_FP32, 32'hDEAD_BEEF, 8'h00, "f32->f32 pass");

        // ---- backpressure: hold out_ready low for a few cycles ----
        @(negedge clk);
        in_bits = 32'h3F80_0000; src_fmt = F_FP32; dst_fmt = F_FP16;
        in_valid = 1'b1;
        @(posedge clk);
        @(negedge clk);
        in_valid = 1'b0;
        repeat (3) @(negedge clk);
        if (!out_valid) begin
            fails = fails + 1;
            $display("FAIL: output not held under backpressure");
        end
        if (out_bits !== 32'h0000_3C00) begin
            fails = fails + 1;
            $display("FAIL: held output changed value");
        end
        out_ready = 1'b1;
        @(negedge clk);
        out_ready = 1'b0;

        if (fails == 0) $display("PASS: tb_format_converter");
        else            $display("FAIL: tb_format_converter (%0d)", fails);
        $finish;
    end

endmodule

`default_nettype wire

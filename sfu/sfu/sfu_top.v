`default_nettype none
// SFU top: two 4-element slices => 8 elements per operation.
module sfu_top #(
    parameter WIDTH  = 8,
    parameter DATA_W = 32
) (
    input  wire                  clk,
    input  wire                  rst_n,
    input  wire                  flush,

    input  wire                  in_valid,
    output wire                  in_ready,
    input  wire [3:0]            op,
    input  wire [1:0]            fmt_in,
    input  wire [1:0]            fmt_out,
    input  wire [1:0]            mode,
    input  wire [WIDTH*DATA_W-1:0] din,

    output wire                  out_valid,
    input  wire                  out_ready,
    output wire [WIDTH*DATA_W-1:0] dout,
    output wire [7:0]            status
);
    wire rst_n_i = rst_n & ~flush;

    wire       r0, r1;
    wire       v0, v1;
    wire [4*DATA_W-1:0] d0, d1;
    wire [7:0] s0, s1;

    sfu_slice #(.ELEMS(4), .DATA_W(DATA_W)) u_slice0 (
        .clk(clk), .rst_n(rst_n_i),
        .in_valid(in_valid), .in_ready(r0),
        .op(op), .fmt_in(fmt_in), .fmt_out(fmt_out), .mode(mode),
        .din(din[4*DATA_W-1:0]),
        .out_valid(v0), .out_ready(out_ready), .dout(d0), .status(s0)
    );

    sfu_slice #(.ELEMS(4), .DATA_W(DATA_W)) u_slice1 (
        .clk(clk), .rst_n(rst_n_i),
        .in_valid(in_valid), .in_ready(r1),
        .op(op), .fmt_in(fmt_in), .fmt_out(fmt_out), .mode(mode),
        .din(din[WIDTH*DATA_W-1:4*DATA_W]),
        .out_valid(v1), .out_ready(out_ready), .dout(d1), .status(s1)
    );

    assign in_ready  = r0 & r1;
    assign out_valid = v0 & v1;
    assign dout      = {d1, d0};
    assign status    = s0 | s1;
endmodule
`default_nettype wire

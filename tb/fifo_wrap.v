//=============================================================================
// fifo_wrap : op_fifo at the real operand width, for a Verilator harness
//
//   op_fifo is the only buffering between the MPU feed and gemv_cluster.
//   At WIDTH=2304 it is the widest array in the design, so it gets its own
//   standalone model: a Verilator problem here would look exactly like the
//   "correct operands in, zero result out" signature seen end to end.
//
//   Two instances, A (1024b) and B (2304b), both depth 8.
//=============================================================================
`timescale 1ns/1ps
`default_nettype none

module fifo_wrap (
    input  wire          clk,
    input  wire          rst_n,

    input  wire          a_wr,
    input  wire [1023:0] a_din,
    input  wire          a_rd,
    output wire          a_full,
    output wire          a_empty,
    output wire [1023:0] a_dout,

    input  wire          b_wr,
    input  wire [2303:0] b_din,
    input  wire          b_rd,
    output wire          b_full,
    output wire          b_empty,
    output wire [2303:0] b_dout
);

    op_fifo #(.WIDTH(1024), .DEPTH(8), .AW(3)) u_a (
        .clk(clk), .rst_n(rst_n),
        .wr(a_wr), .din(a_din), .rd(a_rd),
        .full(a_full), .empty(a_empty), .dout(a_dout));

    op_fifo #(.WIDTH(2304), .DEPTH(8), .AW(3)) u_b (
        .clk(clk), .rst_n(rst_n),
        .wr(b_wr), .din(b_din), .rd(b_rd),
        .full(b_full), .empty(b_empty), .dout(b_dout));

endmodule

`default_nettype wire

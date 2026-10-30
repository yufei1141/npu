//=============================================================================
// sync_fifo : single-clock, show-ahead (FWFT) FIFO
//
//   - dout is the head entry and is valid whenever !empty.
//   - one write and one read may fire in the same cycle.
//   - DEPTH must be a power of two.
//   - reset clears the pointers/count only; memory contents are don't-care.
//=============================================================================
`timescale 1ns/1ps
`default_nettype none

module sync_fifo #(
    parameter WIDTH = 32,
    parameter DEPTH = 8
)(
    input  wire             clk,
    input  wire             rst_n,

    input  wire             wr,
    input  wire [WIDTH-1:0] din,
    output wire             full,

    input  wire             rd,
    output wire [WIDTH-1:0] dout,
    output wire             empty,

    output wire [31:0]      count
);

    localparam AW = $clog2(DEPTH);

    reg [WIDTH-1:0] mem [0:DEPTH-1];
    reg [AW-1:0]    wp, rp;
    reg [AW:0]      cnt;

    wire do_wr = wr && !full;
    wire do_rd = rd && !empty;

    assign full  = (cnt == DEPTH[AW:0]);
    assign empty = (cnt == {(AW+1){1'b0}});
    assign dout  = mem[rp];
    assign count = {{(31-AW){1'b0}}, cnt};

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            wp  <= {AW{1'b0}};
            rp  <= {AW{1'b0}};
            cnt <= {(AW+1){1'b0}};
        end else begin
            if (do_wr) begin
                mem[wp] <= din;
                wp      <= wp + 1'b1;
            end
            if (do_rd)
                rp <= rp + 1'b1;

            case ({do_wr, do_rd})
                2'b10:   cnt <= cnt + 1'b1;
                2'b01:   cnt <= cnt - 1'b1;
                default: cnt <= cnt;
            endcase
        end
    end

endmodule

`default_nettype wire

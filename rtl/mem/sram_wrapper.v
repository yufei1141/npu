//=============================================================================
// sram_wrapper : synchronous single-port scratchpad RAM with fixed latency
//
//   Follows docs/npu_integration/05_memory_subsystem.md section 5.
//
//   One access is accepted when en=1 (there is no ready).  The response is
//   presented exactly L_RAM cycles later: rvalid/rdata/error.  Writes use
//   per-byte strobes and return zero data; reads ignore wstrb.
//
//   This is the simulation / FPGA-macro behavior model.  A real macro must
//   match the same synchronous write and registered read behavior.
//=============================================================================
`timescale 1ns/1ps
`default_nettype none

module sram_wrapper #(
    parameter MEM_DATA_W = 256,
    parameter DEPTH      = 1024,
    parameter L_RAM      = 1
)(
    input  wire                    clk,
    input  wire                    rst_n,

    input  wire                    en,
    input  wire                    we,
    input  wire [$clog2(DEPTH)-1:0] addr,
    input  wire [MEM_DATA_W-1:0]   wdata,
    input  wire [MEM_DATA_W/8-1:0] wstrb,

    output wire                    rvalid,
    output wire [MEM_DATA_W-1:0]   rdata,
    output wire                    error,

    input  wire                    err_inj
);

    localparam B = MEM_DATA_W / 8;
    localparam AW = $clog2(DEPTH);

    reg [MEM_DATA_W-1:0] mem [0:DEPTH-1];

    reg [L_RAM-1:0]      vpipe;
    reg [L_RAM-1:0]      epipe;
    reg [MEM_DATA_W-1:0] dpipe [0:L_RAM-1];

    integer bi, j;

    assign rvalid = vpipe[L_RAM-1];
    assign rdata  = dpipe[L_RAM-1];
    assign error  = epipe[L_RAM-1];

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            vpipe <= {L_RAM{1'b0}};
            epipe <= {L_RAM{1'b0}};
        end else begin
            vpipe[0] <= en;
            epipe[0] <= en && err_inj;
            for (j = 1; j < L_RAM; j = j + 1) begin
                vpipe[j] <= vpipe[j-1];
                epipe[j] <= epipe[j-1];
            end

            if (en) begin
                if (we) begin
                    for (bi = 0; bi < B; bi = bi + 1) begin
                        if (wstrb[bi])
                            mem[addr][8*bi +: 8] <= wdata[8*bi +: 8];
                    end
                    dpipe[0] <= {MEM_DATA_W{1'b0}};
                end else begin
                    dpipe[0] <= mem[addr];
                end
            end

            for (j = 1; j < L_RAM; j = j + 1)
                dpipe[j] <= dpipe[j-1];
        end
    end

endmodule

`default_nettype wire

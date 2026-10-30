//=============================================================================
// extmem_model : behavioural external memory for the DMA frame protocol
//
//   One read or write row transaction at a time, 128-bit beats, exact keep on
//   the final beat, last only on the final beat.  Write confirmation is
//   returned WR_RSP_DELAY cycles after the final beat is accepted.
//
//   mem[] is exposed so a testbench can preload and inspect bytes directly.
//=============================================================================
`timescale 1ns/1ps
`default_nettype none

module extmem_model #(
    parameter MEM_BYTES    = 65536,
    parameter WR_RSP_DELAY = 2
)(
    input  wire         clk,
    input  wire         rst_n,

    input  wire         rd_req_valid,
    output wire         rd_req_ready,
    input  wire [63:0]  rd_req_addr,
    input  wire [31:0]  rd_req_bytes,
    input  wire [15:0]  rd_req_tag,

    output wire         rd_valid,
    input  wire         rd_ready,
    output wire [127:0] rd_data,
    output wire [15:0]  rd_keep,
    output wire         rd_last,
    output wire [15:0]  rd_tag,

    input  wire         wr_req_valid,
    output wire         wr_req_ready,
    input  wire [63:0]  wr_req_addr,
    input  wire [31:0]  wr_req_bytes,
    input  wire [15:0]  wr_req_tag,

    input  wire         wr_valid,
    output wire         wr_ready,
    input  wire [127:0] wr_data,
    input  wire [15:0]  wr_keep,
    input  wire         wr_last,
    input  wire [15:0]  wr_tag,

    output wire         wr_rsp_valid,
    input  wire         wr_rsp_ready,
    output wire [15:0]  wr_rsp_tag,
    output reg          wr_rsp_error
);

    reg [7:0] mem [0:MEM_BYTES-1];

    // ---------------- read path ----------------
    localparam R_IDLE = 2'd0;
    localparam R_WAIT = 2'd1;
    localparam R_DATA = 2'd2;

    reg [1:0]  r_state;
    reg [63:0] r_addr;
    reg [31:0] r_rem;
    reg [15:0] r_tag;

    assign rd_req_ready = (r_state == R_IDLE);
    assign rd_valid     = (r_state == R_DATA);
    assign rd_keep      = (r_rem >= 16) ? 16'hFFFF
                                        : ((16'h0001 << r_rem) - 16'h0001);
    assign rd_last      = (r_state == R_DATA) && (r_rem <= 16);
    assign rd_tag       = r_tag;

    // The beat is assembled in the clocked block on purpose.  Reading a
    // memory array from an always @* puts the whole array on that block's
    // sensitivity list, and every preload write then re-evaluates the block,
    // which turns a large image load quadratic in the simulator.
    integer rk;
    reg [127:0] rd_data_r;
    assign rd_data = rd_data_r;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            r_state <= R_IDLE;
            r_addr  <= 64'd0;
            r_rem   <= 32'd0;
            r_tag   <= 16'd0;
            rd_data_r <= 128'd0;
        end else begin
            case (r_state)
            R_IDLE: if (rd_req_valid) begin
                // A masked address would silently return the wrong bytes for a
                // malformed 64-bit address (this is exactly how a {P1,P0}
                // LOAD_W address went unnoticed), so flag it loudly.
                if ((rd_req_addr + rd_req_bytes) > MEM_BYTES)
                    $display("EXTMEM ERROR: read 0x%016h +%0d exceeds %0d bytes",
                             rd_req_addr, rd_req_bytes, MEM_BYTES);
                r_addr  <= rd_req_addr;
                r_rem   <= rd_req_bytes;
                r_tag   <= rd_req_tag;
                r_state <= R_WAIT;
            end
            R_WAIT: begin
                for (rk = 0; rk < 16; rk = rk + 1)
                    rd_data_r[8*rk +: 8] <= (rk < r_rem)
                        ? mem[r_addr[15:0] + rk] : 8'h00;
                r_state <= R_DATA;
            end
            R_DATA: if (rd_ready) begin
                if (r_rem <= 16) begin
                    r_state <= R_IDLE;
                end else begin
                    r_addr <= r_addr + 64'd16;
                    r_rem  <= r_rem - 32'd16;
                    for (rk = 0; rk < 16; rk = rk + 1)
                        rd_data_r[8*rk +: 8] <= (rk < r_rem - 32'd16)
                            ? mem[r_addr[15:0] + 16 + rk] : 8'h00;
                end
            end
            default: r_state <= R_IDLE;
            endcase
        end
    end

    // ---------------- write path ----------------
    localparam W_IDLE = 2'd0;
    localparam W_DATA = 2'd1;
    localparam W_RSP  = 2'd2;

    reg [1:0]  w_state;
    reg [63:0] w_addr;
    reg [31:0] w_rem;
    reg [15:0] w_tag;
    reg [3:0]  w_delay;
    integer    k;
    reg        w_rsp_valid_r;
    reg [15:0] w_rsp_tag_r;

    assign wr_req_ready  = (w_state == W_IDLE);
    assign wr_ready      = (w_state == W_DATA);
    assign wr_rsp_valid  = w_rsp_valid_r;
    assign wr_rsp_tag    = w_rsp_tag_r;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            w_state <= W_IDLE;
            w_addr  <= 64'd0;
            w_rem   <= 32'd0;
            w_tag   <= 16'd0;
            w_delay <= 4'd0;
            w_rsp_valid_r <= 1'b0;
            w_rsp_tag_r   <= 16'd0;
            wr_rsp_error  <= 1'b0;
        end else begin
            case (w_state)
            W_IDLE: if (wr_req_valid) begin
                if ((wr_req_addr + wr_req_bytes) > MEM_BYTES) begin
                    $display("EXTMEM ERROR: write 0x%016h +%0d exceeds %0d bytes",
                             wr_req_addr, wr_req_bytes, MEM_BYTES);
                    wr_rsp_error <= 1'b1;
                end else begin
                    wr_rsp_error <= 1'b0;
                end
                w_addr  <= wr_req_addr;
                w_rem   <= wr_req_bytes;
                w_tag   <= wr_req_tag;
                w_state <= W_DATA;
            end
            W_DATA: if (wr_valid) begin
                for (k = 0; k < 16; k = k + 1)
                    if (wr_keep[k])
                        mem[w_addr[15:0] + k] <= wr_data[8*k +: 8];
                if (wr_last) begin
                    w_delay <= WR_RSP_DELAY[3:0];
                    w_state <= W_RSP;
                end else begin
                    w_addr <= w_addr + 64'd16;
                    w_rem  <= w_rem - 32'd16;
                end
            end
            W_RSP: begin
                if (w_rsp_valid_r) begin
                    if (wr_rsp_ready) begin
                        w_rsp_valid_r <= 1'b0;
                        w_state       <= W_IDLE;
                    end
                end else if (w_delay != 4'd0) begin
                    w_delay <= w_delay - 4'd1;
                end else begin
                    w_rsp_valid_r <= 1'b1;
                    w_rsp_tag_r   <= w_tag;
                end
            end
            default: w_state <= W_IDLE;
            endcase
        end
    end

endmodule

`default_nettype wire

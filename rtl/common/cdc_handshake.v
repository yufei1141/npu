//=============================================================================
// cdc_handshake : four-phase request/acknowledge mailbox with a wide payload
//
//   Source side: pulse src_start with a stable src_data.  src_busy is high
//   from the start pulse until the full request/ack_r round trip has completed;
//   src_done is a one-cycle pulse at that point.
//
//   Destination side: dst_valid/dst_data present the captured payload and
//   stay stable until dst_ready fires.  Only after that does the destination
//   acknowledge, and only after the source drops req does ack_r drop.
//
//   The payload is not synchronized bit-by-bit.  It is held stable by the
//   source from before req rises until the source has sampled ack_r high, and
//   the destination samples it once into its own holding register.
//=============================================================================
`timescale 1ns/1ps
`default_nettype none

module cdc_handshake #(
    parameter WIDTH  = 128,
    parameter SYNC   = 2
)(
    // source domain
    input  wire              src_clk,
    input  wire              src_rst_n,
    input  wire              src_start,
    input  wire [WIDTH-1:0]  src_data,
    output wire              src_busy,
    output wire              src_done,

    // destination domain
    input  wire              dst_clk,
    input  wire              dst_rst_n,
    output wire              dst_valid,
    output wire [WIDTH-1:0]  dst_data,
    input  wire              dst_ready
);

    localparam SRC_IDLE      = 2'd0;
    localparam SRC_WAIT_ACK1 = 2'd1;
    localparam SRC_WAIT_ACK0 = 2'd2;

    localparam DST_IDLE      = 2'd0;
    localparam DST_HOLD      = 2'd1;
    localparam DST_WAIT_REQ0 = 2'd2;

    // ------------------------------------------------------------------
    // source domain
    // ------------------------------------------------------------------
    reg [1:0] src_state;
    reg       req;

    (* ASYNC_REG = "TRUE" *) reg [SYNC-1:0] ack_sync;
    wire ack_seen = ack_sync[SYNC-1];

    reg done_r;

    assign src_busy = (src_state != SRC_IDLE);
    assign src_done = done_r;

    always @(posedge src_clk or negedge src_rst_n) begin
        if (!src_rst_n) begin
            src_state <= SRC_IDLE;
            req       <= 1'b0;
            done_r    <= 1'b0;
            ack_sync  <= {SYNC{1'b0}};
        end else begin
            done_r   <= 1'b0;
            ack_sync <= {ack_sync[SYNC-2:0], ack_r};

            case (src_state)
                SRC_IDLE: begin
                    if (src_start) begin
                        req       <= 1'b1;
                        src_state <= SRC_WAIT_ACK1;
                    end
                end
                SRC_WAIT_ACK1: begin
                    if (ack_seen) begin
                        req       <= 1'b0;
                        src_state <= SRC_WAIT_ACK0;
                    end
                end
                SRC_WAIT_ACK0: begin
                    if (!ack_seen) begin
                        done_r    <= 1'b1;
                        src_state <= SRC_IDLE;
                    end
                end
                default: src_state <= SRC_IDLE;
            endcase
        end
    end

    // ------------------------------------------------------------------
    // destination domain
    // ------------------------------------------------------------------
    reg [1:0] dst_state;
    reg [WIDTH-1:0] data_r;
    reg       valid_r;
    reg       ack_r;

    (* ASYNC_REG = "TRUE" *) reg [SYNC-1:0] req_sync;
    wire req_seen = req_sync[SYNC-1];

    assign dst_valid = valid_r;
    assign dst_data  = data_r;

    always @(posedge dst_clk or negedge dst_rst_n) begin
        if (!dst_rst_n) begin
            dst_state <= DST_IDLE;
            data_r    <= {WIDTH{1'b0}};
            valid_r   <= 1'b0;
            ack_r       <= 1'b0;
            req_sync  <= {SYNC{1'b0}};
        end else begin
            req_sync <= {req_sync[SYNC-2:0], req};

            case (dst_state)
                DST_IDLE: begin
                    // Capture only on a fresh rising request.  ack_r is low in
                    // this state, so this is the first observation.
                    if (req_seen && !ack_r) begin
                        data_r    <= src_data;
                        valid_r   <= 1'b1;
                        dst_state <= DST_HOLD;
                    end
                end
                DST_HOLD: begin
                    if (valid_r && dst_ready) begin
                        valid_r   <= 1'b0;
                        ack_r       <= 1'b1;
                        dst_state <= DST_WAIT_REQ0;
                    end
                end
                DST_WAIT_REQ0: begin
                    if (!req_seen) begin
                        ack_r       <= 1'b0;
                        dst_state <= DST_IDLE;
                    end
                end
                default: dst_state <= DST_IDLE;
            endcase
        end
    end

endmodule

`default_nettype wire

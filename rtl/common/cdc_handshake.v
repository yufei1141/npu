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
//=============================================================================
// 中文说明
//
//   四相握手（req / ack）的跨时钟域邮箱，用来安全搬运一段**宽数据**。它
//   之所以不是"把每一根数据线都打两拍"，是因为那只能保证每一根线各自稳定，
//   却保证不了它们在同一拍被采到——多比特信号跨域必须靠握手或格雷码，这里
//   选的是握手。
//
//   源端时序：
//     - 把 src_data 摆稳之后打一拍 src_start（单周期脉冲）；
//     - src_busy 从 start 一直保持到 req/ack_r 整个往返结束；
//     - 往返结束那一拍给出 src_done 单周期脉冲，表示"对端已经收下"。
//
//   目的端时序：
//     - 用 dst_valid/dst_data 呈现已经捕获的载荷，数据保持稳定直到 dst_ready
//       握手成功；
//     - 只有握手成功之后目的端才回 ack；源端看到 ack 拉高、撤掉 req 之后，
//       ack_r 才会落下——这就是"四相"：req↑、ack↑、req↓、ack↓。
//
//   关键不变量：
//     - 载荷不是逐位同步的，它在 req 上升之前就已经稳定，并且一直保持到
//       源端采到 ack_r 为高，目的端只在自己域内采样一次并锁存；
//     - 因此载荷的建立/保持时间覆盖了整个同步器延迟，不存在采到中间值的
//       窗口；
//     - 一次只能传一段载荷：在 src_busy 为高期间不得再次触发 start。
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

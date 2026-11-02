//=============================================================================
// mem_arb2 : two-way grant arbiter for one scratchpad client
//
//   The MPU owns exactly one memory client (MPU_A) and the memory subsystem
//   allows one request in flight on it.  Two producers now want that client:
//   the A packer (activations) and the B frame reader (activation-side B).
//   Splitting the client would need a sixth port in every bank, so instead
//   the grant is held from the moment a request is offered until its response
//   has been taken; the losing side simply waits, which it already handles
//   because its own FSM stalls with req_valid high.
//
//   Fairness: strict alternation after every grant, so a long B burst cannot
//   starve the A packer of the beats that keep the core's A FIFO fed.
//=============================================================================
//=============================================================================
// 中文说明
//
//   给同一个 scratchpad 客户端做两路授权仲裁。MPU 在 memory_subsystem 里只
//   占用一个客户端（MPU_A），而该客户端规则是"最多一个请求在途"。现在有两个
//   生产者都要用它：A packer（激活 A）和 B frame reader（激活侧的 B）。
//
//   为什么不拆端口：拆成两个客户端意味着每个 bank 要多一个读写端口，8 个
//   bank 就是 8 个新端口，面积和布线代价远大于这里的收益。所以采取的折中是
//   ——**从请求提出开始就锁住授权，直到它的响应被取走**，另一方在原地等。
//   落败的一方不需要额外逻辑：它自己的状态机本来就是 req_valid 保持高、
//   等 ready 的写法，天然能承受等待。
//
//   公平性：每次授权之后严格交替，因此一次很长的 B 帧突发也无法把 A packer
//   饿死——A FIFO 一旦断供，tensorcore 就会停转，所以这里用严格轮转而不是
//   固定优先级。
//=============================================================================
`timescale 1ns/1ps
`default_nettype none

`include "npu_defs.vh"

module mem_arb2 #(
    parameter MEM_DATA_W = 256
)(
    input  wire        clk_core,
    input  wire        rst_core_n,

    // side A (activation packer)
    input  wire        a_req_valid,
    output wire        a_req_ready,
    input  wire [`NPU_REQ_W(MEM_DATA_W, MEM_DATA_W/8)-1:0] a_req,
    output wire        a_rsp_valid,
    input  wire        a_rsp_ready,
    output wire [`NPU_RSP_W(MEM_DATA_W)-1:0] a_rsp,

    // side B (B frame reader)
    input  wire        b_req_valid,
    output wire        b_req_ready,
    input  wire [`NPU_REQ_W(MEM_DATA_W, MEM_DATA_W/8)-1:0] b_req,
    output wire        b_rsp_valid,
    input  wire        b_rsp_ready,
    output wire [`NPU_RSP_W(MEM_DATA_W)-1:0] b_rsp,

    // memory client
    output wire        mem_req_valid,
    input  wire        mem_req_ready,
    output wire [`NPU_REQ_W(MEM_DATA_W, MEM_DATA_W/8)-1:0] mem_req,
    input  wire        mem_rsp_valid,
    output wire        mem_rsp_ready,
    input  wire [`NPU_RSP_W(MEM_DATA_W)-1:0] mem_rsp
);

    localparam REQ_W = `NPU_REQ_W(MEM_DATA_W, MEM_DATA_W/8);
    localparam RSP_W = `NPU_RSP_W(MEM_DATA_W);

    localparam G_NONE = 2'd0;
    localparam G_A    = 2'd1;
    localparam G_B    = 2'd2;

    reg [1:0] grant;
    reg       pref_b;          // preference for the next grant

    // A request may be offered while the other side still owns the client;
    // only the granted side reaches the memory port.
    wire pick_b = b_req_valid && (!a_req_valid || pref_b);

    // Nothing is offered to the client while the grant is being chosen: the
    // grant is registered, and offering in the same cycle would let the
    // memory subsystem accept a request whose owner never saw req_ready (its
    // ready is gated by the grant).  The owner would then sit in its request
    // state while its response waits with rsp_ready low -- one dead client.
    assign mem_req       = (grant == G_B) ? b_req : a_req;
    assign mem_req_valid = (grant == G_A) ? a_req_valid :
                           (grant == G_B) ? b_req_valid : 1'b0;

    assign a_req_ready = (grant == G_A) && mem_req_ready;
    assign b_req_ready = (grant == G_B) && mem_req_ready;

    assign a_rsp_valid = mem_rsp_valid && (grant == G_A);
    assign b_rsp_valid = mem_rsp_valid && (grant == G_B);
    assign a_rsp       = mem_rsp;
    assign b_rsp       = mem_rsp;
    assign mem_rsp_ready = (grant == G_B) ? b_rsp_ready : a_rsp_ready;

    always @(posedge clk_core or negedge rst_core_n) begin
        if (!rst_core_n) begin
            grant  <= G_NONE;
            pref_b <= 1'b1;
        end else begin
            case (grant)
            G_NONE: begin
                if (pick_b) begin
                    grant  <= G_B;
                    pref_b <= 1'b0;
                end else if (a_req_valid) begin
                    grant  <= G_A;
                    pref_b <= 1'b1;
                end
            end
            // The grant is held from the offer until the response has been
            // accepted, so the client never sees two outstanding requests.
            G_A: begin
                if (mem_rsp_valid && a_rsp_ready) grant <= G_NONE;
            end
            G_B: begin
                if (mem_rsp_valid && b_rsp_ready) grant <= G_NONE;
            end
            default: grant <= G_NONE;
            endcase
        end
    end

endmodule

`default_nettype wire

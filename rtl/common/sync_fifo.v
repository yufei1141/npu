//=============================================================================
// sync_fifo : single-clock, show-ahead (FWFT) FIFO
//
//   - dout is the head entry and is valid whenever !empty.
//   - one write and one read may fire in the same cycle.
//   - DEPTH must be a power of two.
//   - reset clears the pointers/count only; memory contents are don't-care.
//=============================================================================
//=============================================================================
// 中文说明
//
//   单时钟、首字直通（FWFT / show-ahead）的同步 FIFO。读端不需要"先看到
//   empty 再发 rd_en、下一拍才取到数据"这种两段式操作：只要 !empty，dout
//   上就已经是队头条目，rd_en 只负责在握手成功时把它弹出。
//
//   接口与时序：
//     - 写：wr_en && !full 时在时钟上升沿写入一条 WIDTH 位数据；
//     - 读：rd_en && !empty 时弹出队头，dout 在同一沿之后指向下一条；
//     - 同一拍可以同时完成一次写和一次读，稳态吞吐为每周期一条；
//     - full / empty 由指针与计数组合产生，可以同拍直接使用，不需要额外
//       的"满/空延迟一拍"寄存器。
//
//   约束与假设：
//     - DEPTH 必须是 2 的幂，指针回绕靠位宽自然截断，避免显式的取模比较；
//     - 复位只清指针与计数，存储阵列内容视为无效（don't care），因此复位
//       后必须"先写后读"，不能依赖复位后读到的数据；
//     - 本模块不做任何跨时钟域处理，跨域请用 async_fifo。
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

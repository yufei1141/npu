//=============================================================================
// async_fifo : Gray-pointer dual-clock FIFO, show-ahead read
//
//   Standard independent-clock FIFO:
//     - binary pointers stay local, Gray pointers cross the clock boundary
//     - two-flop synchronizers per direction
//     - full is computed in the write domain, empty in the read domain
//   DEPTH must be a power of two (>= 2).  Reset is expected to be coordinated
//   by the platform: this module only resets its own pointer state.
//
//   The memory is written on wclk and read combinationally on the read side.
//   That is the usual simulation/synthesis split for a dual-clock RAM; the
//   real macro must provide the same synchronous write / async-or-registered
//   read behavior chosen here.
//=============================================================================
//=============================================================================
// 中文说明
//
//   带格雷码指针的双时钟 FIFO，读端同样是首字直通。核心思想：二进制指针
//   只在本时钟域内使用，跨域只传格雷码指针——格雷码每次递增只翻转 1 位，
//   即使被异步采到，读到的也只可能是"旧值"或"新值"，不会出现多比特同时
//   变化的中间态，因此不需要握手就能安全传指针。
//
//   结构：
//     - 每个方向两级触发器同步器，把对方的格雷码指针同步到本域；
//     - full 在写时钟域算（用本域写指针 + 同步来的读指针），empty 在读
//       时钟域算（用本域读指针 + 同步来的写指针），互不跨越组合路径；
//     - 存储阵列在 wclk 上同步写，读侧组合读出，因此读延迟为 0。
//
//   约束与假设：
//     - DEPTH 必须是 2 的幂且 >= 2；
//     - 复位必须由平台统一协调，本模块只清自己的指针状态。两端复位释放
//       不同步会让指针相位错乱、满/空判断失效——这是此类 FIFO 最常见的
//       误用方式；
//     - 存储模型是"同步写 + 组合读"。综合时若映射到只有寄存器读的宏，
//       必须在读侧补一级延迟，并相应把 empty 的判定提前一拍，否则首字
//       直通的语义会被破坏。
//=============================================================================
`timescale 1ns/1ps
`default_nettype none

module async_fifo #(
    parameter WIDTH = 32,
    parameter DEPTH = 8
)(
    input  wire             wclk,
    input  wire             wrst_n,
    input  wire             wr,
    input  wire [WIDTH-1:0] din,
    output wire             full,

    input  wire             rclk,
    input  wire             rrst_n,
    input  wire             rd,
    output wire [WIDTH-1:0] dout,
    output wire             empty
);

    localparam AW = $clog2(DEPTH);

    reg [WIDTH-1:0] mem [0:DEPTH-1];

    // write domain
    reg  [AW:0] wbin, wgray;
    reg  [AW:0] rq1_rgray, rq2_rgray;

    // read domain
    reg  [AW:0] rbin, rgray;
    reg  [AW:0] wq1_wgray, wq2_wgray;

    wire [AW:0] wbin_inc  = wbin + {{AW{1'b0}}, (wr && !full)};
    wire [AW:0] wgray_inc = (wbin_inc >> 1) ^ wbin_inc;

    wire [AW:0] rbin_inc  = rbin + {{AW{1'b0}}, (rd && !empty)};
    wire [AW:0] rgray_inc = (rbin_inc >> 1) ^ rbin_inc;

    // Flags are computed from the current pointers, not from the next ones.
    // Using the next pointer would make full/empty depend combinationally on
    // wr/rd while wr/rd depend on the flags: at the fill boundary that is a
    // zero-delay loop.  The current-pointer form is the same condition
    // (distance == DEPTH for full, pointers equal for empty).
    assign full = (wgray ==
                   {~rq2_rgray[AW:AW-1], rq2_rgray[AW-2:0]});

    assign empty = (rgray == wq2_wgray);

    assign dout = mem[rbin[AW-1:0]];

    // ------------------------------------------------------------------
    // write clock domain
    // ------------------------------------------------------------------
    always @(posedge wclk or negedge wrst_n) begin
        if (!wrst_n) begin
            wbin      <= {(AW+1){1'b0}};
            wgray     <= {(AW+1){1'b0}};
            rq1_rgray <= {(AW+1){1'b0}};
            rq2_rgray <= {(AW+1){1'b0}};
        end else begin
            rq1_rgray <= rgray;
            rq2_rgray <= rq1_rgray;

            if (wr && !full) begin
                mem[wbin[AW-1:0]] <= din;
                wbin              <= wbin_inc;
                wgray             <= wgray_inc;
            end
        end
    end

    // ------------------------------------------------------------------
    // read clock domain
    // ------------------------------------------------------------------
    always @(posedge rclk or negedge rrst_n) begin
        if (!rrst_n) begin
            rbin      <= {(AW+1){1'b0}};
            rgray     <= {(AW+1){1'b0}};
            wq1_wgray <= {(AW+1){1'b0}};
            wq2_wgray <= {(AW+1){1'b0}};
        end else begin
            wq1_wgray <= wgray;
            wq2_wgray <= wq1_wgray;

            if (rd && !empty) begin
                rbin  <= rbin_inc;
                rgray <= rgray_inc;
            end
        end
    end

endmodule

`default_nettype wire

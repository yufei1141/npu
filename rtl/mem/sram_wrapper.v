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
//=============================================================================
// 中文说明
//
//   同步单端口 scratchpad RAM，延迟固定。实现依据
//   docs/npu_integration/05_memory_subsystem.md 第 5 节。
//
//   接口语义：
//     - en=1 即接受一次访问，没有 ready 信号（上游必须自己保证时序）；
//     - 响应严格在 L_RAM 拍之后出现：rvalid/rdata/error 一起到达；
//     - 写用逐字节 strobe（wstrb），写回数据恒为 0；读忽略 wstrb；
//     - 同一拍读写由 we 决定，写优先。
//
//   存储阵列的写法（这一版为了能上 FPGA 做过调整）：
//     - 阵列访问与寄存器读放在**无复位**的 always 块里，和时钟沿同步。这是
//       Vivado 能推断出 block RAM 的形状；如果阵列也跟着异步复位，综合器会
//       把整个阵列展开成寄存器（256 KB 会变成 26 万个触发器）；
//     - vpipe/epipe 和 dpipe[1..] 仍在带异步复位的块里，复位后 rvalid=0，
//       所以期间 rdata 的取值无人使用；
//     - initial 块把阵列和 dpipe 清 0，保持"复位后可读到 0"的既有仿真行为，
//       BRAM 推断也接受初值。
//
//   读写同拍的取值：写用非阻塞赋值，读表达式取的是**写之前**的旧内容，因此
//   同地址同拍读写读到旧值。真实存储宏必须提供同样的"同步写 + 寄存器读"
//   行为，否则需要在本模块外加旁路逻辑。
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

    // The array carries no reset: resetting a RAM makes it a bank of
    // registers, which is what Vivado rejected ("RAM is sensitive to
    // asynchronous reset signal").  A zero power-up value keeps the
    // simulation behaviour the protocol testbenches relied on, and BRAM
    // inference accepts an initial value.
    initial begin
        for (j = 0; j < DEPTH; j = j + 1)
            mem[j] = {MEM_DATA_W{1'b0}};
        for (j = 0; j < L_RAM; j = j + 1)
            dpipe[j] = {MEM_DATA_W{1'b0}};
    end

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
        end
    end

    // Array access and the registered read share one clocked block with no
    // reset, which is the pattern a block RAM infers from.  The read returns
    // the old contents when a write lands in the same cycle, as before.
    always @(posedge clk) begin
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
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (j = 1; j < L_RAM; j = j + 1)
                dpipe[j] <= {MEM_DATA_W{1'b0}};
        end else begin
            for (j = 1; j < L_RAM; j = j + 1)
                dpipe[j] <= dpipe[j-1];
        end
    end

endmodule

`default_nettype wire

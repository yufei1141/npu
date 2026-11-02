//=============================================================================
// weight_bank : one ping/pong weight tile bank
//
//   Follows docs/npu_integration/07_weight_buffer.md sections 1 and 4.
//
//   Storage per group j: weight low/high 128 bit x K_STEPS and scale
//   128 bit x ceil(K_STEPS/4).  Write port is in the io domain, read port in
//   the core domain; the ownership FSMs make sure they never overlap.
//
//   io side   FREE -> FILLING -> COMMITTING -> CORE_OWNED -> FREE
//   core side EMPTY -> READY -> READING -> RELEASING -> EMPTY
//
//   The descriptor is handed over with a four-phase request/acknowledge
//   mailbox (io -> core); release uses the mirror image (core -> io).  The
//   re-acknowledge of a new commit is only given in EMPTY, so a replacement
//   tile cannot overtake a bank that is still releasing.
//
//   RAMs are modelled as behavioural arrays: synchronous write on clk_io,
//   registered read on clk_core with one cycle of latency.  A real memory
//   macro must match that timing.
//=============================================================================
//=============================================================================
// 中文说明
//
//   一组 ping/pong 权重 tile bank。实现依据
//   docs/npu_integration/07_weight_buffer.md 第 1、4 节。
//
//   存储布局（每组 j）：权重低/高各 128 位 × K_STEPS，scale 128 位 ×
//   ceil(K_STEPS/4)。写口在 io 域、读口在 core 域，两侧各自的所有权状态机
//   保证两者永不重叠——这是本模块的核心约束，不是性能优化。
//
//   所有权状态机：
//     io 侧  ：FREE -> FILLING -> COMMITTING -> CORE_OWNED -> FREE
//     core 侧：EMPTY -> READY -> READING -> RELEASING -> EMPTY
//   描述符通过四相请求/应答邮箱从 io 交给 core，释放则用镜像方向（core ->
//   io）。新的填充只在 **EMPTY** 状态才被应答，因此替换 tile 不会越过一个
//   还在释放的 bank——否则会出现 loader 已经往 bank 里写、而 core 仍在读旧
//   tile 的撕裂。
//
//   RAM 建模：行为数组，clk_io 上同步写、clk_core 上寄存器读、延迟一拍。真实
//   存储宏必须提供同样时序，否则两侧状态机的"不重叠"论证就失效了。
//=============================================================================
`timescale 1ns/1ps
`default_nettype none

`include "npu_defs.vh"

module weight_bank #(
    parameter GROUPS  = 8,
    parameter K_STEPS = 256
)(
    input  wire        clk_io,
    input  wire        rst_io_n,
    input  wire        clk_core,
    input  wire        rst_core_n,

    // ---------------- io side: allocation and fill ----------------
    output wire        io_free,
    input  wire        io_fill_start,
    input  wire        io_publish,
    input  wire        io_abort,
    input  wire [`NPU_WTILE_W-1:0] io_desc,
    output wire        io_commit_done,
    output wire [15:0] io_commit_status,

    input  wire        wr_en,
    input  wire        wr_kind,                       // 0 weight, 1 scale
    input  wire [2:0]  wr_grp,
    input  wire        wr_half,
    input  wire [$clog2(K_STEPS)-1:0] wr_addr,
    input  wire [127:0] wr_data,

    // ---------------- core side: descriptor handoff ----------------
    output wire        core_desc_valid,
    output wire [`NPU_WTILE_W-1:0] core_desc,
    input  wire        core_accept,
    input  wire        core_release,
    output wire        core_release_done,
    output wire [15:0] core_release_status,
    output wire        core_busy,

    // ---------------- core side: registered read port ----------------
    input  wire [GROUPS-1:0]      rd_en,
    input  wire [$clog2(K_STEPS)-1:0] rd_addr,
    output wire                   rd_valid,
    output wire [GROUPS*256-1:0]  weight_lo_hi,
    output wire [GROUPS*128-1:0]  scale_word
);

    localparam AW   = $clog2(K_STEPS);
    localparam SC   = (K_STEPS + 3) / 4;       // scale words per group

    // ------------------------------------------------------------------
    // storage
    // ------------------------------------------------------------------
    (* ram_style = "block" *) reg [127:0] w_lo [0:GROUPS*K_STEPS-1];
    (* ram_style = "block" *) reg [127:0] w_hi [0:GROUPS*K_STEPS-1];
    (* ram_style = "block" *) reg [127:0] sc   [0:GROUPS*SC-1];

    wire [AW-1:0] wr_word = wr_addr;

    always @(posedge clk_io) begin
        if (wr_en) begin
            if (!wr_kind) begin
                if (wr_half) w_hi[wr_grp*K_STEPS + wr_word] <= wr_data;
                else         w_lo[wr_grp*K_STEPS + wr_word] <= wr_data;
            end else begin
                sc[wr_grp*SC + (wr_addr >> 2)] <= wr_data;
            end
        end
    end

    reg                      rd_valid_r;
    reg [GROUPS*256-1:0]     wlo_r;
    reg [GROUPS*128-1:0]     scr_r;
    reg [GROUPS-1:0]         rd_en_r;
    reg [GROUPS*256-1:0]     wlo_g;
    reg [GROUPS*128-1:0]     scr_g;
    integer g;

    // The arrays are read every core cycle into a plain register bank: no
    // reset on the read side and no conditional wrapped around the read
    // itself, which is the shape that maps to block RAM.  With the old
    // "if (rd_en) read else zero" form inside the reset block each bank
    // expanded to ~11k distributed RAM primitives, which does not fit.
    //
    // The per-group enable is registered alongside the data and applied at
    // the output, so a group that was not selected still presents zeros one
    // cycle later -- exactly the cycle the old zeroing landed in.
    always @(posedge clk_core) begin
        for (g = 0; g < GROUPS; g = g + 1) begin
            wlo_r[g*256 +: 128]     <= w_lo[g*K_STEPS + rd_addr];
            wlo_r[g*256+128 +: 128] <= w_hi[g*K_STEPS + rd_addr];
            scr_r[g*128 +: 128]     <= sc[g*SC + (rd_addr >> 2)];
        end
    end

    always @(posedge clk_core or negedge rst_core_n) begin
        if (!rst_core_n) begin
            rd_valid_r <= 1'b0;
            rd_en_r    <= {GROUPS{1'b0}};
        end else begin
            rd_valid_r <= |rd_en;
            rd_en_r    <= rd_en;
        end
    end

    always @* begin
        for (g = 0; g < GROUPS; g = g + 1) begin
            wlo_g[g*256 +: 128]     = rd_en_r[g] ? wlo_r[g*256 +: 128] : 128'd0;
            wlo_g[g*256+128 +: 128] = rd_en_r[g] ? wlo_r[g*256+128 +: 128] : 128'd0;
            scr_g[g*128 +: 128]     = rd_en_r[g] ? scr_r[g*128 +: 128] : 128'd0;
        end
    end

    assign rd_valid     = rd_valid_r;
    assign weight_lo_hi = wlo_g;
    assign scale_word   = scr_g;

    // ------------------------------------------------------------------
    // CDC mailboxes
    // ------------------------------------------------------------------
    reg  [`NPU_WTILE_W-1:0] desc_hold;
    reg                     commit_start;
    wire                    commit_busy, commit_done;
    wire                    commit_dst_valid, commit_dst_ready;
    wire [`NPU_WTILE_W-1:0] commit_dst_data;

    cdc_handshake #(.WIDTH(`NPU_WTILE_W)) u_commit (
        .src_clk(clk_io), .src_rst_n(rst_io_n),
        .src_start(commit_start), .src_data(desc_hold),
        .src_busy(commit_busy), .src_done(commit_done),
        .dst_clk(clk_core), .dst_rst_n(rst_core_n),
        .dst_valid(commit_dst_valid), .dst_data(commit_dst_data),
        .dst_ready(commit_dst_ready)
    );

    reg  [`NPU_WREL_W-1:0] rel_payload;
    reg                    rel_start;
    wire                   rel_busy, rel_done;
    wire                   rel_dst_valid, rel_dst_ready;
    wire [`NPU_WREL_W-1:0] rel_dst_data;

    cdc_handshake #(.WIDTH(`NPU_WREL_W)) u_release (
        .src_clk(clk_core), .src_rst_n(rst_core_n),
        .src_start(rel_start), .src_data(rel_payload),
        .src_busy(rel_busy), .src_done(rel_done),
        .dst_clk(clk_io), .dst_rst_n(rst_io_n),
        .dst_valid(rel_dst_valid), .dst_data(rel_dst_data),
        .dst_ready(rel_dst_ready)
    );

    // ------------------------------------------------------------------
    // io ownership
    // ------------------------------------------------------------------
    localparam IO_FREE       = 3'd0;
    localparam IO_FILLING    = 3'd1;
    localparam IO_COMMITTING = 3'd2;
    localparam IO_CORE_OWNED = 3'd3;

    reg [2:0] io_state;
    reg       commit_done_r;

    assign io_free          = (io_state == IO_FREE);
    assign io_commit_done   = commit_done_r;
    assign io_commit_status = `NPU_ST_OK;
    assign rel_dst_ready    = (io_state == IO_CORE_OWNED);

    always @(posedge clk_io or negedge rst_io_n) begin
        if (!rst_io_n) begin
            io_state      <= IO_FREE;
            desc_hold     <= {`NPU_WTILE_W{1'b0}};
            commit_start  <= 1'b0;
            commit_done_r <= 1'b0;
        end else begin
            commit_start  <= 1'b0;
            commit_done_r <= 1'b0;

            case (io_state)
            IO_FREE: begin
                if (io_fill_start) io_state <= IO_FILLING;
            end
            IO_FILLING: begin
                if (io_abort) begin
                    io_state <= IO_FREE;
                end else if (io_publish && !commit_busy) begin
                    desc_hold    <= io_desc;
                    commit_start <= 1'b1;
                    io_state     <= IO_COMMITTING;
                end
            end
            IO_COMMITTING: begin
                if (commit_done) begin
                    commit_done_r <= 1'b1;
                    io_state      <= IO_CORE_OWNED;
                end
            end
            IO_CORE_OWNED: begin
                // four-phase release from the core; the mailbox completes the
                // ack round trip itself
                if (rel_dst_valid && rel_dst_ready)
                    io_state <= IO_FREE;
            end
            default: io_state <= IO_FREE;
            endcase
        end
    end

    // ------------------------------------------------------------------
    // core ownership
    // ------------------------------------------------------------------
    localparam CR_EMPTY     = 2'd0;
    localparam CR_READY     = 2'd1;
    localparam CR_READING   = 2'd2;
    localparam CR_RELEASING = 2'd3;

    reg [1:0]               core_state;
    reg [`NPU_WTILE_W-1:0]  desc_r;
    reg                     desc_valid_r;
    reg                     rel_done_r;

    assign core_desc_valid    = desc_valid_r;
    assign core_desc          = desc_r;
    assign core_busy          = (core_state == CR_READING) ||
                                (core_state == CR_RELEASING);
    assign core_release_done  = rel_done_r;
    assign core_release_status = `NPU_ST_OK;
    assign commit_dst_ready   = (core_state == CR_EMPTY);

    always @(posedge clk_core or negedge rst_core_n) begin
        if (!rst_core_n) begin
            core_state   <= CR_EMPTY;
            desc_r       <= {`NPU_WTILE_W{1'b0}};
            desc_valid_r <= 1'b0;
            rel_start    <= 1'b0;
            rel_done_r   <= 1'b0;
            rel_payload  <= {`NPU_WREL_W{1'b0}};
        end else begin
            rel_start  <= 1'b0;
            rel_done_r <= 1'b0;

            case (core_state)
            CR_EMPTY: begin
                if (commit_dst_valid && commit_dst_ready) begin
                    desc_r       <= commit_dst_data;
                    desc_valid_r <= 1'b1;
                    core_state   <= CR_READY;
                end
            end
            CR_READY: begin
                if (core_accept) begin
                    desc_valid_r <= 1'b0;
                    core_state   <= CR_READING;
                end
            end
            CR_READING: begin
                if (core_release && !rel_busy) begin
                    rel_payload <= {1'b0,
                                    `NPU_WTILE_JOB(desc_r),
                                    `NPU_WTILE_INDEX(desc_r)};
                    rel_start   <= 1'b1;
                    core_state  <= CR_RELEASING;
                end
            end
            CR_RELEASING: begin
                if (rel_done) begin
                    rel_done_r <= 1'b1;
                    core_state <= CR_EMPTY;
                end
            end
            default: core_state <= CR_EMPTY;
            endcase
        end
    end

endmodule

`default_nettype wire

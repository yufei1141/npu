//=============================================================================
// npu_top : platform-independent NPU peripheral
//
//   Follows docs/npu_integration/14_npu_top.md.
//
//   Domain split
//   ------------
//   core : command_frontend -> task_scheduler -> {dma, mpu, vpu, sfu}
//          memory_subsystem, result_writeback, weight_reader,
//          completion_manager, configuration registers, top FSM
//   io   : dma_engine io side, weight_loader, weight_bank write side
//
//   clk_io and clk_core are independent.  dma_engine and weight_bank each
//   contain their own CDC; the top only synchronizes reset release and the
//   io-ready indication.  No data crosses a clock boundary here without one
//   of those structures.
//
//   Reset
//   -----
//   reset_n is asynchronous; each domain shifts it out for three cycles, so
//   both protocol states leave reset inside their own clock domain.  BOOT
//   waits for the io ready indication to be visible in the core domain and
//   for platform_isolated, which the platform may only assert once every
//   previously accepted external transaction has really been terminated.
//
//   Top FSM
//   -------
//   RESET 0, BOOT 1, IDLE 2, RUN 3, DRAIN 4, FAULT 5.
//   IDLE allows configuration; RUN opens the command port; DRAIN closes it
//   and waits for the frontend queue, the scheduler, every engine, the
//   scratchpad and the weight banks to go idle.  FAULT is sticky until the
//   coordinated reset.  A completion record still queued is not a reason to
//   stay out of IDLE; the host drains it from IDLE.
//
//   Configuration (section 3 of doc 14)
//   -----------------------------------
//   Implemented: VERSION, STATUS, CAPABILITY, WIDTHS, LIMITS, FATAL_INFO,
//   COUNTER_SNAPSHOT, SNAPSHOT_STATUS, the per-buffer map registers and the
//   core-domain counters (core_cycles, cmd_accepted, cmd_committed,
//   fault_count); io-domain counters read 0 until the snapshot mailbox is
//   built.  The per-buffer diagnostic word at +0x18 reads 0: the reader and
//   writer counts live in modules that do not export them yet.
//   Map writes are only accepted in IDLE, an out-of-range or misaligned
//   access is rejected without modifying anything, and RUN requires a map
//   that passes the overlap / bank / alignment check.
//
//   SFU capability is a build-time parameter.  The FP32 datapath in
//   sfu/sfu is verified against a double precision reference (tb_sfu_slice:
//   EXP2 0.95, LOG2 0.50, RCP 1.36, RSQRT 1.52, SQRT 2.01, SIN 1.37,
//   COS 1.17, TANH 3.05 ULP), so all four adapter subops (EXP2, RCP, RSQRT,
//   SINCOS) and all three storage formats are enabled.  Clearing a bit still
//   rejects the corresponding task with UNSUPPORTED.
//=============================================================================
//=============================================================================
// 中文说明
//
//   平台无关的 NPU 外设顶层。实现依据 docs/npu_integration/14_npu_top.md。
//
//   时钟域划分：
//     core 域：command_frontend -> task_scheduler -> {dma, mpu, vpu, sfu}、
//              memory_subsystem、result_writeback、weight_reader、
//              completion_manager、配置寄存器、顶层状态机；
//     io   域：dma_engine 的外部侧、weight_loader、weight_bank 的写侧。
//   clk_io 与 clk_core 相互独立。dma_engine 和 weight_bank 各自内含 CDC，
//   顶层只负责同步复位释放和 io-ready 指示；**没有任何数据在缺少这两类结构
//   的情况下跨越时钟边界**。
//
//   复位：reset_n 是异步的，每个域各自把它移出三拍，让两侧的协议状态都在
//   自己的时钟域内离开复位。BOOT 状态要等到 io-ready 指示在 core 域可见、
//   并且 platform_isolated 已经置起——后者由平台保证"此前被接受的所有外部
//   事务都真正结束"，否则会在边界上丢写。
//
//   顶层状态机：RESET(0)、BOOT(1)、IDLE(2)、RUN(3)、DRAIN(4)、FAULT(5)。
//     - IDLE 允许写配置；
//     - RUN 打开命令端口；
//     - DRAIN 关闭命令端口，等待前端队列、调度器、所有引擎、scratchpad 和
//       权重 bank 全部空闲；
//     - FAULT 是粘滞的，只有协调复位才能清。
//   已经排队的完成记录不构成"不能回到 IDLE"的理由——主机在 IDLE 里照样可以
//   把它取走。
//
//   配置接口（doc 14 第 3 节）：已实现 VERSION、STATUS、CAPABILITY、WIDTHS、
//   LIMITS、FATAL_INFO、COUNTER_SNAPSHOT、SNAPSHOT_STATUS、每个 buffer 的映射
//   寄存器，以及 core 域计数器（core_cycles、cmd_accepted、cmd_committed、
//   fault_count）。io 域计数器在快照邮箱做出来之前读 0；+0x18 的诊断字读 0，
//   因为读写计数位于尚未导出这些计数的模块内部。
//   映射寄存器只在 IDLE 接受写入；越界或非对齐访问整条拒绝、不修改任何东西；
//   进入 RUN 要求映射通过重叠 / bank / 对齐检查。
//
//   SFU 能力是编译期参数。sfu/sfu 的 FP32 数据通路已经对照双精度参考验证过
//   （tb_sfu_slice：EXP2 0.95、LOG2 0.50、RCP 1.36、RSQRT 1.52、SQRT 2.01、
//   SIN 1.37、COS 1.17、TANH 3.05 ULP），所以四个 adapter subop（EXP2、RCP、
//   RSQRT、SINCOS）和三种存储格式全部打开。清掉某一位仍然会以 UNSUPPORTED
//   拒绝对应任务。
//=============================================================================
`timescale 1ns/1ps
`default_nettype none

`include "npu_defs.vh"

module npu_top #(
    parameter MEM_DATA_W   = 256,
    parameter IO_DATA_W    = 128,
    parameter EXT_WR_W     = 128,
    parameter EVENT_SLOTS  = 16,
    parameter RESULT_SLOTS = 4,
    parameter WBUF_K_STEPS = 256,
    parameter V_LANES      = 4,
    parameter BANKS        = 8,
    parameter BANK_DEPTH   = 1024,
    parameter GROUPS       = 8,
    parameter CMD_FIFO_DEPTH = 8,
    parameter [8:0] SFU_OPS = 9'h1FF,     // verified SFU functions, one per subop
    parameter [2:0] SFU_FORMATS = 3'b111  // verified storage formats
)(
    input  wire        clk_core,
    input  wire        clk_io,
    input  wire        reset_n,

    input  wire        platform_isolated,
    input  wire        run_req,

    // command stream (core)
    input  wire        cmd_valid,
    output wire        cmd_ready,
    input  wire [511:0] cmd_data,

    // host completion stream (core)
    output wire        cpl_valid,
    input  wire        cpl_ready,
    output wire [127:0] cpl_data,

    // configuration (core, one outstanding)
    input  wire        cfg_valid,
    output wire        cfg_ready,
    input  wire        cfg_write,
    input  wire [15:0] cfg_addr,
    input  wire [63:0] cfg_wdata,
    output wire        cfg_rsp_valid,
    input  wire        cfg_rsp_ready,
    output wire [63:0] cfg_rdata,
    output wire        cfg_error,

    output wire        irq,
    output wire [2:0]  top_state,
    output wire        fatal,
    output wire [15:0] fatal_code,

    // external memory read (io)
    output wire        rd_req_valid,
    input  wire        rd_req_ready,
    output wire [63:0] rd_req_addr,
    output wire [31:0] rd_req_bytes,
    output wire [15:0] rd_req_tag,
    input  wire        rd_valid,
    output wire        rd_ready,
    input  wire [IO_DATA_W-1:0] rd_data,
    input  wire [IO_DATA_W/8-1:0] rd_keep,
    input  wire        rd_last,
    input  wire [15:0] rd_tag,
    input  wire        rd_error,

    // external memory write (io)
    output wire        wr_req_valid,
    input  wire        wr_req_ready,
    output wire [63:0] wr_req_addr,
    output wire [31:0] wr_req_bytes,
    output wire [15:0] wr_req_tag,
    output wire        wr_valid,
    input  wire        wr_ready,
    output wire [EXT_WR_W-1:0] wr_data,
    output wire [EXT_WR_W/8-1:0] wr_keep,
    output wire        wr_last,
    output wire [15:0] wr_tag,
    input  wire        wr_rsp_valid,
    output wire        wr_rsp_ready,
    input  wire [15:0] wr_rsp_tag,
    input  wire        wr_rsp_error
);

    localparam DW    = MEM_DATA_W;
    localparam B     = MEM_DATA_W / 8;
    localparam REQ_W = `NPU_REQ_W(MEM_DATA_W, B);
    localparam RSP_W = `NPU_RSP_W(MEM_DATA_W);

    localparam S_RESET = 3'd0;
    localparam S_BOOT  = 3'd1;
    localparam S_IDLE  = 3'd2;
    localparam S_RUN   = 3'd3;
    localparam S_DRAIN = 3'd4;
    localparam S_FAULT = 3'd5;

    // ------------------------------------------------------------------
    // reset release per domain
    // ------------------------------------------------------------------
    reg [2:0] rsts_core, rsts_io;

    always @(posedge clk_core or negedge reset_n)
        if (!reset_n) rsts_core <= 3'b000;
        else          rsts_core <= {rsts_core[1:0], 1'b1};

    always @(posedge clk_io or negedge reset_n)
        if (!reset_n) rsts_io <= 3'b000;
        else          rsts_io <= {rsts_io[1:0], 1'b1};

    wire rst_core_n = rsts_core[2];
    wire rst_io_n   = rsts_io[2];

    // io ready seen from the core domain
    reg [1:0] io_ready_sync;
    always @(posedge clk_core or negedge rst_core_n)
        if (!rst_core_n) io_ready_sync <= 2'b00;
        else             io_ready_sync <= {io_ready_sync[0], rst_io_n};

    wire io_ready = io_ready_sync[1];

    // ------------------------------------------------------------------
    // top FSM
    // ------------------------------------------------------------------
    reg [2:0] state;
    assign top_state = state;

    // ------------------------------------------------------------------
    // configuration registers
    // ------------------------------------------------------------------
    reg [7:0]   cfg_en;
    reg [7:0]   cfg_bank [0:7];
    reg [31:0]  cfg_base [0:7];       // words
    reg [31:0]  cfg_size [0:7];       // bytes

    wire [7:0]  buf_enable = cfg_en;
    wire [63:0] buf_bank_flat;
    wire [255:0] buf_base_flat;
    wire [255:0] buf_size_flat;

    assign buf_bank_flat[8*0 +: 8] = cfg_bank[0];
    assign buf_bank_flat[8*1 +: 8] = cfg_bank[1];
    assign buf_bank_flat[8*2 +: 8] = cfg_bank[2];
    assign buf_bank_flat[8*3 +: 8] = cfg_bank[3];
    assign buf_bank_flat[8*4 +: 8] = cfg_bank[4];
    assign buf_bank_flat[8*5 +: 8] = cfg_bank[5];
    assign buf_bank_flat[8*6 +: 8] = cfg_bank[6];
    assign buf_bank_flat[8*7 +: 8] = cfg_bank[7];

    assign buf_base_flat[32*0 +: 32] = cfg_base[0];
    assign buf_base_flat[32*1 +: 32] = cfg_base[1];
    assign buf_base_flat[32*2 +: 32] = cfg_base[2];
    assign buf_base_flat[32*3 +: 32] = cfg_base[3];
    assign buf_base_flat[32*4 +: 32] = cfg_base[4];
    assign buf_base_flat[32*5 +: 32] = cfg_base[5];
    assign buf_base_flat[32*6 +: 32] = cfg_base[6];
    assign buf_base_flat[32*7 +: 32] = cfg_base[7];

    assign buf_size_flat[32*0 +: 32] = cfg_size[0];
    assign buf_size_flat[32*1 +: 32] = cfg_size[1];
    assign buf_size_flat[32*2 +: 32] = cfg_size[2];
    assign buf_size_flat[32*3 +: 32] = cfg_size[3];
    assign buf_size_flat[32*4 +: 32] = cfg_size[4];
    assign buf_size_flat[32*5 +: 32] = cfg_size[5];
    assign buf_size_flat[32*6 +: 32] = cfg_size[6];
    assign buf_size_flat[32*7 +: 32] = cfg_size[7];

    // map check: bank range, 32-byte granularity, no overlap inside a bank
    reg         map_ok;
    integer     bi, bj;
    reg [63:0]  bbase [0:7];
    reg [63:0]  bend  [0:7];
    reg [63:0]  bank_bytes;

    always @* begin
        bank_bytes = BANKS * BANK_DEPTH * 32;
        map_ok     = 1'b1;
        for (bi = 0; bi < 8; bi = bi + 1) begin
            bbase[bi] = {32'd0, cfg_base[bi]} << 5;
            bend[bi]  = bbase[bi] + {32'd0, cfg_size[bi]};
            if (cfg_en[bi]) begin
                if (cfg_bank[bi] >= BANKS)          map_ok = 1'b0;
                if (cfg_size[bi] == 32'd0)          map_ok = 1'b0;
                if (cfg_size[bi][4:0] != 5'd0)      map_ok = 1'b0;
                if (bend[bi] > bank_bytes)          map_ok = 1'b0;
            end
        end
        for (bi = 0; bi < 8; bi = bi + 1)
            for (bj = bi + 1; bj < 8; bj = bj + 1)
                if (cfg_en[bi] && cfg_en[bj] &&
                    (cfg_bank[bi] == cfg_bank[bj]) &&
                    (bbase[bi] < bend[bj]) && (bbase[bj] < bend[bi]))
                    map_ok = 1'b0;
    end

    // counters
    reg [63:0] cnt_core_cycles;
    reg [63:0] cnt_cpl_committed;
    reg [63:0] cnt_fault;
    reg [63:0] snap_generation;
    reg        snap_valid, snap_busy;

    // ------------------------------------------------------------------
    // stream / completion / scheduler wires
    // ------------------------------------------------------------------
    wire        accept_enable;
    wire        issue_valid, issue_ready;
    wire [511:0] issue_cmd;
    wire [15:0] issue_error;
    wire        issue_identity_error;
    wire        held_valid;
    wire [15:0] held_job;
    wire [31:0] fe_fifo_level;
    wire [63:0] fe_accepted, fe_rejected;

    wire        run_enable = (state == S_RUN) || (state == S_DRAIN);

    // ------------------------------------------------------------------
    // engine task channels
    // ------------------------------------------------------------------
    wire dma_task_valid, dma_task_ready, mpu_task_valid, mpu_task_ready;
    wire vpu_task_valid, vpu_task_ready, sfu_task_valid, sfu_task_ready;
    wire [511:0] dma_task_cmd, mpu_task_cmd, vpu_task_cmd, sfu_task_cmd;

    // engine events
    wire        dma_ev_valid, dma_ev_ready, mpu_ev_valid, mpu_ev_ready;
    wire        vpu_ev_valid, vpu_ev_ready, sfu_ev_valid, sfu_ev_ready;
    wire        adm_ev_valid, adm_ev_ready;
    wire [127:0] dma_ev_data, mpu_ev_data, vpu_ev_data, sfu_ev_data, adm_ev_data;

    // ------------------------------------------------------------------
    // scratchpad clients
    // ------------------------------------------------------------------
    wire [4:0]  c_req_valid;
    wire [4:0]  c_req_ready;
    wire [5*REQ_W-1:0] c_req;
    wire [4:0]  c_rsp_valid;
    wire [4:0]  c_rsp_ready;
    wire [5*RSP_W-1:0] c_rsp;

    wire        mpu_mem_req_valid, mpu_mem_req_ready;
    wire [REQ_W-1:0] mpu_mem_req;
    wire        mpu_mem_rsp_valid, mpu_mem_rsp_ready;
    wire [RSP_W-1:0] mpu_mem_rsp;

    wire        vpu_mem_req_valid, vpu_mem_req_ready;
    wire [REQ_W-1:0] vpu_mem_req;
    wire        vpu_mem_rsp_valid, vpu_mem_rsp_ready;
    wire [RSP_W-1:0] vpu_mem_rsp;

    wire        sfu_mem_req_valid, sfu_mem_req_ready;
    wire [REQ_W-1:0] sfu_mem_req;
    wire        sfu_mem_rsp_valid, sfu_mem_rsp_ready;
    wire [RSP_W-1:0] sfu_mem_rsp;

    wire        wb_mem_req_valid, wb_mem_req_ready;
    wire [REQ_W-1:0] wb_mem_req;
    wire        wb_mem_rsp_valid, wb_mem_rsp_ready;
    wire [RSP_W-1:0] wb_mem_rsp;

    wire        dma_mem_req_valid, dma_mem_req_ready;
    wire [REQ_W-1:0] dma_mem_req;
    wire        dma_mem_rsp_valid, dma_mem_rsp_ready;
    wire [RSP_W-1:0] dma_mem_rsp;

    assign c_req_valid = {dma_mem_req_valid, wb_mem_req_valid,
                          sfu_mem_req_valid, vpu_mem_req_valid,
                          mpu_mem_req_valid};
    assign c_rsp_ready = {dma_mem_rsp_ready, wb_mem_rsp_ready,
                          sfu_mem_rsp_ready, vpu_mem_rsp_ready,
                          mpu_mem_rsp_ready};

    assign c_req[0*REQ_W +: REQ_W] = mpu_mem_req;
    assign c_req[1*REQ_W +: REQ_W] = vpu_mem_req;
    assign c_req[2*REQ_W +: REQ_W] = sfu_mem_req;
    assign c_req[3*REQ_W +: REQ_W] = wb_mem_req;
    assign c_req[4*REQ_W +: REQ_W] = dma_mem_req;

    assign mpu_mem_req_ready = c_req_ready[0];
    assign vpu_mem_req_ready = c_req_ready[1];
    assign sfu_mem_req_ready = c_req_ready[2];
    assign wb_mem_req_ready  = c_req_ready[3];
    assign dma_mem_req_ready = c_req_ready[4];

    assign mpu_mem_rsp_valid = c_rsp_valid[0];
    assign vpu_mem_rsp_valid = c_rsp_valid[1];
    assign sfu_mem_rsp_valid = c_rsp_valid[2];
    assign wb_mem_rsp_valid  = c_rsp_valid[3];
    assign dma_mem_rsp_valid = c_rsp_valid[4];

    assign mpu_mem_rsp = c_rsp[0*RSP_W +: RSP_W];
    assign vpu_mem_rsp = c_rsp[1*RSP_W +: RSP_W];
    assign sfu_mem_rsp = c_rsp[2*RSP_W +: RSP_W];
    assign wb_mem_rsp  = c_rsp[3*RSP_W +: RSP_W];
    assign dma_mem_rsp = c_rsp[4*RSP_W +: RSP_W];

    wire [15:0] mem_outstanding;
    wire [BANKS-1:0] bank_busy;

    // ------------------------------------------------------------------
    // weight path
    // ------------------------------------------------------------------
    wire        mpu_wsel_valid, mpu_wsel_ready;
    wire [15:0] mpu_wsel_job, mpu_wsel_tile, mpu_wsel_s;
    wire [1:0]  mpu_wsel_mode;
    wire [3:0]  mpu_wsel_g;
    wire        rd_b_valid, rd_b_ready;
    wire [2303:0] rd_b_data;
    wire        rd_tile_done_valid, rd_tile_done_ready;
    wire [15:0] rd_tile_done_status;

    wire        wb_job_valid, wb_job_ready;
    wire [511:0] wb_job_cmd;
    wire        wb_done_valid, wb_done_ready;
    wire [15:0] wb_done_status;
    wire [31:0] wb_done_tiles;

    wire        mpu_acc_valid;
    wire [4095:0] mpu_acc_data;

    wire [15:0] wq_job, wq_tile, wq_s;
    wire [1:0]  wq_mode;
    wire [3:0]  wq_g;
    wire        wq_match, wq_fault;

    wire [1:0]  wbank_free;
    wire [1:0]  wl_fill_start, wl_publish, wl_abort;
    wire [`NPU_WTILE_W-1:0] wl_desc;
    wire [1:0]  wbank_commit_done;
    wire [31:0] wbank_commit_status;
    wire [1:0]  wl_wr_en;
    wire        wl_wr_kind, wl_wr_half;
    wire [2:0]  wl_wr_grp;
    wire [$clog2(WBUF_K_STEPS)-1:0] wl_wr_addr;
    wire [127:0] wl_wr_data;

    wire        wl_done_valid, wl_done_ready;
    wire [15:0] wl_done_status;

    wire [1:0]  wbank_desc_valid;
    wire [2*`NPU_WTILE_W-1:0] wbank_desc;
    wire [1:0]  wbank_accept, wbank_release, wbank_release_done;
    wire [31:0] wbank_release_status;
    wire [2*GROUPS-1:0] wbank_rd_en;
    wire [$clog2(WBUF_K_STEPS)-1:0] wbank_rd_addr;
    wire [1:0]  wbank_rd_valid;
    wire [2*GROUPS*256-1:0] wbank_wlo_hi;
    wire [2*GROUPS*128-1:0] wbank_scale;
    wire [1:0]  wbank_busy;

    // ------------------------------------------------------------------
    // faults
    // ------------------------------------------------------------------
    wire mpu_fatal, vpu_fatal, sfu_fatal, dma_fatal;
    wire rd_fatal, wb_fatal, sched_fatal, cpl_fatal;

    wire hw_fatal = mpu_fatal | vpu_fatal | sfu_fatal | dma_fatal |
                    rd_fatal | wb_fatal | cpl_fatal;
    wire fatal_all = hw_fatal | (state == S_FAULT);

    assign fatal = fatal_all;

    reg [15:0] fatal_code_r;
    assign fatal_code = fatal_code_r;

    // first failing block, latched once
    always @(posedge clk_core or negedge rst_core_n) begin
        if (!rst_core_n) begin
            fatal_code_r <= 16'd0;
        end else if (fatal_code_r == 16'd0) begin
            if (mpu_fatal)      fatal_code_r <= 16'h0001;
            else if (vpu_fatal) fatal_code_r <= 16'h0002;
            else if (sfu_fatal) fatal_code_r <= 16'h0003;
            else if (dma_fatal) fatal_code_r <= 16'h0004;
            else if (rd_fatal)  fatal_code_r <= 16'h0005;
            else if (wb_fatal)  fatal_code_r <= 16'h0006;
            else if (cpl_fatal) fatal_code_r <= 16'h0007;
        end
    end

    // ------------------------------------------------------------------
    // configuration access
    // ------------------------------------------------------------------
    reg        cfg_rsp_valid_r, cfg_error_r;
    reg [63:0] cfg_rdata_r;

    assign cfg_rsp_valid = cfg_rsp_valid_r;
    assign cfg_rdata     = cfg_rdata_r;
    assign cfg_error     = cfg_error_r;
    assign cfg_ready     = !cfg_rsp_valid_r && rst_core_n;

    wire cfg_req = cfg_valid && cfg_ready;

    // buffer register region: 0x0100 + 0x20*b, b < 8
    wire        cfg_is_buf   = (cfg_addr >= 16'h0100) && (cfg_addr < 16'h0200);
    wire [2:0]  cfg_buf_idx  = cfg_addr[7:5];
    wire [4:0]  cfg_buf_off  = cfg_addr[4:0];
    wire        cfg_is_cnt   = (cfg_addr >= 16'h1000) && (cfg_addr < 16'h1080);

    wire        cfg_align_ok = (cfg_addr[2:0] == 3'd0);

    // writing the map is only legal in IDLE with an empty submission path
    wire        cfg_buf_wr_ok = (state == S_IDLE) && (fe_fifo_level == 32'd0) &&
                                !issue_valid;

    reg         cfg_hit;
    reg  [63:0] cfg_rd;

    // [8:0] SFU functions (one bit per subop), [11:9] SFU formats, bit12
    // LAYOUT, bit13 ROTATE_PAIR, bit14 MASKED_REDUCE.  The VPU answers LAYOUT
    // (TRANSPOSE2D, PACK_W16), ROTATE_PAIR and masked REDUCE.
    wire [14:0] control_caps = {1'b1, 1'b1, 1'b1, SFU_FORMATS, SFU_OPS};

    // parameter words for the WIDTHS register
    localparam [15:0] P_IO_W   = IO_DATA_W;
    localparam [15:0] P_MEM_W  = MEM_DATA_W;
    localparam [15:0] P_EXTW_W = EXT_WR_W;
    localparam [15:0] P_KSTEPS = WBUF_K_STEPS;
    localparam [15:0] P_EVENTS = EVENT_SLOTS;
    localparam [15:0] P_RESULTS = RESULT_SLOTS;

    // counter read mux
    reg  [63:0] cnt_sel;
    always @* begin
        cnt_sel = 64'd0;
        if (cfg_addr[6:3] == 4'd0)      cnt_sel = cnt_core_cycles;      // i = 0
        else if (cfg_addr[6:3] == 4'd2) cnt_sel = fe_accepted;          // i = 2
        else if (cfg_addr[6:3] == 4'd4) cnt_sel = cnt_cpl_committed;    // i = 4
        else if (cfg_addr[6:3] == 4'd15) cnt_sel = cnt_fault;           // i = 15
    end

    always @* begin
        cfg_hit = 1'b1;
        cfg_rd  = 64'd0;
        if (!cfg_align_ok) begin
            cfg_hit = 1'b0;
        end else if (cfg_is_buf) begin
            case (cfg_buf_off)
            5'h00: cfg_rd = {48'd0, cfg_bank[cfg_buf_idx], 7'd0,
                             cfg_en[cfg_buf_idx]};
            5'h08: cfg_rd = {32'd0, cfg_base[cfg_buf_idx]};
            5'h10: cfg_rd = {32'd0, cfg_size[cfg_buf_idx]};
            5'h18: cfg_rd = 64'd0;                       // diagnostic, see header
            default: cfg_hit = 1'b0;
            endcase
        end else if (cfg_is_cnt) begin
            cfg_rd = cnt_sel;
        end else begin
            case (cfg_addr)
            16'h0000: cfg_rd = 64'h0000_0000_0001_0000;  // VERSION
            16'h0008: cfg_rd = {58'd0, io_ready, rst_core_n, fatal_all, state};
            16'h0010: cfg_rd = {49'd0,
                                control_caps};          // [14:12] LAYOUT/ROT/MASK
            16'h0018: cfg_rd = {P_KSTEPS, P_EXTW_W, P_MEM_W, P_IO_W};
            16'h0020: cfg_rd = {16'd0, P_EVENTS, P_RESULTS, 8'd8, 8'd4};
            16'h0028: cfg_rd = {32'd0, 16'd0, fatal_code_r};
            16'h0030: cfg_rd = 64'd0;                    // WO snapshot trigger
            16'h0038: cfg_rd = {snap_generation[31:0], 30'd0,
                                snap_valid, snap_busy};
            default: begin
                cfg_hit = 1'b0;
            end
            endcase
        end
    end

    // a write is refused when the address is unknown, when it targets a
    // read-only word, or when a reserved field of a writable word is set
    reg cfg_wr_ok;
    always @* begin
        cfg_wr_ok = 1'b1;
        if (!cfg_hit) begin
            cfg_wr_ok = 1'b0;
        end else if (cfg_is_buf) begin
            if (!cfg_buf_wr_ok) cfg_wr_ok = 1'b0;
            else case (cfg_buf_off)
            5'h00: if (cfg_wdata[63:16] != 48'd0 || cfg_wdata[7:1] != 7'd0)
                       cfg_wr_ok = 1'b0;
            5'h08: if (cfg_wdata[63:32] != 32'd0) cfg_wr_ok = 1'b0;
            5'h10: if (cfg_wdata[63:32] != 32'd0) cfg_wr_ok = 1'b0;
            default: cfg_wr_ok = 1'b0;
            endcase
        end else if (cfg_is_cnt) begin
            cfg_wr_ok = 1'b0;                             // read-only
        end else begin
            case (cfg_addr)
            16'h0030: if (cfg_wdata != 64'd1) cfg_wr_ok = 1'b0;
            16'h0000, 16'h0008, 16'h0010, 16'h0018,
            16'h0020, 16'h0028, 16'h0038: cfg_wr_ok = 1'b0;
            default: cfg_wr_ok = 1'b0;
            endcase
        end
    end

    always @(posedge clk_core or negedge rst_core_n) begin
        if (!rst_core_n) begin
            cfg_rsp_valid_r <= 1'b0;
            cfg_error_r     <= 1'b0;
            cfg_rdata_r     <= 64'd0;
            cfg_en          <= 8'h00;
            cfg_bank[0] <= 8'd0;  cfg_bank[1] <= 8'd0;
            cfg_bank[2] <= 8'd0;  cfg_bank[3] <= 8'd0;
            cfg_bank[4] <= 8'd0;  cfg_bank[5] <= 8'd0;
            cfg_bank[6] <= 8'd0;  cfg_bank[7] <= 8'd0;
            cfg_base[0] <= 32'd0; cfg_base[1] <= 32'd0;
            cfg_base[2] <= 32'd0; cfg_base[3] <= 32'd0;
            cfg_base[4] <= 32'd0; cfg_base[5] <= 32'd0;
            cfg_base[6] <= 32'd0; cfg_base[7] <= 32'd0;
            cfg_size[0] <= 32'd0; cfg_size[1] <= 32'd0;
            cfg_size[2] <= 32'd0; cfg_size[3] <= 32'd0;
            cfg_size[4] <= 32'd0; cfg_size[5] <= 32'd0;
            cfg_size[6] <= 32'd0; cfg_size[7] <= 32'd0;
            snap_generation <= 64'd0;
            snap_valid      <= 1'b0;
            snap_busy       <= 1'b0;
        end else begin
            if (cfg_req) begin
                cfg_rsp_valid_r <= 1'b1;
                cfg_rdata_r     <= cfg_rd;
                cfg_error_r     <= cfg_write ? !cfg_wr_ok : !cfg_hit;

                if (cfg_write && cfg_wr_ok) begin
                    if (cfg_is_buf) begin
                        case (cfg_buf_off)
                        5'h00: begin
                            cfg_en[cfg_buf_idx]   <= cfg_wdata[0];
                            cfg_bank[cfg_buf_idx] <= cfg_wdata[15:8];
                        end
                        5'h08: cfg_base[cfg_buf_idx] <= cfg_wdata[31:0];
                        5'h10: cfg_size[cfg_buf_idx] <= cfg_wdata[31:0];
                        default: begin
                        end
                        endcase
                    end else if (cfg_addr == 16'h0030) begin
                        // io counters are not built yet: the snapshot is
                        // recorded so the host sees a generation change
                        snap_generation <= snap_generation + 64'd1;
                        snap_valid      <= 1'b1;
                        snap_busy       <= 1'b1;
                    end
                end
            end else if (cfg_rsp_valid_r && cfg_rsp_ready) begin
                cfg_rsp_valid_r <= 1'b0;
            end

            if (snap_busy) begin
                snap_busy  <= 1'b0;
                snap_valid <= 1'b1;
            end
        end
    end

    // ------------------------------------------------------------------
    // top FSM and counters
    // ------------------------------------------------------------------
    assign accept_enable = (state == S_RUN) && !fatal_all;

    wire engines_idle = mpu_task_ready && vpu_task_ready &&
                        sfu_task_ready && dma_task_ready &&
                        !mpu_ev_valid && !vpu_ev_valid &&
                        !sfu_ev_valid && !dma_ev_valid;

    wire drain_done = (fe_fifo_level == 32'd0) && !issue_valid &&
                      engines_idle && (mem_outstanding == 16'd0) &&
                      (wbank_busy == 2'b00) && !wb_job_valid && !wb_done_valid;

    always @(posedge clk_core or negedge rst_core_n) begin
        if (!rst_core_n) begin
            state           <= S_RESET;
            cnt_core_cycles <= 64'd0;
            cnt_cpl_committed <= 64'd0;
            cnt_fault       <= 64'd0;
        end else begin
            if (cnt_core_cycles != 64'hFFFF_FFFF_FFFF_FFFF)
                cnt_core_cycles <= cnt_core_cycles + 64'd1;
            if (cpl_valid && cpl_ready &&
                (cnt_cpl_committed != 64'hFFFF_FFFF_FFFF_FFFF))
                cnt_cpl_committed <= cnt_cpl_committed + 64'd1;
            if (hw_fatal && (cnt_fault != 64'hFFFF_FFFF_FFFF_FFFF))
                cnt_fault <= cnt_fault + 64'd1;

            case (state)
            S_RESET: state <= S_BOOT;
            S_BOOT: begin
                if (io_ready && platform_isolated) state <= S_IDLE;
            end
            S_IDLE: begin
                if (run_req && map_ok && !fatal_all) state <= S_RUN;
            end
            S_RUN: begin
                if (fatal_all)      state <= S_FAULT;
                else if (!run_req)  state <= S_DRAIN;
            end
            S_DRAIN: begin
                if (fatal_all)      state <= S_FAULT;
                else if (drain_done) state <= S_IDLE;
            end
            S_FAULT: state <= S_FAULT;
            default: state <= S_RESET;
            endcase
        end
    end

    // ------------------------------------------------------------------
    // command frontend
    // ------------------------------------------------------------------
    command_frontend #(
        .EVENT_SLOTS(EVENT_SLOTS), .RESULT_SLOTS(RESULT_SLOTS),
        .WBUF_K_STEPS(WBUF_K_STEPS), .MEM_DATA_W(MEM_DATA_W),
        .V_LANES(V_LANES), .CMD_FIFO_DEPTH(CMD_FIFO_DEPTH)
    ) u_frontend (
        .clk_core(clk_core), .rst_core_n(rst_core_n),
        .accept_enable(accept_enable),
        .cmd_valid(cmd_valid), .cmd_ready(cmd_ready), .cmd_data(cmd_data),
        .issue_valid(issue_valid), .issue_ready(issue_ready),
        .issue_cmd(issue_cmd), .issue_error(issue_error),
        .issue_identity_error(issue_identity_error),
        .held_valid(held_valid), .held_job(held_job),
        .fifo_level(fe_fifo_level),
        .accepted_count(fe_accepted), .rejected_count(fe_rejected)
    );

    // ------------------------------------------------------------------
    // scheduler
    // ------------------------------------------------------------------
    wire [7:0]  lookup_slot, alloc_slot, probe_slot, admin_target_slot;
    wire [15:0] lookup_job, alloc_job, id_query_job, admin_target_job;
    wire [2:0]  lookup_state;
    wire        alloc_valid, alloc_ready, alloc_irq;
    wire [7:0]  alloc_opcode;
    wire        probe_free, id_busy;
    wire        admin_valid, admin_ready, admin_irq;
    wire [127:0] admin_event;
    wire [15:0] admin_status;
    wire [1:0]  admin_action;
    wire        retire_valid;
    wire [127:0] retire_event;

    task_scheduler #(.EVENT_SLOTS(EVENT_SLOTS), .MEM_DATA_W(MEM_DATA_W))
    u_sched (
        .clk_core(clk_core), .rst_core_n(rst_core_n), .run_enable(run_enable),
        .issue_valid(issue_valid), .issue_ready(issue_ready),
        .issue_cmd(issue_cmd), .issue_error(issue_error),
        .issue_identity_error(issue_identity_error),
        .held_valid(held_valid), .held_job(held_job),
        .dma_task_valid(dma_task_valid), .dma_task_ready(dma_task_ready),
        .dma_task_cmd(dma_task_cmd),
        .mpu_task_valid(mpu_task_valid), .mpu_task_ready(mpu_task_ready),
        .mpu_task_cmd(mpu_task_cmd),
        .vpu_task_valid(vpu_task_valid), .vpu_task_ready(vpu_task_ready),
        .vpu_task_cmd(vpu_task_cmd),
        .sfu_task_valid(sfu_task_valid), .sfu_task_ready(sfu_task_ready),
        .sfu_task_cmd(sfu_task_cmd),
        .lookup_slot(lookup_slot), .lookup_job(lookup_job),
        .lookup_state(lookup_state),
        .alloc_valid(alloc_valid), .alloc_ready(alloc_ready),
        .alloc_slot(alloc_slot), .alloc_job(alloc_job),
        .alloc_opcode(alloc_opcode), .alloc_irq(alloc_irq),
        .slot_probe_slot(probe_slot), .slot_probe_free(probe_free),
        .id_query_job(id_query_job), .id_query_busy(id_busy),
        .admin_valid(admin_valid), .admin_ready(admin_ready),
        .admin_event(admin_event), .admin_status(admin_status),
        .admin_irq(admin_irq), .admin_action(admin_action),
        .admin_target_slot(admin_target_slot),
        .admin_target_job(admin_target_job),
        .retire_valid(retire_valid), .retire_event(retire_event),
        .weight_query_job(wq_job), .weight_query_tile(wq_tile),
        .weight_query_mode(wq_mode), .weight_query_g(wq_g),
        .weight_query_s(wq_s),
        .weight_query_match(wq_match), .weight_query_fault(wq_fault),
        .fault_in(hw_fatal), .fatal(sched_fatal),
        .buf_enable(buf_enable), .buf_bank_flat(buf_bank_flat),
        .buf_base_flat(buf_base_flat), .buf_size_flat(buf_size_flat),
        .mem_outstanding_zero(mem_outstanding == 16'd0)
    );

    // ------------------------------------------------------------------
    // completion manager
    // ------------------------------------------------------------------
    wire [4:0] src_valid = {adm_ev_valid, sfu_ev_valid, vpu_ev_valid,
                            mpu_ev_valid, dma_ev_valid};
    wire [4:0] src_ready;
    wire [5*128-1:0] src_event;

    assign src_event[0*128 +: 128] = dma_ev_data;
    assign src_event[1*128 +: 128] = mpu_ev_data;
    assign src_event[2*128 +: 128] = vpu_ev_data;
    assign src_event[3*128 +: 128] = sfu_ev_data;
    assign src_event[4*128 +: 128] = admin_event;

    assign dma_ev_ready = src_ready[0];
    assign mpu_ev_ready = src_ready[1];
    assign vpu_ev_ready = src_ready[2];
    assign sfu_ev_ready = src_ready[3];
    assign adm_ev_valid = admin_valid;
    assign admin_ready  = src_ready[4];

    completion_manager #(.EVENT_SLOTS(EVENT_SLOTS), .CPL_DEPTH(8))
    u_cpl (
        .clk_core(clk_core), .rst_core_n(rst_core_n),
        .src_valid(src_valid), .src_ready(src_ready), .src_event(src_event),
        .alloc_valid(alloc_valid), .alloc_ready(alloc_ready),
        .alloc_slot(alloc_slot), .alloc_job(alloc_job),
        .alloc_opcode(alloc_opcode), .alloc_irq(alloc_irq),
        .lookup_slot(lookup_slot), .lookup_job(lookup_job),
        .lookup_state(lookup_state),
        .slot_probe_slot(probe_slot), .slot_probe_free(probe_free),
        .id_query_job(id_query_job), .id_query_busy(id_busy),
        .admin_action(admin_action), .admin_target_slot(admin_target_slot),
        .admin_target_job(admin_target_job), .admin_irq(admin_irq),
        .cpl_valid(cpl_valid), .cpl_ready(cpl_ready), .cpl_data(cpl_data),
        .retire_valid(retire_valid), .retire_event(retire_event),
        .irq(irq), .fatal_in(hw_fatal), .fatal_latched(cpl_fatal),
        .cpl_level(), .active_slots()
    );

    // ------------------------------------------------------------------
    // scratchpad
    // ------------------------------------------------------------------
    memory_subsystem #(
        .MEM_DATA_W(MEM_DATA_W), .BANKS(BANKS),
        .BANK_DEPTH(BANK_DEPTH), .L_RAM(1)
    ) u_mem (
        .clk_core(clk_core), .rst_core_n(rst_core_n),
        .buf_enable(buf_enable), .buf_bank_flat(buf_bank_flat),
        .buf_base_flat(buf_base_flat), .buf_size_flat(buf_size_flat),
        .c_req_valid(c_req_valid), .c_req_ready(c_req_ready), .c_req(c_req),
        .c_rsp_valid(c_rsp_valid), .c_rsp_ready(c_rsp_ready), .c_rsp(c_rsp),
        .outstanding_count(mem_outstanding),
        .access_fault_valid(), .access_fault_info(),
        .bank_busy(bank_busy)
    );

    // ------------------------------------------------------------------
    // DMA
    // ------------------------------------------------------------------
    wire dma_wload_valid, dma_wload_ready;
    wire [511:0] dma_wload_cmd;
    wire dma_wdata_valid, dma_wdata_ready;
    wire [127:0] dma_wdata;
    wire dma_wdata_last, dma_wdata_error;

    dma_engine #(.MEM_DATA_W(MEM_DATA_W), .EXT_WR_W(EXT_WR_W))
    u_dma (
        .clk_core(clk_core), .rst_core_n(rst_core_n),
        .clk_io(clk_io), .rst_io_n(rst_io_n),
        .task_valid(dma_task_valid), .task_ready(dma_task_ready),
        .task_cmd(dma_task_cmd),
        .event_valid(dma_ev_valid), .event_ready(dma_ev_ready),
        .event_data(dma_ev_data),
        .mem_req_valid(dma_mem_req_valid), .mem_req_ready(dma_mem_req_ready),
        .mem_req(dma_mem_req),
        .mem_rsp_valid(dma_mem_rsp_valid), .mem_rsp_ready(dma_mem_rsp_ready),
        .mem_rsp(dma_mem_rsp),
        .rd_req_valid(rd_req_valid), .rd_req_ready(rd_req_ready),
        .rd_req_addr(rd_req_addr), .rd_req_bytes(rd_req_bytes),
        .rd_req_tag(rd_req_tag),
        .rd_valid(rd_valid), .rd_ready(rd_ready), .rd_data(rd_data),
        .rd_keep(rd_keep), .rd_last(rd_last), .rd_tag(rd_tag),
        .rd_error(rd_error),
        .wr_req_valid(wr_req_valid), .wr_req_ready(wr_req_ready),
        .wr_req_addr(wr_req_addr), .wr_req_bytes(wr_req_bytes),
        .wr_req_tag(wr_req_tag),
        .wr_valid(wr_valid), .wr_ready(wr_ready), .wr_data(wr_data),
        .wr_keep(wr_keep), .wr_last(wr_last), .wr_tag(wr_tag),
        .wr_rsp_valid(wr_rsp_valid), .wr_rsp_ready(wr_rsp_ready),
        .wr_rsp_tag(wr_rsp_tag), .wr_rsp_error(wr_rsp_error),
        .wload_valid(dma_wload_valid), .wload_ready(dma_wload_ready),
        .wload_cmd(dma_wload_cmd),
        .wdata_valid(dma_wdata_valid), .wdata_ready(dma_wdata_ready),
        .wdata(dma_wdata), .wdata_last(dma_wdata_last),
        .wdata_error(dma_wdata_error),
        .wload_done_valid(wl_done_valid), .wload_done_ready(wl_done_ready),
        .wload_status(wl_done_status),
        .fatal(dma_fatal)
    );

    // ------------------------------------------------------------------
    // weight path
    // ------------------------------------------------------------------
    weight_loader #(.GROUPS(GROUPS), .K_STEPS(WBUF_K_STEPS))
    u_wload (
        .clk_io(clk_io), .rst_io_n(rst_io_n),
        .wload_valid(dma_wload_valid), .wload_ready(dma_wload_ready),
        .wload_cmd(dma_wload_cmd),
        .data_valid(dma_wdata_valid), .data_ready(dma_wdata_ready),
        .data(dma_wdata), .data_last(dma_wdata_last),
        .data_error(dma_wdata_error),
        .done_valid(wl_done_valid), .done_ready(wl_done_ready),
        .done_status(wl_done_status),
        .bank_free(wbank_free),
        .bank_fill_start(wl_fill_start), .bank_publish(wl_publish),
        .bank_abort(wl_abort), .bank_desc(wl_desc),
        .bank_commit_done(wbank_commit_done),
        .bank_commit_status(wbank_commit_status),
        .bank_wr_en(wl_wr_en), .wr_kind(wl_wr_kind), .wr_grp(wl_wr_grp),
        .wr_half(wl_wr_half), .wr_addr(wl_wr_addr), .wr_data(wl_wr_data)
    );

    weight_bank #(.GROUPS(GROUPS), .K_STEPS(WBUF_K_STEPS)) u_wbank0 (
        .clk_io(clk_io), .rst_io_n(rst_io_n),
        .clk_core(clk_core), .rst_core_n(rst_core_n),
        .io_free(wbank_free[0]),
        .io_fill_start(wl_fill_start[0]), .io_publish(wl_publish[0]),
        .io_abort(wl_abort[0]), .io_desc(wl_desc),
        .io_commit_done(wbank_commit_done[0]),
        .io_commit_status(wbank_commit_status[15:0]),
        .wr_en(wl_wr_en[0]), .wr_kind(wl_wr_kind), .wr_grp(wl_wr_grp),
        .wr_half(wl_wr_half), .wr_addr(wl_wr_addr), .wr_data(wl_wr_data),
        .core_desc_valid(wbank_desc_valid[0]),
        .core_desc(wbank_desc[0*`NPU_WTILE_W +: `NPU_WTILE_W]),
        .core_accept(wbank_accept[0]), .core_release(wbank_release[0]),
        .core_release_done(wbank_release_done[0]),
        .core_release_status(wbank_release_status[15:0]),
        .core_busy(wbank_busy[0]),
        .rd_en(wbank_rd_en[0*GROUPS +: GROUPS]), .rd_addr(wbank_rd_addr),
        .rd_valid(wbank_rd_valid[0]),
        .weight_lo_hi(wbank_wlo_hi[0*GROUPS*256 +: GROUPS*256]),
        .scale_word(wbank_scale[0*GROUPS*128 +: GROUPS*128])
    );

    weight_bank #(.GROUPS(GROUPS), .K_STEPS(WBUF_K_STEPS)) u_wbank1 (
        .clk_io(clk_io), .rst_io_n(rst_io_n),
        .clk_core(clk_core), .rst_core_n(rst_core_n),
        .io_free(wbank_free[1]),
        .io_fill_start(wl_fill_start[1]), .io_publish(wl_publish[1]),
        .io_abort(wl_abort[1]), .io_desc(wl_desc),
        .io_commit_done(wbank_commit_done[1]),
        .io_commit_status(wbank_commit_status[31:16]),
        .wr_en(wl_wr_en[1]), .wr_kind(wl_wr_kind), .wr_grp(wl_wr_grp),
        .wr_half(wl_wr_half), .wr_addr(wl_wr_addr), .wr_data(wl_wr_data),
        .core_desc_valid(wbank_desc_valid[1]),
        .core_desc(wbank_desc[1*`NPU_WTILE_W +: `NPU_WTILE_W]),
        .core_accept(wbank_accept[1]), .core_release(wbank_release[1]),
        .core_release_done(wbank_release_done[1]),
        .core_release_status(wbank_release_status[31:16]),
        .core_busy(wbank_busy[1]),
        .rd_en(wbank_rd_en[1*GROUPS +: GROUPS]), .rd_addr(wbank_rd_addr),
        .rd_valid(wbank_rd_valid[1]),
        .weight_lo_hi(wbank_wlo_hi[1*GROUPS*256 +: GROUPS*256]),
        .scale_word(wbank_scale[1*GROUPS*128 +: GROUPS*128])
    );

    weight_reader #(.GROUPS(GROUPS), .K_STEPS(WBUF_K_STEPS))
    u_wreader (
        .clk_core(clk_core), .rst_core_n(rst_core_n),
        .select_valid(mpu_wsel_valid), .select_ready(mpu_wsel_ready),
        .select_job(mpu_wsel_job), .select_tile(mpu_wsel_tile),
        .select_mode(mpu_wsel_mode), .select_g(mpu_wsel_g),
        .select_s(mpu_wsel_s),
        .b_valid(rd_b_valid), .b_ready(rd_b_ready), .b_data(rd_b_data),
        .b_step(), .b_last(),
        .tile_done_valid(rd_tile_done_valid),
        .tile_done_ready(rd_tile_done_ready),
        .tile_done_status(rd_tile_done_status),
        .query_job(wq_job), .query_tile(wq_tile), .query_mode(wq_mode),
        .query_g(wq_g), .query_s(wq_s),
        .query_match(wq_match), .query_fault(wq_fault),
        .bank_desc_valid(wbank_desc_valid), .bank_desc(wbank_desc),
        .bank_accept(wbank_accept), .bank_release(wbank_release),
        .bank_release_done(wbank_release_done),
        .bank_release_status(wbank_release_status),
        .bank_rd_en(wbank_rd_en), .bank_rd_addr(wbank_rd_addr),
        .bank_rd_valid(wbank_rd_valid), .bank_wlo_hi(wbank_wlo_hi),
        .bank_scale(wbank_scale),
        .fatal(rd_fatal)
    );

    // ------------------------------------------------------------------
    // MPU
    // ------------------------------------------------------------------
    mpu_engine #(
        .MEM_DATA_W(MEM_DATA_W), .RESULT_SLOTS(RESULT_SLOTS),
        .WBUF_K_STEPS(WBUF_K_STEPS)
    ) u_mpu (
        .clk_core(clk_core), .rst_core_n(rst_core_n),
        .task_valid(mpu_task_valid), .task_ready(mpu_task_ready),
        .task_cmd(mpu_task_cmd),
        .event_valid(mpu_ev_valid), .event_ready(mpu_ev_ready),
        .event_data(mpu_ev_data),
        .mem_req_valid(mpu_mem_req_valid), .mem_req_ready(mpu_mem_req_ready),
        .mem_req(mpu_mem_req),
        .mem_rsp_valid(mpu_mem_rsp_valid), .mem_rsp_ready(mpu_mem_rsp_ready),
        .mem_rsp(mpu_mem_rsp),
        .w_select_valid(mpu_wsel_valid), .w_select_ready(mpu_wsel_ready),
        .w_select_job(mpu_wsel_job), .w_select_tile(mpu_wsel_tile),
        .w_select_mode(mpu_wsel_mode), .w_select_g(mpu_wsel_g),
        .w_select_s(mpu_wsel_s),
        .w_b_valid(rd_b_valid), .w_b_ready(rd_b_ready), .w_b_data(rd_b_data),
        .w_tile_done_valid(rd_tile_done_valid),
        .w_tile_done_ready(rd_tile_done_ready),
        .w_tile_done_status(rd_tile_done_status),
        .wb_job_valid(wb_job_valid), .wb_job_ready(wb_job_ready),
        .wb_job_cmd(wb_job_cmd),
        .wb_done_valid(wb_done_valid), .wb_done_ready(wb_done_ready),
        .wb_done_status(wb_done_status), .wb_done_tiles(wb_done_tiles),
        .acc_valid(mpu_acc_valid), .acc_data(mpu_acc_data),
        .fatal(mpu_fatal)
    );

    // ------------------------------------------------------------------
    // result writeback
    // ------------------------------------------------------------------
    result_writeback #(.MEM_DATA_W(MEM_DATA_W), .RESULT_SLOTS(RESULT_SLOTS))
    u_wb (
        .clk_core(clk_core), .rst_core_n(rst_core_n),
        .job_valid(wb_job_valid), .job_ready(wb_job_ready),
        .job_cmd(wb_job_cmd),
        .acc_valid(mpu_acc_valid), .acc_data(mpu_acc_data),
        .mem_req_valid(wb_mem_req_valid), .mem_req_ready(wb_mem_req_ready),
        .mem_req(wb_mem_req),
        .mem_rsp_valid(wb_mem_rsp_valid), .mem_rsp_ready(wb_mem_rsp_ready),
        .mem_rsp(wb_mem_rsp),
        .done_valid(wb_done_valid), .done_ready(wb_done_ready),
        .done_status(wb_done_status), .done_tiles(wb_done_tiles),
        .reserved_count(), .occupied_count(),
        .fatal(wb_fatal)
    );

    // ------------------------------------------------------------------
    // VPU
    // ------------------------------------------------------------------
    wire [7:0] vpu_fp_exception;

    vector_engine #(.MEM_DATA_W(MEM_DATA_W), .MAX_K_STEPS(WBUF_K_STEPS))
    u_vpu (
        .clk_core(clk_core), .rst_core_n(rst_core_n),
        .task_valid(vpu_task_valid), .task_ready(vpu_task_ready),
        .task_cmd(vpu_task_cmd),
        .event_valid(vpu_ev_valid), .event_ready(vpu_ev_ready),
        .event_data(vpu_ev_data),
        .mem_req_valid(vpu_mem_req_valid), .mem_req_ready(vpu_mem_req_ready),
        .mem_req(vpu_mem_req),
        .mem_rsp_valid(vpu_mem_rsp_valid), .mem_rsp_ready(vpu_mem_rsp_ready),
        .mem_rsp(vpu_mem_rsp),
        .fp_exception(vpu_fp_exception),
        .fatal(vpu_fatal)
    );

    // ------------------------------------------------------------------
    // SFU
    // ------------------------------------------------------------------
    wire         sfu_in_valid, sfu_in_ready;
    wire [3:0]   sfu_op;
    wire [1:0]   sfu_fmt_in, sfu_fmt_out, sfu_mode;
    wire [255:0] sfu_din;
    wire         sfu_out_valid, sfu_out_ready;
    wire [255:0] sfu_dout;
    wire [7:0]   sfu_status;
    wire         sfu_flush;

    sfu_adapter #(.MEM_DATA_W(MEM_DATA_W), .CORE_LANES(8)) u_sfu (
        .clk_core(clk_core), .rst_core_n(rst_core_n),
        .task_valid(sfu_task_valid), .task_ready(sfu_task_ready),
        .task_cmd(sfu_task_cmd),
        .event_valid(sfu_ev_valid), .event_ready(sfu_ev_ready),
        .event_data(sfu_ev_data),
        .mem_req_valid(sfu_mem_req_valid), .mem_req_ready(sfu_mem_req_ready),
        .mem_req(sfu_mem_req),
        .mem_rsp_valid(sfu_mem_rsp_valid), .mem_rsp_ready(sfu_mem_rsp_ready),
        .mem_rsp(sfu_mem_rsp),
        .capability_ops(SFU_OPS), .capability_formats(SFU_FORMATS),
        .core_in_valid(sfu_in_valid), .core_in_ready(sfu_in_ready),
        .core_op(sfu_op),
        .core_fmt_in(sfu_fmt_in), .core_fmt_out(sfu_fmt_out),
        .core_mode(sfu_mode), .core_din(sfu_din),
        .core_out_valid(sfu_out_valid), .core_out_ready(sfu_out_ready),
        .core_dout(sfu_dout), .core_status(sfu_status),
        .core_flush(sfu_flush),
        .fatal(sfu_fatal)
    );

    // the existing SFU core is instantiated unchanged
    sfu_top #(.WIDTH(8), .DATA_W(32)) u_sfu_core (
        .clk(clk_core), .rst_n(rst_core_n), .flush(sfu_flush),
        .in_valid(sfu_in_valid), .in_ready(sfu_in_ready),
        .op(sfu_op), .fmt_in(sfu_fmt_in), .fmt_out(sfu_fmt_out),
        .mode(sfu_mode), .din(sfu_din),
        .out_valid(sfu_out_valid), .out_ready(sfu_out_ready),
        .dout(sfu_dout), .status(sfu_status)
    );

endmodule

`default_nettype wire

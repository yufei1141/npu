//=============================================================================
// completion_manager : event slots, dependency lookup and CPL FIFO
//
//   Follows docs/npu_integration/13_completion_and_fault.md.
//
//   Sources (fixed order): 0 DMA, 1 MPU, 2 VPU, 3 SFU, 4 ADMIN.
//
//   A commit is atomic: source event accepted, CPL record enqueued, slot state
//   updated to OK/FAILED, retire pulse broadcast, and (for a RELEASE admin
//   transaction) the target slot cleared.  All of those happen on the same
//   clock edge.  The retire broadcast is combinational with grant_accept so
//   the scheduler releases resources on that same edge.
//
//   CPL backpressure does not lose an event: a source is not granted until
//   there is a queue entry.  Invalid events (slot mismatch, duplicate
//   completion, unknown slot) are consumed and latch fatal instead of being
//   reported as successful work.
//
//   EVENT_RELEASE is an ADMIN record with event_slot=0xFF.  Its target is
//   validated here: target must be a completed (OK/FAILED) slot owned by the
//   requested job.  A running target temporarily blocks the admin source; a
//   free/mismatched target is accepted and reported as BAD_FIELD without
//   clearing anything.
//=============================================================================
//=============================================================================
// 中文说明
//
//   事件槽（event slot）、依赖查询和完成队列（CPL FIFO）的实现。这是"异步
//   提交"的收口：任务跑完后由这里统一记账、生成 128 位完成记录、广播 retire；
//   实现依据 docs/npu_integration/13_completion_and_fault.md。
//
//   完成来源固定顺序：0 DMA、1 MPU、2 VPU、3 SFU、4 ADMIN。顺序是硬连线的
//   优先级，不表示执行顺序。
//
//   提交（commit）是一个原子动作，全部发生在同一个时钟沿：
//     - 接收来源的完成事件；
//     - 把完成记录压入 CPL 队列；
//     - 把槽状态更新为 OK / FAILED；
//     - 广播 retire 脉冲，调度器在同一拍据此释放 buffer 锁与引擎占用；
//     - 如果是 ADMIN 的 RELEASE 事务，还要清掉目标槽。
//   retire 与 grant_accept 是组合关系，这正是"释放资源和记账同拍完成"的
//   关键，否则会多出一拍的空转窗口。
//
//   反压与错误：
//     - CPL 反压不会丢事件：队列没有空位时干脆不 grant，来源继续等待；
//     - 非法事件（槽号不匹配、重复完成、未知槽号）会被消费并锁存 fatal，
//       绝不伪装成一次成功的任务；
//     - EVENT_RELEASE 是 event_slot=0xFF 的 ADMIN 记录，目标由本模块校验：
//       目标必须是已完成（OK/FAILED）且属于请求 job 的槽。目标还在运行时
//       暂时阻塞 ADMIN 来源；目标空闲或 job 不匹配则接受并以 BAD_FIELD 上报，
//       不做任何清除。
//=============================================================================
`timescale 1ns/1ps
`default_nettype none

`include "npu_defs.vh"

module completion_manager #(
    parameter EVENT_SLOTS = 16,
    parameter CPL_DEPTH   = 8
)(
    input  wire                 clk_core,
    input  wire                 rst_core_n,

    // five event sources
    input  wire [4:0]           src_valid,
    output wire [4:0]           src_ready,
    input  wire [5*128-1:0]     src_event,

    // event slot allocation
    input  wire                 alloc_valid,
    output wire                 alloc_ready,
    input  wire [7:0]           alloc_slot,
    input  wire [15:0]          alloc_job,
    input  wire [7:0]           alloc_opcode,
    input  wire                 alloc_irq,

    // dependency lookup
    input  wire [7:0]           lookup_slot,
    input  wire [15:0]          lookup_job,
    output reg  [2:0]           lookup_state,

    // probe / duplicate-ID check
    input  wire [7:0]           slot_probe_slot,
    output wire                 slot_probe_free,
    input  wire [15:0]          id_query_job,
    output reg                  id_query_busy,

    // ADMIN sideband (valid together with source 4)
    input  wire [1:0]           admin_action,
    input  wire [7:0]           admin_target_slot,
    input  wire [15:0]          admin_target_job,
    input  wire                 admin_irq,

    // host completion stream
    output wire                 cpl_valid,
    input  wire                 cpl_ready,
    output wire [127:0]         cpl_data,

    // commit broadcast (one cycle, no backpressure)
    output wire                 retire_valid,
    output wire [127:0]         retire_event,

    output wire                 irq,

    input  wire                 fatal_in,
    output reg                  fatal_latched,

    output wire [31:0]          cpl_level,
    output wire [31:0]          active_slots
);

    localparam AW_CPL = $clog2(CPL_DEPTH);
    localparam [2:0] S_DMA = 3'd0;
    localparam [2:0] S_MPU = 3'd1;
    localparam [2:0] S_VPU = 3'd2;
    localparam [2:0] S_SFU = 3'd3;
    localparam [2:0] S_ADM = 3'd4;
    localparam [2:0] N_SRC = 3'd5;

    // ------------------------------------------------------------------
    // event table
    // ------------------------------------------------------------------
    reg [15:0] ev_job  [0:EVENT_SLOTS-1];
    reg [7:0]  ev_op   [0:EVENT_SLOTS-1];
    reg        ev_irq  [0:EVENT_SLOTS-1];
    reg [2:0]  ev_state[0:EVENT_SLOTS-1];

    integer i;

    // ------------------------------------------------------------------
    // CPL FIFO (show-ahead) kept local so duplicate-ID search can see entries
    // ------------------------------------------------------------------
    reg [127:0]           cpl_mem [0:CPL_DEPTH-1];
    reg [CPL_DEPTH-1:0]   cpl_vld;
    reg [CPL_DEPTH-1:0]   cpl_irq_bits;
    reg [AW_CPL-1:0]      cpl_wp, cpl_rp;
    reg [AW_CPL:0]        cpl_cnt;

    wire cpl_full  = (cpl_cnt == CPL_DEPTH[AW_CPL:0]);
    wire cpl_empty = (cpl_cnt == {(AW_CPL+1){1'b0}});
    wire cpl_pop   = cpl_valid && cpl_ready;

    assign cpl_valid  = !cpl_empty;
    assign cpl_data   = cpl_mem[cpl_rp];
    assign cpl_level  = {{(31-AW_CPL){1'b0}}, cpl_cnt};
    assign irq        = (pending_irq != 16'd0) || fatal_latched;

    reg [15:0] pending_irq;

    // ------------------------------------------------------------------
    // dependency lookup
    // ------------------------------------------------------------------
    wire lookup_in_range = (lookup_slot < EVENT_SLOTS);

    always @* begin
        if (!lookup_in_range)
            lookup_state = `NPU_LOOKUP_FREE;
        else if (ev_state[lookup_slot] == `NPU_LOOKUP_FREE)
            lookup_state = `NPU_LOOKUP_FREE;
        else if (ev_job[lookup_slot] != lookup_job)
            lookup_state = `NPU_LOOKUP_MISMATCH;
        else
            lookup_state = ev_state[lookup_slot];
    end

    assign slot_probe_free =
        (slot_probe_slot < EVENT_SLOTS) &&
        (ev_state[slot_probe_slot] == `NPU_LOOKUP_FREE);

    // ------------------------------------------------------------------
    // duplicate job_id search across the table and the CPL queue
    // ------------------------------------------------------------------
    always @* begin
        id_query_busy = 1'b0;
        for (i = 0; i < EVENT_SLOTS; i = i + 1) begin
            if ((ev_state[i] != `NPU_LOOKUP_FREE) &&
                (ev_job[i] == id_query_job))
                id_query_busy = 1'b1;
        end
        for (i = 0; i < CPL_DEPTH; i = i + 1) begin
            if (cpl_vld[i] && (cpl_mem[i][15:0] == id_query_job))
                id_query_busy = 1'b1;
        end
    end

    // ------------------------------------------------------------------
    // allocation
    // ------------------------------------------------------------------
    wire alloc_slot_ok   = (alloc_slot < EVENT_SLOTS);
    wire alloc_slot_free = alloc_slot_ok ?
                           (ev_state[alloc_slot] == `NPU_LOOKUP_FREE) : 1'b0;

    assign alloc_ready = !fatal_latched && alloc_slot_ok && alloc_slot_free;

    // ------------------------------------------------------------------
    // round-robin source arbitration
    //
    //   A RELEASE admin record whose target is still RUNNING must wait, but
    //   it must not block the other sources: the target can only become
    //   complete if those sources are allowed to commit.  Blocked admins are
    //   skipped by the arbiter (they keep src_valid high until they win).
    // ------------------------------------------------------------------
    reg [2:0] rr;
    integer   k, idx;
    reg [2:0] grant;
    reg       grant_found;

    wire adm_block = (admin_action == `NPU_ADMIN_RELEASE) &&
                     (admin_target_slot < EVENT_SLOTS) &&
                     (ev_state[admin_target_slot] == `NPU_LOOKUP_RUNNING);

    always @* begin
        grant_found = 1'b0;
        grant       = S_DMA;
        for (k = 0; k < 5; k = k + 1) begin
            idx = (rr + k) % 5;
            if (!grant_found && src_valid[idx] &&
                !(idx == S_ADM && adm_block)) begin
                grant       = idx[2:0];
                grant_found = 1'b1;
            end
        end
    end

    wire [127:0] g_event = src_event[grant*128 +: 128];
    wire [7:0]   g_slot  = g_event[23:16];
    wire [15:0]  g_job   = g_event[15:0];

    wire g_is_admin = (grant == S_ADM);
    wire g_slot_none = (g_slot == `NPU_SLOT_NONE);
    wire g_slot_in_range = (g_slot < EVENT_SLOTS);

    // table identity: a no-slot record is only legal from ADMIN
    wire g_table_ok =
        g_slot_none ? g_is_admin :
        (g_slot_in_range &&
         (ev_state[g_slot] == `NPU_LOOKUP_RUNNING) &&
         (ev_job[g_slot] == g_job));

    // RELEASE target handling
    wire g_release = g_is_admin && (admin_action == `NPU_ADMIN_RELEASE);

    wire rel_target_in_range = (admin_target_slot < EVENT_SLOTS);
    wire [2:0] rel_target_state = rel_target_in_range ?
                                  ev_state[admin_target_slot] :
                                  `NPU_LOOKUP_FREE;
    wire rel_target_done = (rel_target_state == `NPU_LOOKUP_OK) ||
                           (rel_target_state == `NPU_LOOKUP_FAILED);
    wire rel_target_job_ok = rel_target_in_range &&
                             (ev_job[admin_target_slot] == admin_target_job);

    // A target that is still RUNNING makes the admin source wait.  A free or
    // mismatched target is accepted as BAD_FIELD and changes nothing.
    wire rel_block = g_release && rel_target_in_range &&
                     (rel_target_state == `NPU_LOOKUP_RUNNING);
    wire rel_bad   = g_release && !(rel_target_done && rel_target_job_ok);

    wire [15:0] g_status = rel_bad ? `NPU_ST_BAD_FIELD : g_event[47:32];

    wire [127:0] commit_event = {g_event[127:48], g_status, g_event[31:0]};

    wire cpl_space_ok = !cpl_full;
    wire grant_accept = grant_found && !rel_block &&
                        (g_table_ok ? cpl_space_ok : 1'b1);

    // one-hot ready for the granted source
    genvar gs;
    generate
        for (gs = 0; gs < 5; gs = gs + 1) begin : g_ready
            assign src_ready[gs] = grant_accept && (grant == gs[2:0]);
        end
    endgenerate

    wire commit_irq = g_slot_none ? admin_irq : ev_irq[g_slot];

    assign retire_valid = grant_accept && g_table_ok;
    assign retire_event = commit_event;

    // ------------------------------------------------------------------
    // sequential state
    // ------------------------------------------------------------------
    always @(posedge clk_core or negedge rst_core_n) begin
        if (!rst_core_n) begin
            for (i = 0; i < EVENT_SLOTS; i = i + 1) begin
                ev_job[i]   <= 16'd0;
                ev_op[i]    <= 8'd0;
                ev_irq[i]   <= 1'b0;
                ev_state[i] <= `NPU_LOOKUP_FREE;
            end

            for (i = 0; i < CPL_DEPTH; i = i + 1) begin
                cpl_mem[i]      <= 128'd0;
                cpl_vld[i]      <= 1'b0;
                cpl_irq_bits[i] <= 1'b0;
            end

            cpl_wp        <= {AW_CPL{1'b0}};
            cpl_rp        <= {AW_CPL{1'b0}};
            cpl_cnt       <= {(AW_CPL+1){1'b0}};
            pending_irq   <= 16'd0;
            rr            <= S_DMA;
            fatal_latched <= 1'b0;
        end else begin
            // ------------------------------------------------ alloc
            if (alloc_valid && alloc_ready) begin
                ev_job[alloc_slot]   <= alloc_job;
                ev_op[alloc_slot]    <= alloc_opcode;
                ev_irq[alloc_slot]   <= alloc_irq;
                ev_state[alloc_slot] <= `NPU_LOOKUP_RUNNING;
            end

            // ------------------------------------------------ commit
            if (grant_accept) begin
                rr <= (grant + 3'd1) % 5;

                if (g_table_ok) begin
                    cpl_mem[cpl_wp]      <= commit_event;
                    cpl_irq_bits[cpl_wp] <= commit_irq;
                    cpl_vld[cpl_wp]      <= 1'b1;
                    cpl_wp               <= cpl_wp + 1'b1;

                    if (commit_irq) begin
                        if (pending_irq != 16'hFFFF)
                            pending_irq <= pending_irq + 16'd1;
                    end

                    if (!g_slot_none && g_slot_in_range)
                        ev_state[g_slot] <= (g_status == 16'd0) ?
                                            `NPU_LOOKUP_OK :
                                            `NPU_LOOKUP_FAILED;

                    if (g_release && !rel_bad && rel_target_in_range)
                        ev_state[admin_target_slot] <= `NPU_LOOKUP_FREE;
                end else begin
                    // Unknown slot, duplicate completion or a normal source
                    // claiming slot 0xFF: consume and isolate.
                    fatal_latched <= 1'b1;
                end
            end

            // ------------------------------------------------ CPL pop
            if (cpl_pop) begin
                cpl_vld[cpl_rp] <= 1'b0;
                cpl_rp          <= cpl_rp + 1'b1;
                if (cpl_irq_bits[cpl_rp] && (pending_irq != 16'd0))
                    pending_irq <= pending_irq - 16'd1;
            end

            // ------------------------------------------------ queue count
            case ({grant_accept && g_table_ok, cpl_pop})
                2'b10:   cpl_cnt <= cpl_cnt + 1'b1;
                2'b01:   cpl_cnt <= cpl_cnt - 1'b1;
                default: cpl_cnt <= cpl_cnt;
            endcase

            // ------------------------------------------------ fatal
            if (fatal_in)
                fatal_latched <= 1'b1;
        end
    end

    // ------------------------------------------------------------------
    // diagnostics
    // ------------------------------------------------------------------
    reg [31:0] active_r;
    always @* begin
        active_r = 32'd0;
        for (i = 0; i < EVENT_SLOTS; i = i + 1) begin
            if (ev_state[i] != `NPU_LOOKUP_FREE)
                active_r = active_r + 32'd1;
        end
    end
    assign active_slots = active_r;

endmodule

`default_nettype wire

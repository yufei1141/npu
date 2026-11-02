//=============================================================================
// task_scheduler : in-order issue, dependencies, buffer locks, admin records
//
//   One command is held at a time (IDLE -> CHECK -> DEP -> RESERVE ->
//   DISPATCH), plus an ADMIN/REJECT path for FENCE, EVENT_RELEASE and error
//   records.  A compute task and a DMA task may be in flight together; the
//   scheduler keeps one lock set for each and releases it on the retire
//   broadcast of the matching job.
//
//   CHECK uses issue_error from the frontend plus the buffer-resource checks
//   of access_decode.  Identity errors and an invalid/occupied event slot are
//   slot-less rejections; other static errors reserve the command's own slot
//   when it is legal and free and report FAILED there.
//
//   RESERVE waits for the engine, the lock set, the event slot and, for
//   MATMUL, for the weight tile0 descriptor to match.  Slot allocation, lock
//   acquisition and the RESERVE -> DISPATCH transition happen on one edge.
//   DISPATCH holds task_valid and the latched command until task_ready.
//
//   FENCE waits (lock_v_* / busy_* / memory outstanding) and reports OK or
//   DEP_FAILED from the barrier_error interval.  EVENT_RELEASE goes straight
//   to ADMIN; the completion manager validates and waits for its target.
//
//   The admin record is presented on the completion manager's ADMIN source:
//   admin_event/status/irq/action/target.  admin_cmd from the interface draft
//   is not needed: this module already builds the complete 128-bit record.
//=============================================================================
//=============================================================================
// 中文说明
//
//   调度器：**顺序发射**、依赖解析、buffer 锁、以及 admin 记录。这是整个
//   加速器"异步提交、顺序执行"中"顺序执行"的那一半。
//
//   一次只持有一条命令，主路径状态：IDLE -> CHECK -> DEP -> RESERVE ->
//   DISPATCH；另有 ADMIN / REJECT 支路处理 FENCE、EVENT_RELEASE 和错误记录。
//   并发度是**两路**：一路计算任务 + 一路 DMA 任务可同时在飞（busy_c /
//   busy_d 两个标志），各自持有一套 buffer 锁，在对应 job 的 retire 广播时
//   释放。这也是为什么整条链路大约 444 周期/命令——不是带宽限制，而是单发射
//   加上依赖等待。
//
//   各状态的判据：
//     CHECK   用前端给的 issue_error 加上 access_decode 的 buffer 资源检查；
//     DEP     等待 dep_job / dep_slot 指定的前置任务完成；
//     RESERVE 等引擎空闲、等 buffer 锁、等事件槽，MATMUL 还要等权重 tile0
//             描述符与当前选择匹配。**槽分配、锁获取、RESERVE->DISPATCH
//             三者同沿完成**，中间没有半成功状态；
//     DISPATCH 一直保持 task_valid 和锁存的命令，直到 task_ready。
//
//   拒绝语义（很容易搞错的一点）：
//     - 身份错误（job_id 重复）和事件槽非法/被占用属于"**不占槽**的拒绝"——
//       因为槽本来就被占着，再去分配只会破坏 WAIT 语义；
//     - 其它静态错误，只要命令自己的槽合法且空闲，就先占上槽再在槽里报
//       FAILED，这样主机仍然能从完成流里看到这条命令的结局。
//
//   FENCE 会等到 buffer 锁、引擎 busy、以及 memory 在途计数全部归零，然后
//   根据 barrier_error 区间报 OK 或 DEP_FAILED。EVENT_RELEASE 直接走 ADMIN，
//   由 completion_manager 负责校验目标并等待它完成。
//
//   admin 记录通过 completion_manager 的 ADMIN 源送出（admin_event/status/
//   irq/action/target）；接口草案里的 admin_cmd 不需要，因为本模块已经直接
//   拼好了完整的 128 位记录。
//=============================================================================
`timescale 1ns/1ps
`default_nettype none

`include "npu_defs.vh"

module task_scheduler #(
    parameter EVENT_SLOTS = 16,
    parameter MEM_DATA_W  = 256
)(
    input  wire                 clk_core,
    input  wire                 rst_core_n,
    input  wire                 run_enable,

    // command stream from the frontend
    input  wire                 issue_valid,
    output wire                 issue_ready,
    input  wire [511:0]         issue_cmd,
    input  wire [15:0]          issue_error,
    input  wire                 issue_identity_error,
    output wire                 held_valid,
    output wire [15:0]          held_job,

    // engine task channels (payload is the unmodified command)
    output wire                 dma_task_valid,
    input  wire                 dma_task_ready,
    output wire [511:0]         dma_task_cmd,
    output wire                 mpu_task_valid,
    input  wire                 mpu_task_ready,
    output wire [511:0]         mpu_task_cmd,
    output wire                 vpu_task_valid,
    input  wire                 vpu_task_ready,
    output wire [511:0]         vpu_task_cmd,
    output wire                 sfu_task_valid,
    input  wire                 sfu_task_ready,
    output wire [511:0]         sfu_task_cmd,

    // event table
    output wire [7:0]           lookup_slot,
    output wire [15:0]          lookup_job,
    input  wire [2:0]           lookup_state,
    output wire                 alloc_valid,
    input  wire                 alloc_ready,
    output wire [7:0]           alloc_slot,
    output wire [15:0]          alloc_job,
    output wire [7:0]           alloc_opcode,
    output wire                 alloc_irq,
    output wire [7:0]           slot_probe_slot,
    input  wire                 slot_probe_free,
    output wire [15:0]          id_query_job,
    input  wire                 id_query_busy,

    // admin record to the completion manager
    output wire                 admin_valid,
    input  wire                 admin_ready,
    output wire [127:0]         admin_event,
    output wire [15:0]          admin_status,
    output wire                 admin_irq,
    output wire [1:0]           admin_action,
    output wire [7:0]           admin_target_slot,
    output wire [15:0]          admin_target_job,

    // retirement broadcast from the completion manager
    input  wire                 retire_valid,
    input  wire [127:0]         retire_event,

    // weight tile0 query
    output wire [15:0]          weight_query_job,
    output wire [15:0]          weight_query_tile,
    output wire [1:0]           weight_query_mode,
    output wire [3:0]           weight_query_g,
    output wire [15:0]          weight_query_s,
    input  wire                 weight_query_match,
    input  wire                 weight_query_fault,

    // fault
    input  wire                 fault_in,
    output wire                 fatal,

    // buffer configuration (flat, shared with memory_subsystem)
    input  wire [7:0]           buf_enable,
    input  wire [63:0]          buf_bank_flat,
    input  wire [255:0]         buf_base_flat,
    input  wire [255:0]         buf_size_flat,

    // all earlier memory traffic has been answered
    input  wire                 mem_outstanding_zero
);

    localparam S_IDLE     = 3'd0;
    localparam S_CHECK    = 3'd1;
    localparam S_DEP      = 3'd2;
    localparam S_RESERVE  = 3'd3;
    localparam S_DISPATCH = 3'd4;
    localparam S_ADMIN    = 3'd5;
    localparam S_REJECT   = 3'd6;
    localparam S_HALT     = 3'd7;

    localparam K_NONE    = 3'd0;
    localparam K_DMA     = 3'd1;
    localparam K_MPU     = 3'd2;
    localparam K_VPU     = 3'd3;
    localparam K_SFU     = 3'd4;
    localparam K_FENCE   = 3'd5;
    localparam K_RELEASE = 3'd6;

    // ------------------------------------------------------------------
    // held command and decode
    // ------------------------------------------------------------------
    reg  [2:0]   state;
    reg  [511:0] cmd_r;
    reg  [15:0]  err_r;
    reg          id_err_r;
    reg          dep_en_r;
    reg          irq_r;
    reg  [15:0]  rej_status;
    reg          rej_slotless;
    reg          rej_allocated;
    reg          fatal_r;

    wire [2:0]   kind;
    wire [7:0]   dec_opcode;
    wire [7:0]   dec_subop;
    wire [15:0]  dec_job;
    wire [15:0]  dec_dep_job;
    wire [7:0]   dec_dep_slot;
    wire [7:0]   dec_ev_slot;
    wire         dec_dep_en;
    wire         dec_inplace;
    wire         dec_irq;
    wire [7:0]   dec_read_mask;
    wire [7:0]   dec_write_mask;
    wire [7:0]   dec_rel_slot;
    wire [15:0]  dec_rel_job;
    wire [15:0]  dec_chk_status;

    access_decode #(
        .MEM_DATA_W(MEM_DATA_W), .EVENT_SLOTS(EVENT_SLOTS)
    ) u_decode (
        .cmd(cmd_r),
        .buf_enable(buf_enable),
        .buf_bank_flat(buf_bank_flat),
        .buf_base_flat(buf_base_flat),
        .buf_size_flat(buf_size_flat),
        .kind(kind),
        .opcode(dec_opcode),
        .subop(dec_subop),
        .job(dec_job),
        .dep_job(dec_dep_job),
        .dep_slot(dec_dep_slot),
        .ev_slot(dec_ev_slot),
        .dep_en(dec_dep_en),
        .inplace(dec_inplace),
        .irq(dec_irq),
        .read_mask(dec_read_mask),
        .write_mask(dec_write_mask),
        .rel_target_slot(dec_rel_slot),
        .rel_target_job(dec_rel_job),
        .chk_status(dec_chk_status)
    );

    wire is_fence   = (kind == K_FENCE);
    wire is_release = (kind == K_RELEASE);
    wire is_normal  = (kind == K_DMA) || (kind == K_MPU) ||
                      (kind == K_VPU) || (kind == K_SFU);
    wire is_dma     = (kind == K_DMA);

    // ------------------------------------------------------------------
    // in-flight lock state
    // ------------------------------------------------------------------
    reg          lock_v_c;      // compute class (MPU/VPU/SFU)
    reg  [7:0]   lock_r_c, lock_w_c;
    reg  [15:0]  job_c;
    reg          lock_v_d;      // DMA class
    reg  [7:0]   lock_r_d, lock_w_d;
    reg  [15:0]  job_d;
    reg          busy_c, busy_d;

    wire [7:0] live_rmask = (lock_v_c ? lock_r_c : 8'd0) |
                            (lock_v_d ? lock_r_d : 8'd0);
    wire [7:0] live_wmask = (lock_v_c ? lock_w_c : 8'd0) |
                            (lock_v_d ? lock_w_d : 8'd0);

    wire req_w_conflict = |(dec_write_mask & (live_rmask | live_wmask));
    wire req_r_conflict = |(dec_read_mask  &  live_wmask);
    wire locks_ok       = !(req_w_conflict || req_r_conflict);

    wire class_free = is_dma ? !lock_v_d : !lock_v_c;

    wire engine_ready =
        (kind == K_DMA) ? dma_task_ready :
        (kind == K_MPU) ? mpu_task_ready :
        (kind == K_VPU) ? vpu_task_ready :
        (kind == K_SFU) ? sfu_task_ready : 1'b0;

    // A MATMUL with src1_buf set takes its B operand from a w16 activation
    // frame in a scratchpad buffer, not from a weight bank.  No tile is ever
    // loaded for it, so the weight query can only answer "no match" and must
    // not gate dispatch -- otherwise the job waits in S_RESERVE forever.
    wire weight_need = (kind == K_MPU) && (cmd_r[79:72] == 8'd0);

    wire weight_ok  = weight_need ? weight_query_match : 1'b1;
    wire weight_bad = weight_need && weight_query_fault;

    // ------------------------------------------------------------------
    // dependency lookup
    // ------------------------------------------------------------------
    assign lookup_slot = dec_dep_slot;
    assign lookup_job  = dec_dep_job;

    // ------------------------------------------------------------------
    // event slot allocation / probe / identity query
    // ------------------------------------------------------------------
    wire slot_in_range = (dec_ev_slot < EVENT_SLOTS);
    wire slot_free_now = slot_probe_free;

    wire reserve_alloc = (state == S_RESERVE) &&
                         (is_fence ||
                          (is_normal && !weight_bad && class_free &&
                           engine_ready && locks_ok && weight_ok));

    wire rej_alloc = (state == S_REJECT) && !rej_slotless && !rej_allocated;

    assign alloc_valid  = reserve_alloc || rej_alloc;
    assign alloc_slot   = dec_ev_slot;
    assign alloc_job    = dec_job;
    assign alloc_opcode = dec_opcode;
    assign alloc_irq    = (state == S_REJECT) ?
                          (rej_slotless ? 1'b0 : irq_r) : irq_r;

    assign slot_probe_slot = dec_ev_slot;
    assign id_query_job    = dec_job;

    // ------------------------------------------------------------------
    // reject decision
    // ------------------------------------------------------------------
    wire can_rej_alloc = slot_in_range && slot_free_now;

    reg [15:0] chk_rej_status;
    reg        chk_rej_slotless;
    always @* begin
        chk_rej_status   = `NPU_ST_OK;
        chk_rej_slotless = 1'b0;

        if (id_err_r) begin
            chk_rej_status   = `NPU_ST_BAD_FIELD;
            chk_rej_slotless = 1'b1;
        end else if (err_r != `NPU_ST_OK) begin
            chk_rej_status   = err_r;
            chk_rej_slotless = !can_rej_alloc;
        end else if (kind == K_NONE) begin
            chk_rej_status   = `NPU_ST_BAD_OPCODE;
            chk_rej_slotless = !can_rej_alloc;
        end else if (!is_release && !slot_in_range) begin
            chk_rej_status   = `NPU_ST_RANGE;
            chk_rej_slotless = 1'b1;
        end else if (!is_release && !slot_free_now) begin
            chk_rej_status   = `NPU_ST_EVENT_BUSY;
            chk_rej_slotless = 1'b1;
        end else if (id_query_busy) begin
            chk_rej_status   = `NPU_ST_BAD_FIELD;
            chk_rej_slotless = 1'b1;
        end else if (dec_chk_status != `NPU_ST_OK) begin
            chk_rej_status   = dec_chk_status;
            chk_rej_slotless = !can_rej_alloc;
        end
    end

    // ------------------------------------------------------------------
    // admin record
    // ------------------------------------------------------------------
    reg  [15:0]  fence_status;
    wire         fence_ready = !lock_v_c && !lock_v_d &&
                               !busy_c && !busy_d && mem_outstanding_zero;

    always @* begin
        if (is_fence)
            fence_status = barrier_error ? `NPU_ST_DEP_FAILED : `NPU_ST_OK;
        else
            fence_status = `NPU_ST_OK;
    end

    reg [127:0] admin_event_r;
    reg [15:0]  admin_status_r;
    reg         admin_irq_r;
    reg [1:0]   admin_action_r;
    reg [7:0]   admin_target_slot_r;
    reg [15:0]  admin_target_job_r;
    reg         admin_valid_r;

    always @* begin
        admin_event_r       = 128'd0;
        admin_event_r[15:0]  = dec_job;
        admin_status_r      = `NPU_ST_OK;
        admin_irq_r         = 1'b0;
        admin_action_r      = `NPU_ADMIN_NONE;
        admin_target_slot_r = 8'd0;
        admin_target_job_r  = 16'd0;
        admin_valid_r       = 1'b0;

        if (state == S_ADMIN) begin
            admin_event_r[23:16] = dec_ev_slot;
            admin_event_r[31:24] = dec_opcode;
            if (is_release) begin
                admin_event_r[23:16] = `NPU_SLOT_NONE;
                admin_event_r[47:32] = `NPU_ST_OK;
                admin_status_r       = `NPU_ST_OK;
                admin_action_r       = `NPU_ADMIN_RELEASE;
                admin_target_slot_r  = dec_rel_slot;
                admin_target_job_r   = dec_rel_job;
                admin_irq_r          = irq_r;
                admin_valid_r        = 1'b1;
            end else begin
                admin_event_r[47:32] = fence_status;
                admin_status_r       = fence_status;
                admin_irq_r          = irq_r;
                admin_valid_r        = fence_ready;
            end
        end else if (state == S_REJECT) begin
            admin_event_r[23:16] = rej_slotless ? `NPU_SLOT_NONE : dec_ev_slot;
            admin_event_r[31:24] = dec_opcode;
            admin_event_r[47:32] = rej_status;
            admin_status_r       = rej_status;
            admin_irq_r          = rej_slotless ? 1'b0 : irq_r;
            admin_action_r       = `NPU_ADMIN_NONE;
            admin_valid_r        = rej_slotless || rej_allocated;
        end
    end

    assign admin_valid       = admin_valid_r;
    assign admin_event       = admin_event_r;
    assign admin_status      = admin_status_r;
    assign admin_irq         = admin_irq_r;
    assign admin_action      = admin_action_r;
    assign admin_target_slot = admin_target_slot_r;
    assign admin_target_job  = admin_target_job_r;

    // ------------------------------------------------------------------
    // weight tile0 query
    // ------------------------------------------------------------------
    assign weight_query_job  = dec_job;
    assign weight_query_tile = 16'd0;
    assign weight_query_mode = cmd_r[145:144];        // P0[17:16]
    assign weight_query_g    = cmd_r[143:136];        // P0[15:8]
    assign weight_query_s    = cmd_r[191:176];        // P1[31:16]

    // ------------------------------------------------------------------
    // engine task channels
    // ------------------------------------------------------------------
    wire dispatch = (state == S_DISPATCH);

    assign dma_task_valid = dispatch && (kind == K_DMA);
    assign mpu_task_valid = dispatch && (kind == K_MPU);
    assign vpu_task_valid = dispatch && (kind == K_VPU);
    assign sfu_task_valid = dispatch && (kind == K_SFU);
    assign dma_task_cmd   = cmd_r;
    assign mpu_task_cmd   = cmd_r;
    assign vpu_task_cmd   = cmd_r;
    assign sfu_task_cmd   = cmd_r;

    // ------------------------------------------------------------------
    // frontend handshake
    // ------------------------------------------------------------------
    assign issue_ready = (state == S_IDLE) && run_enable && !fatal_r;
    assign held_valid  = (state != S_IDLE) && (state != S_HALT);
    assign held_job    = dec_job;

    assign fatal = fatal_r;

    // ------------------------------------------------------------------
    // barrier error
    // ------------------------------------------------------------------
    reg barrier_error;

    wire admin_fire = admin_valid_r && admin_ready;
    wire fence_fire = admin_fire && (state == S_ADMIN) && is_fence;
    wire rej_fire   = admin_fire && (state == S_REJECT);
    wire ret_fail_c = retire_valid && busy_c &&
                      (retire_event[15:0] == job_c) &&
                      (retire_event[47:32] != `NPU_ST_OK);
    wire ret_fail_d = retire_valid && busy_d &&
                      (retire_event[15:0] == job_d) &&
                      (retire_event[47:32] != `NPU_ST_OK);

    // ------------------------------------------------------------------
    // sequential control
    // ------------------------------------------------------------------
    integer i;

    always @(posedge clk_core or negedge rst_core_n) begin
        if (!rst_core_n) begin
            state          <= S_IDLE;
            cmd_r          <= 512'd0;
            err_r          <= 16'd0;
            id_err_r       <= 1'b0;
            dep_en_r       <= 1'b0;
            irq_r          <= 1'b0;
            rej_status     <= 16'd0;
            rej_slotless   <= 1'b0;
            rej_allocated  <= 1'b0;
            fatal_r        <= 1'b0;
            lock_v_c       <= 1'b0;
            lock_v_d       <= 1'b0;
            lock_r_c       <= 8'd0;
            lock_w_c       <= 8'd0;
            lock_r_d       <= 8'd0;
            lock_w_d       <= 8'd0;
            job_c          <= 16'd0;
            job_d          <= 16'd0;
            busy_c         <= 1'b0;
            busy_d         <= 1'b0;
            barrier_error  <= 1'b0;
        end else begin
            // -------------------------------------------------- fault
            if (fault_in && !fatal_r) begin
                fatal_r <= 1'b1;
                state   <= S_HALT;
            end

            // ------------------------------------------------ retire
            if (retire_valid) begin
                if (lock_v_c && (retire_event[15:0] == job_c)) begin
                    lock_v_c <= 1'b0;
                    lock_r_c <= 8'd0;
                    lock_w_c <= 8'd0;
                    busy_c   <= 1'b0;
                end
                if (lock_v_d && (retire_event[15:0] == job_d)) begin
                    lock_v_d <= 1'b0;
                    lock_r_d <= 8'd0;
                    lock_w_d <= 8'd0;
                    busy_d   <= 1'b0;
                end
            end

            // ------------------------------------------- barrier error
            if (fence_fire)
                barrier_error <= 1'b0;
            else if (rej_fire)
                barrier_error <= 1'b1;
            else if (ret_fail_c || ret_fail_d)
                barrier_error <= 1'b1;

            // --------------------------------------------- reject alloc
            if (rej_alloc && alloc_ready)
                rej_allocated <= 1'b1;

            // --------------------------------------------------- FSM
            case (state)
            S_IDLE: begin
                if (issue_valid && issue_ready) begin
                    cmd_r     <= issue_cmd;
                    err_r     <= issue_error;
                    id_err_r  <= issue_identity_error;
                    dep_en_r  <= issue_cmd[24];
                    irq_r     <= issue_cmd[26];
                    state     <= S_CHECK;
                end
            end

            S_CHECK: begin
                if (chk_rej_status != `NPU_ST_OK) begin
                    rej_status    <= chk_rej_status;
                    rej_slotless  <= chk_rej_slotless;
                    rej_allocated <= 1'b0;
                    state         <= S_REJECT;
                end else begin
                    state <= S_DEP;
                end
            end

            S_DEP: begin
                if (!dep_en_r) begin
                    state <= is_release ? S_ADMIN : S_RESERVE;
                end else begin
                    case (lookup_state)
                    `NPU_LOOKUP_OK: begin
                        state <= is_release ? S_ADMIN : S_RESERVE;
                    end
                    `NPU_LOOKUP_RUNNING: begin
                        state <= S_DEP;
                    end
                    `NPU_LOOKUP_FAILED: begin
                        rej_status    <= `NPU_ST_DEP_FAILED;
                        rej_slotless  <= 1'b0;
                        rej_allocated <= 1'b0;
                        state         <= S_REJECT;
                    end
                    default: begin   // FREE or MISMATCH
                        rej_status    <= `NPU_ST_DEP_MISSING;
                        rej_slotless  <= 1'b0;
                        rej_allocated <= 1'b0;
                        state         <= S_REJECT;
                    end
                    endcase
                end
            end

            S_RESERVE: begin
                if (is_fence) begin
                    if (alloc_ready)
                        state <= S_ADMIN;
                end else if (weight_bad) begin
                    rej_status    <= `NPU_ST_BAD_FIELD;
                    rej_slotless  <= !can_rej_alloc;
                    rej_allocated <= 1'b0;
                    state         <= S_REJECT;
                end else if (class_free && engine_ready && locks_ok &&
                             weight_ok && alloc_ready) begin
                    if (is_dma) begin
                        lock_v_d <= 1'b1;
                        lock_r_d <= dec_read_mask;
                        lock_w_d <= dec_write_mask;
                        job_d    <= dec_job;
                    end else begin
                        lock_v_c <= 1'b1;
                        lock_r_c <= dec_read_mask;
                        lock_w_c <= dec_write_mask;
                        job_c    <= dec_job;
                    end
                    state <= S_DISPATCH;
                end
            end

            S_DISPATCH: begin
                if (dma_task_ready && (kind == K_DMA)) begin
                    busy_d <= 1'b1;
                    state  <= S_IDLE;
                end else if ((mpu_task_ready && (kind == K_MPU)) ||
                             (vpu_task_ready && (kind == K_VPU)) ||
                             (sfu_task_ready && (kind == K_SFU))) begin
                    busy_c <= 1'b1;
                    state  <= S_IDLE;
                end
            end

            S_ADMIN: begin
                if (admin_fire)
                    state <= S_IDLE;
            end

            S_REJECT: begin
                if (admin_fire)
                    state <= S_IDLE;
            end

            S_HALT: begin
                state <= S_HALT;
            end

            default: state <= S_IDLE;
            endcase
        end
    end

    // The in-flight job is also the job whose retire releases the lock.
    // (job_c/job_d are loaded at RESERVE, before busy is set.)

endmodule

`default_nettype wire

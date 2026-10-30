//=============================================================================
// tb_control_plane : frontend + scheduler + completion manager + engine stubs
//
//   Covers: LOAD/MATMUL dependency, compute/DMA overlap, duplicate job_id,
//   occupied/out-of-range event slot, FENCE barrier, failed-task propagation,
//   EVENT_RELEASE, and job_id reuse after release.
//=============================================================================
`timescale 1ns/1ps
`default_nettype none

`include "npu_defs.vh"

module tb_control_plane;
    reg clk = 1'b0;
    reg rst_n = 1'b0;
    reg run_enable = 1'b0;

    // ---------------------------------------------------------------- host
    reg         cmd_valid = 1'b0;
    wire        cmd_ready;
    reg [511:0] cmd_data = 512'd0;

    // ------------------------------------------------------ frontend/sched
    wire        issue_valid;
    wire        issue_ready;
    wire [511:0] issue_cmd;
    wire [15:0] issue_error;
    wire        issue_id_err;
    wire        held_valid;
    wire [15:0] held_job;
    wire [31:0] fifo_level;
    wire [63:0] accepted_count, rejected_count;

    // ---------------------------------------------------------- scheduler
    wire        dma_valid, mpu_valid, vpu_valid, sfu_valid;
    wire        dma_ready, mpu_ready, vpu_ready, sfu_ready;
    wire [511:0] dma_cmd, mpu_cmd, vpu_cmd, sfu_cmd;

    wire [7:0]  lookup_slot;
    wire [15:0] lookup_job;
    wire [2:0]  lookup_state;
    wire        alloc_valid, alloc_ready;
    wire [7:0]  alloc_slot;
    wire [15:0] alloc_job;
    wire [7:0]  alloc_opcode;
    wire        alloc_irq;
    wire [7:0]  slot_probe_slot;
    wire        slot_probe_free;
    wire [15:0] id_query_job;
    wire        id_query_busy;

    wire        admin_valid, admin_ready;
    wire [127:0] admin_event;
    wire [15:0] admin_status;
    wire        admin_irq;
    wire [1:0]  admin_action;
    wire [7:0]  admin_target_slot;
    wire [15:0] admin_target_job;

    wire        retire_valid;
    wire [127:0] retire_event;

    wire [15:0] wq_job;
    wire [15:0] wq_tile;
    wire [1:0]  wq_mode;
    wire [3:0]  wq_g;
    wire [15:0] wq_s;
    reg         wq_match = 1'b1;
    reg         wq_fault = 1'b0;

    reg         fault_in = 1'b0;
    wire        fatal;

    wire [7:0]   buf_enable;
    wire [63:0]  buf_bank_flat;
    wire [255:0] buf_base_flat;
    wire [255:0] buf_size_flat;

    // ------------------------------------------------------ completion mgr
    wire [4:0]  src_valid, src_ready;
    wire [5*128-1:0] src_event;
    wire        cpl_valid;
    reg         cpl_ready = 1'b1;
    wire [127:0] cpl_data;
    wire        irq;
    wire        fatal_latched;
    wire [31:0] cpl_level;
    wire [31:0] active_slots;

    // ------------------------------------------------------------ engines
    wire        dma_ev_valid, mpu_ev_valid, vpu_ev_valid, sfu_ev_valid;
    wire        dma_ev_ready, mpu_ev_ready, vpu_ev_ready, sfu_ev_ready;
    wire [127:0] dma_ev, mpu_ev, vpu_ev, sfu_ev;
    reg         dma_fail = 1'b0;

    integer errors = 0;
    always #5 clk = ~clk;

    // ------------------------------------------------------------------
    // buffer configuration: one bank per logical buffer, 4 KiB each
    // ------------------------------------------------------------------
    assign buf_enable      = 8'hFF;
    assign buf_bank_flat   = {8'd7, 8'd6, 8'd5, 8'd4, 8'd3, 8'd2, 8'd1, 8'd0};
    assign buf_base_flat   = {256{1'b0}};
    assign buf_size_flat   = {8{32'd4096}};

    // ------------------------------------------------------------------
    // DUTs
    // ------------------------------------------------------------------
    command_frontend #(
        .EVENT_SLOTS(16), .RESULT_SLOTS(2), .WBUF_K_STEPS(256),
        .MEM_DATA_W(256), .V_LANES(4), .CMD_FIFO_DEPTH(8)
    ) u_front (
        .clk_core(clk), .rst_core_n(rst_n),
        .accept_enable(run_enable && !fatal),
        .cmd_valid(cmd_valid), .cmd_ready(cmd_ready), .cmd_data(cmd_data),
        .issue_valid(issue_valid), .issue_ready(issue_ready),
        .issue_cmd(issue_cmd), .issue_error(issue_error),
        .issue_identity_error(issue_id_err),
        .held_valid(held_valid), .held_job(held_job),
        .fifo_level(fifo_level),
        .accepted_count(accepted_count), .rejected_count(rejected_count)
    );

    task_scheduler #(
        .EVENT_SLOTS(16), .MEM_DATA_W(256)
    ) u_sched (
        .clk_core(clk), .rst_core_n(rst_n), .run_enable(run_enable),
        .issue_valid(issue_valid), .issue_ready(issue_ready),
        .issue_cmd(issue_cmd), .issue_error(issue_error),
        .issue_identity_error(issue_id_err),
        .held_valid(held_valid), .held_job(held_job),
        .dma_task_valid(dma_valid), .dma_task_ready(dma_ready),
        .dma_task_cmd(dma_cmd),
        .mpu_task_valid(mpu_valid), .mpu_task_ready(mpu_ready),
        .mpu_task_cmd(mpu_cmd),
        .vpu_task_valid(vpu_valid), .vpu_task_ready(vpu_ready),
        .vpu_task_cmd(vpu_cmd),
        .sfu_task_valid(sfu_valid), .sfu_task_ready(sfu_ready),
        .sfu_task_cmd(sfu_cmd),
        .lookup_slot(lookup_slot), .lookup_job(lookup_job),
        .lookup_state(lookup_state),
        .alloc_valid(alloc_valid), .alloc_ready(alloc_ready),
        .alloc_slot(alloc_slot), .alloc_job(alloc_job),
        .alloc_opcode(alloc_opcode), .alloc_irq(alloc_irq),
        .slot_probe_slot(slot_probe_slot), .slot_probe_free(slot_probe_free),
        .id_query_job(id_query_job), .id_query_busy(id_query_busy),
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
        .fault_in(fault_in), .fatal(fatal),
        .buf_enable(buf_enable),
        .buf_bank_flat(buf_bank_flat),
        .buf_base_flat(buf_base_flat),
        .buf_size_flat(buf_size_flat),
        .mem_outstanding_zero(1'b1)
    );

    completion_manager #(
        .EVENT_SLOTS(16), .CPL_DEPTH(8)
    ) u_cm (
        .clk_core(clk), .rst_core_n(rst_n),
        .src_valid(src_valid), .src_ready(src_ready), .src_event(src_event),
        .alloc_valid(alloc_valid), .alloc_ready(alloc_ready),
        .alloc_slot(alloc_slot), .alloc_job(alloc_job),
        .alloc_opcode(alloc_opcode), .alloc_irq(alloc_irq),
        .lookup_slot(lookup_slot), .lookup_job(lookup_job),
        .lookup_state(lookup_state),
        .slot_probe_slot(slot_probe_slot), .slot_probe_free(slot_probe_free),
        .id_query_job(id_query_job), .id_query_busy(id_query_busy),
        .admin_action(admin_action), .admin_target_slot(admin_target_slot),
        .admin_target_job(admin_target_job), .admin_irq(admin_irq),
        .cpl_valid(cpl_valid), .cpl_ready(cpl_ready), .cpl_data(cpl_data),
        .retire_valid(retire_valid), .retire_event(retire_event),
        .irq(irq), .fatal_in(fault_in), .fatal_latched(fatal_latched),
        .cpl_level(cpl_level), .active_slots(active_slots)
    );

    engine_stub #(.DELAY(3)) u_dma (
        .clk(clk), .rst_n(rst_n),
        .task_valid(dma_valid), .task_ready(dma_ready), .task_cmd(dma_cmd),
        .fail(dma_fail),
        .ev_valid(dma_ev_valid), .ev_ready(dma_ev_ready), .ev(dma_ev)
    );

    engine_stub #(.DELAY(4)) u_mpu (
        .clk(clk), .rst_n(rst_n),
        .task_valid(mpu_valid), .task_ready(mpu_ready), .task_cmd(mpu_cmd),
        .fail(1'b0),
        .ev_valid(mpu_ev_valid), .ev_ready(mpu_ev_ready), .ev(mpu_ev)
    );

    engine_stub #(.DELAY(4)) u_vpu (
        .clk(clk), .rst_n(rst_n),
        .task_valid(vpu_valid), .task_ready(vpu_ready), .task_cmd(vpu_cmd),
        .fail(1'b0),
        .ev_valid(vpu_ev_valid), .ev_ready(vpu_ev_ready), .ev(vpu_ev)
    );

    engine_stub #(.DELAY(2)) u_sfu (
        .clk(clk), .rst_n(rst_n),
        .task_valid(sfu_valid), .task_ready(sfu_ready), .task_cmd(sfu_cmd),
        .fail(1'b0),
        .ev_valid(sfu_ev_valid), .ev_ready(sfu_ev_ready), .ev(sfu_ev)
    );

    assign src_valid = {admin_valid, sfu_ev_valid, vpu_ev_valid,
                        mpu_ev_valid, dma_ev_valid};
    assign dma_ev_ready = src_ready[0];
    assign mpu_ev_ready = src_ready[1];
    assign vpu_ev_ready = src_ready[2];
    assign sfu_ev_ready = src_ready[3];
    assign admin_ready  = src_ready[4];
    assign src_event[0*128 +: 128] = dma_ev;
    assign src_event[1*128 +: 128] = mpu_ev;
    assign src_event[2*128 +: 128] = vpu_ev;
    assign src_event[3*128 +: 128] = sfu_ev;
    assign src_event[4*128 +: 128] = admin_event;

    // ------------------------------------------------------------------
    // completion monitor
    // ------------------------------------------------------------------
    reg [15:0] rec_job   [0:63];
    reg [7:0]  rec_slot  [0:63];
    reg [7:0]  rec_op    [0:63];
    reg [15:0] rec_status[0:63];
    integer    rec_count = 0;

    always @(posedge clk) begin
        if (cpl_valid && cpl_ready) begin
            rec_job[rec_count]    <= cpl_data[15:0];
            rec_slot[rec_count]   <= cpl_data[23:16];
            rec_op[rec_count]     <= cpl_data[31:24];
            rec_status[rec_count] <= cpl_data[47:32];
            rec_count <= rec_count + 1;
        end
    end

    // ------------------------------------------------------------------
    // helpers
    // ------------------------------------------------------------------
    task chk;
        input cond;
        input [255:0] msg;
        begin
            if (!cond) begin
                $display("FAIL @%0t: %0s", $time, msg);
                errors = errors + 1;
            end
        end
    endtask

    task clear_cmd;
        begin
            cmd_data = 512'd0;
            cmd_data[7:0] = 8'h01;
        end
    endtask

    task send_cmd;
        integer guard;
        begin
            guard = 0;
            @(negedge clk);
            cmd_valid = 1'b1;
            #1;
            while (!cmd_ready && guard < 200) begin
                @(negedge clk); #1;
                guard = guard + 1;
            end
            chk(cmd_ready, "command not accepted");
            @(posedge clk);
            @(negedge clk);
            cmd_valid = 1'b0;
            @(posedge clk);
        end
    endtask

    task wait_record;
        input integer n;
        integer guard;
        begin
            guard = 0;
            while ((rec_count <= n) && (guard < 4000)) begin
                @(posedge clk);
                guard = guard + 1;
            end
            chk(rec_count > n, "completion record timed out");
        end
    endtask

    task expect_rec;
        input integer n;
        input [15:0] job;
        input [7:0]  slot;
        input [7:0]  op;
        input [15:0] status;
        input [255:0] msg;
        begin
            wait_record(n);
            chk(rec_job[n] === job, msg);
            chk(rec_slot[n] === slot, msg);
            chk(rec_op[n] === op, msg);
            chk(rec_status[n] === status, msg);
        end
    endtask

    // ------------------------------------------------------------------
    // command builders
    // ------------------------------------------------------------------
    task mk_load;
        input [15:0] job;
        input [7:0]  slot;
        input [7:0]  dbuf_id;
        input [31:0] bytes;
        begin
            clear_cmd();
            cmd_data[15:8]  = `NPU_OP_LOAD;
            cmd_data[47:32] = job;
            cmd_data[87:80] = dbuf_id;
            cmd_data[103:96]= slot;
            cmd_data[159:128] = 32'd0;        // P0 ext addr low
            cmd_data[191:160] = 32'd0;        // P1 ext addr high
            cmd_data[223:192] = 32'd0;        // P2 local offset
            cmd_data[255:224] = bytes;        // P3 row bytes
            cmd_data[287:256] = 32'd1;        // P4 rows
        end
    endtask

    task mk_matmul;
        input [15:0] job;
        input [7:0]  slot;
        input [7:0]  abuf;
        input [7:0]  dbuf;
        input        dep_en;
        input [7:0]  dep_slot;
        input [15:0] dep_job;
        begin
            clear_cmd();
            cmd_data[15:8]  = `NPU_OP_MATMUL;
            cmd_data[24]    = dep_en;
            cmd_data[47:32] = job;
            cmd_data[63:48] = dep_job;
            cmd_data[71:64] = abuf;
            cmd_data[87:80] = dbuf;
            cmd_data[95:88] = 8'h8C;          // src0 FP16, src1 INT4, dst FP16, acc FP32
            cmd_data[103:96]= slot;
            cmd_data[111:104] = dep_slot;
            cmd_data[159:128] = {14'd0, 2'b00, 8'd1, 8'd1}; // mode W4FP16, g=1, m_len=1
            cmd_data[191:160] = {16'd1, 16'd1};             // k_steps=1, tiles=1
            cmd_data[223:192] = 32'd16;       // P2 K_valid
            cmd_data[255:224] = 32'd4;        // P3 N_valid
            cmd_data[287:256] = 32'd0;        // P4 A_base
            cmd_data[319:288] = 32'd32;       // P5 A_row_stride
            cmd_data[351:320] = 32'd0;        // P6 A_tile_stride
            cmd_data[383:352] = 32'd0;        // P7 D_base
            cmd_data[415:384] = 32'd8;        // P8 D_row_stride
        end
    endtask

    task mk_vector;
        input [15:0] job;
        input [7:0]  slot;
        input [7:0]  s0;
        input [7:0]  s1;
        input [7:0]  d;
        input [31:0] count;
        begin
            clear_cmd();
            cmd_data[15:8]  = `NPU_OP_VECTOR;
            cmd_data[23:16] = `NPU_VEC_ADD;
            cmd_data[47:32] = job;
            cmd_data[71:64] = s0;
            cmd_data[79:72] = s1;
            cmd_data[87:80] = d;
            cmd_data[95:88] = 8'h80;          // FP16 src/dst, FP32 acc
            cmd_data[103:96]= slot;
            cmd_data[159:128] = count;
        end
    endtask

    task mk_sfu;
        input [15:0] job;
        input [7:0]  slot;
        input [7:0]  s0;
        input [7:0]  d;
        input [31:0] count;
        begin
            clear_cmd();
            cmd_data[15:8]  = `NPU_OP_SFU;
            cmd_data[47:32] = job;
            cmd_data[71:64] = s0;
            cmd_data[87:80] = d;
            cmd_data[95:88] = 8'h00;          // FP16 src/dst, acc unused (FP16)
            cmd_data[103:96]= slot;
            cmd_data[159:128] = count;
        end
    endtask

    task mk_fence;
        input [15:0] job;
        input [7:0]  slot;
        begin
            clear_cmd();
            cmd_data[15:8]  = `NPU_OP_FENCE;
            cmd_data[47:32] = job;
            cmd_data[103:96]= slot;
        end
    endtask

    task mk_release;
        input [15:0] job;
        input [7:0]  target_slot;
        input [15:0] target_job;
        begin
            clear_cmd();
            cmd_data[15:8]   = `NPU_OP_RELEASE;
            cmd_data[47:32]  = job;
            cmd_data[103:96] = `NPU_SLOT_NONE;
            cmd_data[159:128] = {8'd0, target_job, target_slot};
        end
    endtask

    // ------------------------------------------------------------------
    // stimulus
    // ------------------------------------------------------------------
    initial begin
        repeat (3) @(posedge clk);
        rst_n = 1'b1;
        repeat (2) @(posedge clk);
        run_enable = 1'b1;
        repeat (2) @(posedge clk);

        // ------------------------------------------------------------
        // A. LOAD completes and produces a record
        // ------------------------------------------------------------
        mk_load(16'h0010, 8'd0, 8'd1, 32'd32);
        send_cmd();
        expect_rec(0, 16'h0010, 8'd0, `NPU_OP_LOAD, `NPU_ST_OK,
                   "LOAD record");

        // ------------------------------------------------------------
        // B. MATMUL depends on the LOAD slot
        // ------------------------------------------------------------
        mk_matmul(16'h0011, 8'd1, 8'd0, 8'd1, 1'b1, 8'd0, 16'h0010);
        send_cmd();
        expect_rec(1, 16'h0011, 8'd1, `NPU_OP_MATMUL, `NPU_ST_OK,
                   "MATMUL record after dependency");

        // ------------------------------------------------------------
        // C. compute and DMA overlap; a FENCE waits for both
        // ------------------------------------------------------------
        mk_vector(16'h0012, 8'd2, 8'd0, 8'd1, 8'd2, 32'd16);
        send_cmd();
        mk_load(16'h0013, 8'd3, 8'd3, 32'd32);
        send_cmd();
        mk_fence(16'h0030, 8'd4);
        send_cmd();

        expect_rec(2, 16'h0012, 8'd2, `NPU_OP_VECTOR, `NPU_ST_OK,
                   "vector record");
        expect_rec(3, 16'h0013, 8'd3, `NPU_OP_LOAD, `NPU_ST_OK,
                   "overlapped load record");
        expect_rec(4, 16'h0030, 8'd4, `NPU_OP_FENCE, `NPU_ST_OK,
                   "fence record after both tasks");

        chk(active_slots == 32'd5, "five live slots after test C");

        // ------------------------------------------------------------
        // D. duplicate job_id on a free slot
        // ------------------------------------------------------------
        mk_load(16'h0013, 8'd8, 8'd3, 32'd32);
        send_cmd();
        expect_rec(5, 16'h0013, `NPU_SLOT_NONE, `NPU_OP_LOAD,
                   `NPU_ST_BAD_FIELD, "duplicate job rejected slot-less");

        // ------------------------------------------------------------
        // E. occupied event slot
        // ------------------------------------------------------------
        mk_load(16'h0020, 8'd3, 8'd3, 32'd32);
        send_cmd();
        expect_rec(6, 16'h0020, `NPU_SLOT_NONE, `NPU_OP_LOAD,
                   `NPU_ST_EVENT_BUSY, "occupied slot rejected");

        // ------------------------------------------------------------
        // F. out-of-range event slot
        // ------------------------------------------------------------
        mk_load(16'h0021, 8'd16, 8'd3, 32'd32);
        send_cmd();
        expect_rec(7, 16'h0021, `NPU_SLOT_NONE, `NPU_OP_LOAD,
                   `NPU_ST_RANGE, "out-of-range slot rejected");

        // ------------------------------------------------------------
        // G. a failed task poisons the next FENCE, the one after clears
        // ------------------------------------------------------------
        dma_fail = 1'b1;
        mk_load(16'h0040, 8'd5, 8'd3, 32'd32);
        send_cmd();
        expect_rec(8, 16'h0040, 8'd5, `NPU_OP_LOAD, `NPU_ST_INTERNAL,
                   "failed load record");
        dma_fail = 1'b0;

        mk_fence(16'h0031, 8'd6);
        send_cmd();
        expect_rec(9, 16'h0031, 8'd6, `NPU_OP_FENCE, `NPU_ST_DEP_FAILED,
                   "fence reports barrier failure");

        mk_fence(16'h0032, 8'd7);
        send_cmd();
        expect_rec(10, 16'h0032, 8'd7, `NPU_OP_FENCE, `NPU_ST_OK,
                   "fence clears the barrier");

        // ------------------------------------------------------------
        // H. EVENT_RELEASE frees a slot and the job id can be reused
        // ------------------------------------------------------------
        mk_release(16'h0050, 8'd0, 16'h0010);
        send_cmd();
        expect_rec(11, 16'h0050, `NPU_SLOT_NONE, `NPU_OP_RELEASE,
                   `NPU_ST_OK, "release record");

        mk_load(16'h0010, 8'd0, 8'd3, 32'd32);
        send_cmd();
        expect_rec(12, 16'h0010, 8'd0, `NPU_OP_LOAD, `NPU_ST_OK,
                   "job id reused after release");

        // ------------------------------------------------------------
        // I. release of a free target reports BAD_FIELD
        // ------------------------------------------------------------
        mk_release(16'h0051, 8'd9, 16'h0010);
        send_cmd();
        expect_rec(13, 16'h0051, `NPU_SLOT_NONE, `NPU_OP_RELEASE,
                   `NPU_ST_BAD_FIELD, "release of free target");

        // ------------------------------------------------------------
        // J. SFU task completes
        // ------------------------------------------------------------
        mk_sfu(16'h0060, 8'd10, 8'd0, 8'd1, 32'd8);
        send_cmd();
        expect_rec(14, 16'h0060, 8'd10, `NPU_OP_SFU, `NPU_ST_OK,
                   "sfu record");

        repeat (5) @(posedge clk);
        chk(fatal_latched === 1'b0, "no fatal in control plane");
        chk(fatal === 1'b0, "scheduler not fatal");

        if (errors == 0)
            $display("PASS: tb_control_plane");
        else
            $display("FAIL: tb_control_plane (%0d errors)", errors);
        $finish;
    end

    initial begin
        #100000;
        $display("TIMEOUT: tb_control_plane");
        $finish;
    end

endmodule

`default_nettype wire

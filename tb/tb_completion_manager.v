//=============================================================================
// tb_completion_manager : allocation, completion, release, CPL and faults
//=============================================================================
`timescale 1ns/1ps
`default_nettype none

`include "npu_defs.vh"

module tb_completion_manager;
    reg clk = 1'b0;
    reg rst_n = 1'b0;

    reg  [4:0]        src_valid = 5'd0;
    wire [4:0]        src_ready;
    reg  [5*128-1:0]  src_event = {5*128{1'b0}};

    reg               alloc_valid = 1'b0;
    wire              alloc_ready;
    reg  [7:0]        alloc_slot = 8'd0;
    reg  [15:0]       alloc_job = 16'd0;
    reg  [7:0]        alloc_opcode = 8'd0;
    reg               alloc_irq = 1'b0;

    reg  [7:0]        lookup_slot = 8'd0;
    reg  [15:0]       lookup_job = 16'd0;
    wire [2:0]        lookup_state;

    reg  [7:0]        slot_probe_slot = 8'd0;
    wire              slot_probe_free;
    reg  [15:0]       id_query_job = 16'd0;
    wire              id_query_busy;

    reg  [1:0]        admin_action = `NPU_ADMIN_NONE;
    reg  [7:0]        admin_target_slot = 8'd0;
    reg  [15:0]       admin_target_job = 16'd0;
    reg               admin_irq = 1'b0;

    wire              cpl_valid;
    reg               cpl_ready = 1'b0;
    wire [127:0]      cpl_data;

    wire              retire_valid;
    wire [127:0]      retire_event;

    wire              irq;
    reg               fatal_in = 1'b0;
    wire              fatal_latched;
    wire [31:0]       cpl_level;
    wire [31:0]       active_slots;

    integer errors = 0;

    always #5 clk = ~clk;

    completion_manager #(
        .EVENT_SLOTS(16), .CPL_DEPTH(8)
    ) dut (
        .clk_core(clk), .rst_core_n(rst_n),
        .src_valid(src_valid), .src_ready(src_ready), .src_event(src_event),
        .alloc_valid(alloc_valid), .alloc_ready(alloc_ready),
        .alloc_slot(alloc_slot), .alloc_job(alloc_job),
        .alloc_opcode(alloc_opcode), .alloc_irq(alloc_irq),
        .lookup_slot(lookup_slot), .lookup_job(lookup_job),
        .lookup_state(lookup_state),
        .slot_probe_slot(slot_probe_slot), .slot_probe_free(slot_probe_free),
        .id_query_job(id_query_job), .id_query_busy(id_query_busy),
        .admin_action(admin_action),
        .admin_target_slot(admin_target_slot),
        .admin_target_job(admin_target_job),
        .admin_irq(admin_irq),
        .cpl_valid(cpl_valid), .cpl_ready(cpl_ready), .cpl_data(cpl_data),
        .retire_valid(retire_valid), .retire_event(retire_event),
        .irq(irq),
        .fatal_in(fatal_in), .fatal_latched(fatal_latched),
        .cpl_level(cpl_level), .active_slots(active_slots)
    );

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

    function [127:0] ev_word;
        input [15:0] job;
        input [7:0]  slot;
        input [7:0]  op;
        input [15:0] status;
        begin
            ev_word          = 128'd0;
            ev_word[15:0]    = job;
            ev_word[23:16]   = slot;
            ev_word[31:24]   = op;
            ev_word[47:32]   = status;
        end
    endfunction

    reg retire_seen;
    reg [127:0] retire_cap;
    always @(posedge clk) begin
        if (retire_valid) begin
            retire_seen <= 1'b1;
            retire_cap  <= retire_event;
        end
    end

    task alloc_one;
        input [7:0]  slot;
        input [15:0] job;
        input [7:0]  op;
        input        irq;
        begin
            @(negedge clk);
            alloc_valid  = 1'b1;
            alloc_slot   = slot;
            alloc_job    = job;
            alloc_opcode = op;
            alloc_irq    = irq;
            #1;
            chk(alloc_ready, "alloc_ready expected");
            @(posedge clk);
            @(negedge clk);
            alloc_valid = 1'b0;
            @(posedge clk);
            #1;
        end
    endtask

    // Drive one source until it is accepted, then release it.
    task send_src;
        input integer src;
        input [127:0] ev;
        begin
            @(negedge clk);
            src_event[src*128 +: 128] = ev;
            src_valid[src] = 1'b1;
            #1;
            while (!src_ready[src]) begin
                @(negedge clk);
                #1;
            end
            @(posedge clk);
            @(negedge clk);
            src_valid[src] = 1'b0;
            @(posedge clk);
            #1;
        end
    endtask

    task send_admin;
        input [1:0]  action;
        input [7:0]  tslot;
        input [15:0] tjob;
        begin
            @(negedge clk);
            admin_action      = action;
            admin_target_slot = tslot;
            admin_target_job  = tjob;
            admin_irq         = 1'b0;
            src_event[4*128 +: 128] = ev_word(16'd0, `NPU_SLOT_NONE,
                                              `NPU_OP_RELEASE, `NPU_ST_OK);
            src_valid[4] = 1'b1;
            #1;
            while (!src_ready[4]) begin
                @(negedge clk);
                #1;
            end
            @(posedge clk);
            @(negedge clk);
            src_valid[4] = 1'b0;
            @(posedge clk);
            #1;
        end
    endtask

    task pop_cpl;
        input [127:0] exp;
        input [255:0] msg;
        begin
            chk(cpl_valid, "cpl_valid expected");
            chk(cpl_data === exp, msg);
            @(negedge clk); cpl_ready = 1'b1;
            @(posedge clk);
            @(negedge clk); cpl_ready = 1'b0;
            @(posedge clk);
            #1;
        end
    endtask

    task lookup;
        input [7:0]  slot;
        input [15:0] job;
        input [2:0]  exp;
        input [255:0] msg;
        begin
            lookup_slot = slot;
            lookup_job  = job;
            #1;
            chk(lookup_state === exp, msg);
        end
    endtask

    integer n;

    initial begin
        repeat (3) @(posedge clk);
        rst_n = 1'b1;
        repeat (2) @(posedge clk);

        // ------------------------------------------------------------
        // 1. reset state
        // ------------------------------------------------------------
        chk(cpl_valid === 1'b0, "cpl empty after reset");
        chk(active_slots == 32'd0, "no active slots after reset");
        chk(irq === 1'b0, "irq low after reset");
        lookup(8'd0, 16'd0, `NPU_LOOKUP_FREE, "slot 0 free");
        lookup(8'd31, 16'd5, `NPU_LOOKUP_FREE, "out of range reads free");

        slot_probe_slot = 8'd3; #1;
        chk(slot_probe_free === 1'b1, "slot 3 free before alloc");

        // ------------------------------------------------------------
        // 2. allocation and lookup
        // ------------------------------------------------------------
        alloc_one(8'd3, 16'h0101, `NPU_OP_MATMUL, 1'b1);
        chk(active_slots == 32'd1, "one active slot");
        lookup(8'd3, 16'h0101, `NPU_LOOKUP_RUNNING, "slot 3 running");
        lookup(8'd3, 16'h0102, `NPU_LOOKUP_MISMATCH, "wrong job mismatch");

        slot_probe_slot = 8'd3; #1;
        chk(slot_probe_free === 1'b0, "slot 3 busy");
        slot_probe_slot = 8'd4; #1;
        chk(slot_probe_free === 1'b1, "slot 4 free");

        id_query_job = 16'h0101; #1;
        chk(id_query_busy === 1'b1, "job in table is busy");
        id_query_job = 16'h0102; #1;
        chk(id_query_busy === 1'b0, "unknown job is not busy");

        // allocating an occupied slot must not be ready
        @(negedge clk);
        alloc_valid = 1'b1; alloc_slot = 8'd3; alloc_job = 16'h0103;
        alloc_opcode = `NPU_OP_VECTOR; alloc_irq = 1'b0;
        #1;
        chk(alloc_ready === 1'b0, "alloc of busy slot not ready");
        @(negedge clk); alloc_valid = 1'b0;
        @(posedge clk); #1;

        // ------------------------------------------------------------
        // 3. normal completion
        // ------------------------------------------------------------
        retire_seen = 1'b0;
        send_src(1, ev_word(16'h0101, 8'd3, `NPU_OP_MATMUL, `NPU_ST_OK));
        chk(retire_seen === 1'b1, "retire pulse seen");
        chk(retire_cap === ev_word(16'h0101, 8'd3, `NPU_OP_MATMUL, `NPU_ST_OK),
            "retire event matches");
        chk(cpl_valid === 1'b1, "completion in CPL");
        chk(cpl_level == 32'd1, "CPL level 1");
        chk(irq === 1'b1, "irq raised by irq completion");
        lookup(8'd3, 16'h0101, `NPU_LOOKUP_OK, "slot 3 OK");
        chk(active_slots == 32'd1, "completed slot still active");

        id_query_job = 16'h0101; #1;
        chk(id_query_busy === 1'b1, "job still visible in CPL");

        pop_cpl(ev_word(16'h0101, 8'd3, `NPU_OP_MATMUL, `NPU_ST_OK),
                "CPL record");
        chk(cpl_level == 32'd0, "CPL emptied");
        chk(irq === 1'b0, "irq cleared by CPL pop");
        id_query_job = 16'h0101; #1;
        chk(id_query_busy === 1'b1,
            "job stays visible in the event table until release");

        // ------------------------------------------------------------
        // 4. EVENT_RELEASE of a completed target
        // ------------------------------------------------------------
        send_admin(`NPU_ADMIN_RELEASE, 8'd3, 16'h0101);
        lookup(8'd3, 16'h0101, `NPU_LOOKUP_FREE, "target slot freed");
        chk(active_slots == 32'd0, "no active slots after release");
        pop_cpl(ev_word(16'd0, `NPU_SLOT_NONE, `NPU_OP_RELEASE, `NPU_ST_OK),
                "release record");
        id_query_job = 16'h0101; #1;
        chk(id_query_busy === 1'b0,
            "job gone after release and CPL consumption");

        // ------------------------------------------------------------
        // 5. EVENT_RELEASE of a free/mismatched target is BAD_FIELD
        // ------------------------------------------------------------
        send_admin(`NPU_ADMIN_RELEASE, 8'd3, 16'h0101);
        pop_cpl(ev_word(16'd0, `NPU_SLOT_NONE, `NPU_OP_RELEASE,
                        `NPU_ST_BAD_FIELD),
                "release of free target BAD_FIELD");
        chk(active_slots == 32'd0, "release of free target changes nothing");

        // ------------------------------------------------------------
        // 6. release waits for a running target without blocking others
        // ------------------------------------------------------------
        alloc_one(8'd5, 16'h0202, `NPU_OP_VECTOR, 1'b0);

        @(negedge clk);
        admin_action      = `NPU_ADMIN_RELEASE;
        admin_target_slot = 8'd5;
        admin_target_job  = 16'h0202;
        admin_irq         = 1'b0;
        src_event[4*128 +: 128] = ev_word(16'd0, `NPU_SLOT_NONE,
                                          `NPU_OP_RELEASE, `NPU_ST_OK);
        src_valid[4] = 1'b1;
        for (n = 0; n < 3; n = n + 1) begin
            #1;
            chk(src_ready[4] === 1'b0, "release blocked while target runs");
            @(negedge clk);
        end

        // Another source must still be able to commit while the release is
        // pending (src_valid[4] stays high during this call).  The release
        // then wins on the first edge after the target completes, which is
        // the last edge inside send_src.
        send_src(2, ev_word(16'h0202, 8'd5, `NPU_OP_VECTOR, `NPU_ST_OK));
        src_valid[4] = 1'b0;
        @(posedge clk); #1;
        lookup(8'd5, 16'h0202, `NPU_LOOKUP_FREE, "target freed after wait");
        pop_cpl(ev_word(16'h0202, 8'd5, `NPU_OP_VECTOR, `NPU_ST_OK),
                "vector record");
        pop_cpl(ev_word(16'd0, `NPU_SLOT_NONE, `NPU_OP_RELEASE, `NPU_ST_OK),
                "release record after wait");

        // ------------------------------------------------------------
        // 7. CPL backpressure stalls a source, not the event
        // ------------------------------------------------------------
        alloc_one(8'd0, 16'h0404, `NPU_OP_VECTOR, 1'b0);
        for (n = 8; n < 16; n = n + 1)
            alloc_one(n[7:0], 16'h0300 + n[15:0], `NPU_OP_VECTOR, 1'b0);
        for (n = 8; n < 16; n = n + 1)
            send_src(1, ev_word(16'h0300 + n[15:0], n[7:0],
                                `NPU_OP_VECTOR, `NPU_ST_OK));
        chk(cpl_level == 32'd8, "CPL full");

        // one more completion must be held back
        @(negedge clk);
        src_event[0 +: 128] = ev_word(16'h0404, 8'd0, `NPU_OP_VECTOR,
                                      `NPU_ST_OK);
        src_valid[0] = 1'b1;
        repeat (3) begin
            #1;
            chk(src_ready[0] === 1'b0, "source held while CPL full");
            @(negedge clk);
        end
        // make room with a single pop pulse
        @(negedge clk); cpl_ready = 1'b1;
        @(negedge clk); cpl_ready = 1'b0;
        #1;
        chk(src_ready[0] === 1'b1, "held source granted once space frees");
        @(posedge clk);
        @(negedge clk);
        src_valid[0] = 1'b0;
        @(posedge clk); #1;
        chk(cpl_level == 32'd8, "CPL refilled by held completion");

        // drain the whole queue
        cpl_ready = 1'b1;
        repeat (12) @(posedge clk);
        @(negedge clk); cpl_ready = 1'b0;
        @(posedge clk); #1;
        chk(cpl_level == 32'd0, "CPL drained");

        // ------------------------------------------------------------
        // 8. duplicate completion latches fatal
        // ------------------------------------------------------------
        alloc_one(8'd7, 16'h0505, `NPU_OP_SFU, 1'b0);
        send_src(3, ev_word(16'h0505, 8'd7, `NPU_OP_SFU, `NPU_ST_OK));
        chk(cpl_level == 32'd1, "one record queued");
        chk(fatal_latched === 1'b0, "no fault yet");
        cpl_ready = 1'b1;
        @(posedge clk);
        @(negedge clk); cpl_ready = 1'b0;
        @(posedge clk); #1;

        send_src(3, ev_word(16'h0505, 8'd7, `NPU_OP_SFU, `NPU_ST_OK));
        chk(fatal_latched === 1'b1, "duplicate completion latches fatal");
        chk(cpl_level == 32'd0, "fault does not enqueue a record");
        chk(alloc_ready === 1'b0, "allocation blocked in fault");
        chk(irq === 1'b1, "fatal raises irq");

        if (errors == 0)
            $display("PASS: tb_completion_manager");
        else
            $display("FAIL: tb_completion_manager (%0d errors)", errors);
        $finish;
    end

    initial begin
        #40000;
        $display("TIMEOUT: tb_completion_manager");
        $finish;
    end

endmodule
`default_nettype wire

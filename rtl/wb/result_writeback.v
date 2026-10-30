//=============================================================================
// result_writeback : MPU result slots, format conversion and strobed writes
//
//   Follows docs/npu_integration/09_result_writeback.md.
//
//   The core result interface has no ready: acc_valid is captured in the
//   cycle it appears, into a slot reserved by wb_job.  Slots are consumed in
//   capture order by a one-element-at-a-time serializer that converts FP32 to
//   the destination format and writes each element with byte strobes at an
//   element (not word) aligned destination address.
//
//   Slot accounting (C = RESULT_SLOTS):
//       reserve T : R += T      (on job_fire, requires R+O+T <= C)
//       capture   : R -= 1, O += 1
//       release   : O -= 1      (after the last write of the tile is acked)
//   Release is conservative, so O includes the slot being serialized; a slot
//   is never reused while the serializer can still read it.
//
//   Single job context: a context is held from job_fire until done is
//   accepted, so job_ready is low while a job is in flight.  done requires
//   every tile captured and written and R = O = 0.
//
//   Errors (numeric overflow/invalid, memory error, address range, capture
//   without reservation) end the job with a non-OK status and latch fatal;
//   the slot ring is not reused until reset.
//=============================================================================
`timescale 1ns/1ps
`default_nettype none

`include "npu_defs.vh"

module result_writeback #(
    parameter MEM_DATA_W   = 256,
    parameter RESULT_SLOTS = 4
)(
    input  wire        clk_core,
    input  wire        rst_core_n,

    input  wire        job_valid,
    output wire        job_ready,
    input  wire [511:0] job_cmd,

    // core result pulse, no back pressure
    input  wire         acc_valid,
    input  wire [4095:0] acc_data,

    // scratchpad write channel (client RESULT_WB)
    output wire        mem_req_valid,
    input  wire        mem_req_ready,
    output wire [`NPU_REQ_W(MEM_DATA_W, MEM_DATA_W/8)-1:0] mem_req,
    input  wire        mem_rsp_valid,
    output wire        mem_rsp_ready,
    input  wire [`NPU_RSP_W(MEM_DATA_W)-1:0] mem_rsp,

    // job summary, held until accepted
    output wire        done_valid,
    input  wire        done_ready,
    output wire [15:0] done_status,
    output wire [31:0] done_tiles,

    output wire [$clog2(RESULT_SLOTS+1)-1:0] reserved_count,
    output wire [$clog2(RESULT_SLOTS+1)-1:0] occupied_count,
    output reg         fatal
);

    localparam DW    = MEM_DATA_W;
    localparam B     = MEM_DATA_W / 8;
    localparam REQ_W = `NPU_REQ_W(DW, B);
    localparam RSP_W = `NPU_RSP_W(DW);
    localparam CW    = $clog2(RESULT_SLOTS + 1);
    localparam PW    = (RESULT_SLOTS > 1) ? $clog2(RESULT_SLOTS) : 1;

    localparam W_IDLE  = 3'd0;
    localparam W_LOCK  = 3'd1;
    localparam W_BLD   = 3'd2;
    localparam W_CONV  = 3'd3;
    localparam W_REQ   = 3'd4;
    localparam W_ACK   = 3'd5;
    localparam W_DONE  = 3'd6;      // done asserted (OK or error), held
    localparam W_FAULT = 3'd7;      // unrecoverable, reset required

    localparam F_FP16 = `NPU_FMT_FP16;
    localparam F_FP32 = `NPU_FMT_FP32;

    // ------------------------------------------------------------------
    // job command decode
    // ------------------------------------------------------------------
    wire [15:0] d_tiles = job_cmd[175:160];
    wire [1:0]  d_dfmt  = job_cmd[93:92];   // formats[5:4]
    wire [7:0]  d_dbuf  = job_cmd[87:80];

    // ------------------------------------------------------------------
    // slot storage and context
    // ------------------------------------------------------------------
    reg [4095:0] slot_dat [0:RESULT_SLOTS-1];
    reg [15:0]   slot_tile[0:RESULT_SLOTS-1];
    reg [7:0]    slot_cols[0:RESULT_SLOTS-1];
    reg          slot_vld [0:RESULT_SLOTS-1];

    reg [PW-1:0] cap_ptr, rel_ptr;

    reg          ctx_active;
    reg [15:0]   job_tiles;
    reg [2:0]    job_mlen;
    reg [3:0]    job_ngrp;
    reg [31:0]   job_nvalid;
    reg [31:0]   job_base;
    reg [31:0]   job_stride;
    reg [1:0]    job_dfmt;
    reg [7:0]    job_dbuf;

    reg [CW-1:0] rsv_cnt, occ_cnt;
    reg [15:0]   cap_tile;          // tile index of the next acc pulse
    reg [15:0]   cap_cnt, wr_cnt;   // tiles captured / tiles written

    assign reserved_count = rsv_cnt;
    assign occupied_count = occ_cnt;

    reg [3:0]  state;
    reg        done_r;
    reg [15:0] done_status_r;
    reg [31:0] done_tiles_r;

    assign done_valid  = done_r;
    assign done_status = done_status_r;
    assign done_tiles  = done_tiles_r;

    // ------------------------------------------------------------------
    // job admission
    // ------------------------------------------------------------------
    wire [CW-1:0] slot_sum  = rsv_cnt + occ_cnt;
    wire [32:0]   req_total = {28'd0, slot_sum} + {17'd0, d_tiles};

    wire d_tiles_ok = (d_tiles != 16'd0) && (d_tiles <= RESULT_SLOTS);
    wire d_fmt_ok   = (d_dfmt != `NPU_FMT_INT4);

    wire job_free  = !ctx_active && (state == W_IDLE) && !fatal;
    wire room_ok   = d_tiles_ok && (req_total <= RESULT_SLOTS);

    assign job_ready = job_free && d_fmt_ok && room_ok;
    wire   job_fire  = job_valid && job_ready;

    // ------------------------------------------------------------------
    // capture-side arithmetic
    // ------------------------------------------------------------------
    wire [31:0] g4_w    = {26'd0, job_ngrp, 2'b00};
    wire [31:0] cstart  = g4_w * {16'd0, cap_tile};
    wire [31:0] nrem    = (job_nvalid > cstart) ? (job_nvalid - cstart)
                                                : 32'd0;
    wire [31:0] cols32  = (nrem >= g4_w) ? g4_w :
                          (g4_w == 32'd0) ? 32'd0 : nrem;

    wire cap_ok   = !fatal && ctx_active && (rsv_cnt != 32'd0);
    wire cap_fire = acc_valid && cap_ok;

    // A pulse with no reservation is a protocol fault.  With a context it is
    // reported through the job summary; without one there is nobody to tell.
    wire cap_stray   = acc_valid && !ctx_active && !fatal;
    wire cap_orphan  = acc_valid && ctx_active && (rsv_cnt == 0) && !fatal;

    // ------------------------------------------------------------------
    // serializer element coordinates
    // ------------------------------------------------------------------
    reg [1:0]  m_r;
    reg [4:0]  c_r;
    reg [7:0]  cur_cols;
    reg [7:0]  cur_cbase;
    reg [4095:0] cur_dat;

    wire [5:0] c_nxt = {1'b0, c_r} + 6'd1;
    wire [2:0] m_nxt = m_r + 3'd1;
    wire last_col = (c_nxt >= {2'd0, cur_cols});
    wire last_row = (m_nxt >= job_mlen);
    wire last_elem = last_col && last_row;

    wire [11:0] elem_idx = {m_r, c_r, 5'b0};
    wire [31:0] elem_val = cur_dat[elem_idx +: 32];

    // ------------------------------------------------------------------
    // format converter (one element in flight)
    // ------------------------------------------------------------------
    wire        cvt_in_ready, cvt_out_valid;
    wire [31:0] cvt_out_bits;
    wire [7:0]  cvt_flags;

    wire cvt_in_valid = (state == W_BLD) && (cur_cols != 8'd0);

    format_converter u_cvt (
        .clk(clk_core), .rst_n(rst_core_n),
        .in_valid(cvt_in_valid), .in_ready(cvt_in_ready),
        .in_bits(elem_val),
        .src_fmt(F_FP32), .dst_fmt(job_dfmt), .in_tag(8'h00),
        .out_valid(cvt_out_valid), .out_ready(state == W_CONV),
        .out_bits(cvt_out_bits), .out_tag(), .fp_flags(cvt_flags)
    );

    // ------------------------------------------------------------------
    // destination address / strobes
    // ------------------------------------------------------------------
    wire [7:0]  col_w   = cur_cbase + {3'd0, c_r};
    wire [2:0]  elem_b  = (job_dfmt == F_FP32) ? 3'd4 : 3'd2;
    wire [11:0] col_off = {4'd0, col_w} * elem_b;
    wire [33:0] row_off = {32'd0, m_r} * {2'd0, job_stride};
    wire [39:0] ea_w    = {8'd0, job_base} + {6'd0, row_off} + {28'd0, col_off};
    wire        ea_hi   = |ea_w[39:32];
    // D_base must be element aligned: an element may not straddle a word
    wire        ea_cross = ({27'd0, ea_w[4:0]} + {29'd0, elem_b}) > 32'd32;

    wire [B-1:0] strb_one = (32'd1 << {30'd0, elem_b}) - 32'd1;

    reg [DW-1:0] wq_data;
    reg [B-1:0]  wq_strb;
    reg [31:0]   wq_off;

    assign mem_req_valid = (state == W_REQ);
    assign mem_req = {8'h5A, wq_strb, wq_data, wq_off, job_dbuf, 1'b1};
    assign mem_rsp_ready = (state == W_ACK);

    // mem_rsp_t = {reason[8], error[1], tag[8], rdata[DW]}
    wire rsp_err    = mem_rsp[DW+8];
    wire ack_fire   = mem_rsp_valid && mem_rsp_ready;
    wire rel_fire_ack  = (state == W_ACK) && ack_fire && !rsp_err &&
                         last_elem;
    wire rel_fire_lock = (state == W_LOCK) && (slot_cols[rel_ptr] == 8'd0);
    wire rel_fire      = rel_fire_ack || rel_fire_lock;

    // ------------------------------------------------------------------
    // main sequential block: FSM, capture, slot ring, counters
    // ------------------------------------------------------------------
    integer i;

    always @(posedge clk_core or negedge rst_core_n) begin
        if (!rst_core_n) begin
            state         <= W_IDLE;
            ctx_active    <= 1'b0;
            job_tiles     <= 16'd0;
            job_mlen      <= 3'd0;
            job_ngrp      <= 4'd0;
            job_nvalid    <= 32'd0;
            job_base      <= 32'd0;
            job_stride    <= 32'd0;
            job_dfmt      <= 2'd0;
            job_dbuf      <= 8'd0;
            rsv_cnt       <= {CW{1'b0}};
            occ_cnt       <= {CW{1'b0}};
            cap_tile      <= 16'd0;
            cap_cnt       <= 16'd0;
            wr_cnt        <= 16'd0;
            cap_ptr       <= {PW{1'b0}};
            rel_ptr       <= {PW{1'b0}};
            done_r        <= 1'b0;
            done_status_r <= `NPU_ST_OK;
            done_tiles_r  <= 32'd0;
            fatal         <= 1'b0;
            m_r           <= 2'd0;
            c_r           <= 5'd0;
            cur_cols      <= 8'd0;
            cur_cbase     <= 8'd0;
            cur_dat       <= 4096'd0;
            wq_data       <= {DW{1'b0}};
            wq_strb       <= {B{1'b0}};
            wq_off        <= 32'd0;
            for (i = 0; i < RESULT_SLOTS; i = i + 1) begin
                slot_dat[i]  <= 4096'd0;
                slot_tile[i] <= 16'd0;
                slot_cols[i] <= 8'd0;
                slot_vld[i]  <= 1'b0;
            end
        end else begin
            // ---- result capture (independent of the FSM state) ----
            if (cap_fire) begin
                slot_dat[cap_ptr]  <= acc_data;
                slot_tile[cap_ptr] <= cap_tile;
                slot_cols[cap_ptr] <= cols32[7:0];
                slot_vld[cap_ptr]  <= 1'b1;
                cap_ptr  <= (cap_ptr == RESULT_SLOTS - 1) ? {PW{1'b0}}
                                                          : cap_ptr + 1'b1;
                cap_tile <= cap_tile + 16'd1;
                cap_cnt  <= cap_cnt + 16'd1;
            end

            // ---- slot release ----
            if (rel_fire) begin
                slot_vld[rel_ptr] <= 1'b0;
                rel_ptr <= (rel_ptr == RESULT_SLOTS - 1) ? {PW{1'b0}}
                                                         : rel_ptr + 1'b1;
                wr_cnt  <= wr_cnt + 16'd1;
            end

            // ---- reservation / occupancy accounting ----
            if (!fatal) begin
                rsv_cnt <= rsv_cnt + (job_fire ? d_tiles[CW-1:0] : {CW{1'b0}})
                                   - (cap_fire ? {{CW-1{1'b0}}, 1'b1}
                                               : {CW{1'b0}});
                occ_cnt <= occ_cnt + (cap_fire ? {{CW-1{1'b0}}, 1'b1}
                                               : {CW{1'b0}})
                                   - (rel_fire ? {{CW-1{1'b0}}, 1'b1}
                                               : {CW{1'b0}});
            end

            // ---- serializer FSM ----
            case (state)
            W_IDLE: begin
                if (job_fire) begin
                    ctx_active <= 1'b1;
                    job_tiles  <= d_tiles;
                    job_mlen   <= job_cmd[130:128];
                    job_ngrp   <= job_cmd[139:136];
                    job_nvalid <= job_cmd[255:224];
                    job_base   <= job_cmd[383:352];
                    job_stride <= job_cmd[415:384];
                    job_dfmt   <= d_dfmt;
                    job_dbuf   <= d_dbuf;
                    cap_tile   <= 16'd0;
                    cap_cnt    <= 16'd0;
                    wr_cnt     <= 16'd0;
                end else if (ctx_active) begin
                    if ((cap_cnt == job_tiles) && (wr_cnt == job_tiles) &&
                        (rsv_cnt == {CW{1'b0}}) && (occ_cnt == {CW{1'b0}})) begin
                        done_r        <= 1'b1;
                        done_status_r <= `NPU_ST_OK;
                        done_tiles_r  <= {16'd0, wr_cnt};
                        state         <= W_DONE;
                    end else if (occ_cnt != {CW{1'b0}}) begin
                        state <= W_LOCK;
                    end
                end
            end

            W_LOCK: begin
                cur_dat   <= slot_dat[rel_ptr];
                cur_cols  <= slot_cols[rel_ptr];
                cur_cbase <= slot_tile[rel_ptr][6:0] * job_ngrp * 4;
                m_r       <= 2'd0;
                c_r       <= 5'd0;
                state     <= (slot_cols[rel_ptr] == 8'd0) ? W_IDLE : W_BLD;
            end

            W_BLD: begin
                if (cvt_in_ready) state <= W_CONV;
            end

            W_CONV: begin
                if (cvt_out_valid) begin
                    if (cvt_flags[0] || cvt_flags[1]) begin
                        // invalid / overflow: the element is not written
                        done_r        <= 1'b1;
                        done_status_r <= `NPU_ST_NUMERIC;
                        done_tiles_r  <= {16'd0, wr_cnt};
                        fatal         <= 1'b1;
                        state         <= W_DONE;
                    end else if (ea_hi) begin
                        done_r        <= 1'b1;
                        done_status_r <= `NPU_ST_RANGE;
                        done_tiles_r  <= {16'd0, wr_cnt};
                        fatal         <= 1'b1;
                        state         <= W_DONE;
                    end else if (ea_cross) begin
                        done_r        <= 1'b1;
                        done_status_r <= `NPU_ST_ALIGN;
                        done_tiles_r  <= {16'd0, wr_cnt};
                        fatal         <= 1'b1;
                        state         <= W_DONE;
                    end else begin
                        wq_off  <= {ea_w[31:5], 5'd0};
                        wq_data <= {{(DW-32){1'b0}}, cvt_out_bits} <<
                                   {ea_w[4:0], 3'd0};
                        wq_strb <= strb_one << ea_w[4:0];
                        state   <= W_REQ;
                    end
                end
            end

            W_REQ: begin
                if (mem_req_ready) state <= W_ACK;
            end

            W_ACK: begin
                if (ack_fire) begin
                    if (rsp_err) begin
                        done_r        <= 1'b1;
                        done_status_r <= `NPU_ST_MEM_ACCESS;
                        done_tiles_r  <= {16'd0, wr_cnt};
                        fatal         <= 1'b1;
                        state         <= W_DONE;
                    end else if (last_elem) begin
                        state <= W_IDLE;
                    end else begin
                        if (last_col) begin
                            c_r <= 5'd0;
                            m_r <= m_nxt[1:0];
                        end else begin
                            c_r <= c_r + 5'd1;
                        end
                        state <= W_BLD;
                    end
                end
            end

            W_DONE: begin
                if (done_ready) begin
                    done_r     <= 1'b0;
                    ctx_active <= 1'b0;
                    state      <= fatal ? W_FAULT : W_IDLE;
                end
            end

            W_FAULT: begin
                // reset required
            end

            default: state <= W_FAULT;
            endcase

            // ---- protocol fault: pulse with no reservation ----
            if (cap_stray) begin
                fatal <= 1'b1;
                state <= W_FAULT;
            end else if (cap_orphan && (state != W_DONE) &&
                         (state != W_FAULT)) begin
                done_r        <= 1'b1;
                done_status_r <= `NPU_ST_INTERNAL;
                done_tiles_r  <= {16'd0, wr_cnt};
                fatal         <= 1'b1;
                state         <= W_DONE;
            end
        end
    end

endmodule

`default_nettype wire

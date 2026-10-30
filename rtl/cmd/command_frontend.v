//=============================================================================
// command_frontend : 512-bit command FIFO + static decode/check
//
//   Follows docs/npu_integration/03_command_frontend.md.
//
//   - accept_enable gates enqueue only; dequeue continues while disabled so an
//     accepted queue can be drained after RUN is lowered.
//   - Every enqueue is checked statically.  A bad command is still queued in
//     order and carries issue_error to the scheduler.
//   - A job_id that equals an older queued command (including the one being
//     popped this cycle) or the scheduler's held command sets
//     issue_identity_error.  The command is still queued; the scheduler turns
//     it into a slot-less rejection.
//   - No partial checks are claimed: buffer capacity and runtime effects are
//     checked by the scheduler and engines.
//=============================================================================
`timescale 1ns/1ps
`default_nettype none

`include "npu_defs.vh"

module command_frontend #(
    parameter EVENT_SLOTS   = 16,
    parameter RESULT_SLOTS  = 2,
    parameter WBUF_K_STEPS  = 256,
    parameter MEM_DATA_W    = 256,
    parameter V_LANES       = 4,
    parameter CMD_FIFO_DEPTH= 8
)(
    input  wire                    clk_core,
    input  wire                    rst_core_n,

    input  wire                    accept_enable,

    input  wire                    cmd_valid,
    output wire                    cmd_ready,
    input  wire [511:0]            cmd_data,

    output wire                    issue_valid,
    input  wire                    issue_ready,
    output wire [511:0]            issue_cmd,
    output wire [15:0]             issue_error,
    output wire                    issue_identity_error,

    input  wire                    held_valid,
    input  wire [15:0]             held_job,

    output wire [31:0]             fifo_level,
    output reg  [63:0]             accepted_count,
    output reg  [63:0]             rejected_count
);

    localparam AW = $clog2(CMD_FIFO_DEPTH);
    localparam LW = $clog2(CMD_FIFO_DEPTH+1);
    localparam B  = MEM_DATA_W/8;

    // ------------------------------------------------------------------
    // static checker
    // ------------------------------------------------------------------
    function [15:0] check_static;
        input [511:0] c;
        reg [7:0]  op, sub, f_src0, f_src1, f_dst, f_acc;
        reg [11:0] fmt;
        reg [31:0] p0,p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11;
        reg [31:0] m_len, g, mode, tiles, ksteps, kvalid, nvalid, kstep;
        reg [31:0] s_steps, w_bus;
        reg [63:0] tmp64;
        reg [5:0]  vsub;
        reg [1:0]  bcast;
        reg [7:0]  lay_g;
        begin
            check_static = `NPU_ST_OK;

            op  = c[15:8];
            sub = c[23:16];
            fmt = c[95:88];
            f_src0 = fmt[1:0];
            f_src1 = fmt[3:2];
            f_dst  = fmt[5:4];
            f_acc  = fmt[7:6];

            p0  = c[159:128];
            p1  = c[191:160];
            p2  = c[223:192];
            p3  = c[255:224];
            p4  = c[287:256];
            p5  = c[319:288];
            p6  = c[351:320];
            p7  = c[383:352];
            p8  = c[415:384];
            p9  = c[447:416];
            p10 = c[479:448];
            p11 = c[511:480];

            // ---- common header ----
            if (c[7:0] != 8'h01) begin
                check_static = `NPU_ST_BAD_VERSION;
            end else if (c[127:112] != 16'd0) begin
                check_static = `NPU_ST_BAD_FIELD;
            end else if ((c[31:24] & 8'hF8) != 8'd0) begin
                check_static = `NPU_ST_BAD_FIELD;
            end else begin
                case (op)
                // ------------------------------------------------------
                `NPU_OP_LOAD: begin
                    if (sub == 8'd0) begin
                        if (c[71:64] != 0 || c[79:72] != 0)
                            check_static = `NPU_ST_BAD_FIELD;
                        else if (fmt != 8'h00)
                            check_static = `NPU_ST_BAD_FIELD;
                        else if (c[103:96] >= EVENT_SLOTS)
                            check_static = `NPU_ST_RANGE;
                        else if (p3 == 0 || p4[15:0] == 0 || p4[31:16] != 0)
                            check_static = `NPU_ST_BAD_FIELD;
                        else if (p2[4:0] != 0 || p6[4:0] != 0 ||
                                 p5[3:0] != 0)
                            check_static = `NPU_ST_ALIGN;
                        else if (p4[15:0] > 16'd1 && (p6 < p3 || p5 < p3))
                            check_static = `NPU_ST_BAD_FIELD;
                        else if ((p7|p8|p9|p10|p11) != 32'd0)
                            check_static = `NPU_ST_BAD_FIELD;
                        else if (c[25] == 1'b1)      // INPLACE not valid for DMA
                            check_static = `NPU_ST_BAD_FIELD;
                    end else if (sub == 8'd1) begin
                        // LOAD_W: weight-tile load
                        s_steps = p3[31:16];
                        g       = {24'd0, p3[11:8]};
                        mode    = {30'd0, p3[1:0]};
                        if (c[71:64] != 0 || c[79:72] != 0 || c[87:80] != 0)
                            check_static = `NPU_ST_BAD_FIELD;
                        else if (fmt != 8'h00)
                            check_static = `NPU_ST_BAD_FIELD;
                        else if (c[103:96] >= EVENT_SLOTS)
                            check_static = `NPU_ST_RANGE;
                        else if (p4 == 0 || p4[3:0] != 0)
                            check_static = `NPU_ST_BAD_FIELD;
                        else if (p0[3:0] != 0 || p1[3:0] != 0)
                            check_static = `NPU_ST_ALIGN;
                        else if (p3[7:2] != 0 || p3[15:12] != 0 ||
                                 g < 1 || g > 8 || s_steps == 0)
                            check_static = `NPU_ST_BAD_FIELD;
                        else if (s_steps > WBUF_K_STEPS)
                            check_static = `NPU_ST_RANGE;
                        else if ((p5|p6|p7|p8|p9|p10|p11) != 32'd0)
                            check_static = `NPU_ST_BAD_FIELD;
                        else if (c[25] == 1'b1)
                            check_static = `NPU_ST_BAD_FIELD;
                        else begin
                            // W_bus = 32*g*S + (W4 ? 16*g*ceil(S/4) : 0)
                            if (mode[0] == 1'b0)
                                w_bus = 32*g*s_steps +
                                        16*g*((s_steps + 3) >> 2);
                            else
                                w_bus = 32*g*s_steps;
                            if (w_bus != p4)
                                check_static = `NPU_ST_BAD_FIELD;
                        end
                    end else begin
                        check_static = `NPU_ST_BAD_FIELD;
                    end
                end

                // ------------------------------------------------------
                `NPU_OP_STORE: begin
                    if (sub != 8'd0)
                        check_static = `NPU_ST_BAD_FIELD;
                    else if (c[79:72] != 0 || c[87:80] != 0)
                        check_static = `NPU_ST_BAD_FIELD;
                    else if (fmt != 8'h00)
                        check_static = `NPU_ST_BAD_FIELD;
                    else if (c[103:96] >= EVENT_SLOTS)
                        check_static = `NPU_ST_RANGE;
                    else if (p3 == 0 || p4[15:0] == 0 || p4[31:16] != 0)
                        check_static = `NPU_ST_BAD_FIELD;
                    else if (p2[4:0] != 0 || p6[4:0] != 0 ||
                             p5[3:0] != 0)
                        check_static = `NPU_ST_ALIGN;
                    else if (p4[15:0] > 16'd1 && (p6 < p3 || p5 < p3))
                        check_static = `NPU_ST_BAD_FIELD;
                    else if ((p7|p8|p9|p10|p11) != 32'd0)
                        check_static = `NPU_ST_BAD_FIELD;
                    else if (c[25] == 1'b1)
                        check_static = `NPU_ST_BAD_FIELD;
                end

                // ------------------------------------------------------
                `NPU_OP_MATMUL: begin
                    m_len  = {24'd0, p0[7:0]};
                    g      = {24'd0, p0[15:8]};
                    mode   = {30'd0, p0[17:16]};
                    tiles  = {16'd0, p1[15:0]};
                    ksteps = {16'd0, p1[31:16]};
                    kvalid = p2;
                    nvalid = p3;
                    kstep  = mode[0] ? 32'd4 : 32'd16;

                    if (sub != 8'd0)
                        check_static = `NPU_ST_BAD_FIELD;
                    else if (c[79:72] != 0)
                        check_static = `NPU_ST_BAD_FIELD;
                    else if (c[71:64] == c[87:80])
                        check_static = `NPU_ST_BAD_FIELD;
                    else if (c[103:96] >= EVENT_SLOTS)
                        check_static = `NPU_ST_RANGE;
                    else if (p0[31:18] != 0)
                        check_static = `NPU_ST_BAD_FIELD;
                    else if (m_len < 1 || m_len > 4 || g < 1 || g > 8)
                        check_static = `NPU_ST_BAD_FIELD;
                    else if (tiles == 0 || tiles > RESULT_SLOTS)
                        check_static = `NPU_ST_RANGE;
                    else if (ksteps == 0 || ksteps > WBUF_K_STEPS)
                        check_static = `NPU_ST_RANGE;
                    else if (kvalid == 0 || nvalid == 0)
                        check_static = `NPU_ST_BAD_FIELD;
                    else if (((kvalid + kstep - 1) / kstep) != ksteps)
                        check_static = `NPU_ST_BAD_FIELD;
                    else if (!((4*g*(tiles-1)) < nvalid &&
                               (nvalid <= 4*g*tiles)))
                        check_static = `NPU_ST_BAD_FIELD;
                    else if (p6 != 0 || (p9|p10|p11) != 0)
                        check_static = `NPU_ST_BAD_FIELD;
                    else if (p5 < (2*kvalid) || p8 < (nvalid *
                             `NPU_ELEM_BYTES(f_dst)))
                        check_static = `NPU_ST_BAD_FIELD;
                    else if (p4[1:0] != 0 || p5[1:0] != 0)
                        check_static = `NPU_ST_ALIGN;
                    else if (`NPU_ELEM_BYTES(f_dst) == 4 &&
                             (p7[1:0] != 0 || p8[1:0] != 0))
                        check_static = `NPU_ST_ALIGN;
                    else if (`NPU_ELEM_BYTES(f_dst) == 2 &&
                             (p7[0] != 0 || p8[0] != 0))
                        check_static = `NPU_ST_ALIGN;
                    else if (f_acc != `NPU_FMT_FP32)
                        check_static = `NPU_ST_BAD_FIELD;
                    else if (f_dst > `NPU_FMT_FP32)
                        check_static = `NPU_ST_BAD_FIELD;
                    else if (c[25] == 1'b1)    // MATMUL never in-place
                        check_static = `NPU_ST_BAD_FIELD;
                    else if (mode[0] == 1'b0) begin
                        // W4 paths: A is 16-bit, B is INT4
                        if (f_src1 != `NPU_FMT_INT4)
                            check_static = `NPU_ST_BAD_FIELD;
                        else if (mode[1] == 1'b0 && f_src0 != `NPU_FMT_FP16)
                            check_static = `NPU_ST_BAD_FIELD;
                        else if (mode[1] == 1'b1 && f_src0 != `NPU_FMT_BF16)
                            check_static = `NPU_ST_BAD_FIELD;
                    end else begin
                        // W16 paths: A and B use the same 16-bit format
                        if (f_src1 != f_src0)
                            check_static = `NPU_ST_BAD_FIELD;
                        else if (f_src0 > `NPU_FMT_BF16)
                            check_static = `NPU_ST_BAD_FIELD;
                    end
                end

                // ------------------------------------------------------
                `NPU_OP_VECTOR: begin
                    vsub  = sub[5:0];
                    bcast = sub[7:6];
                    if (c[103:96] >= EVENT_SLOTS)
                        check_static = `NPU_ST_RANGE;
                    else if (p0 == 0)
                        check_static = `NPU_ST_BAD_FIELD;
                    else if (f_src0 > `NPU_FMT_FP32 || f_dst > `NPU_FMT_FP32)
                        check_static = `NPU_ST_BAD_FIELD;
                    else if (f_acc != `NPU_FMT_FP32)
                        check_static = `NPU_ST_BAD_FIELD;
                    else begin
                        case (vsub)
                        `NPU_VEC_ADD, `NPU_VEC_SUB,
                        `NPU_VEC_MUL, `NPU_VEC_MAX: begin
                            if (bcast == 2'b11)
                                check_static = `NPU_ST_BAD_FIELD;
                            else if (f_src1 > `NPU_FMT_FP32)
                                check_static = `NPU_ST_BAD_FIELD;
                            else if (bcast == 2'b10 && p0[0] != 0)
                                check_static = `NPU_ST_BAD_FIELD;
                            else if (p4 != 0 || p5 != 0 || p6 != 0)
                                check_static = `NPU_ST_BAD_FIELD;
                        end
                        `NPU_VEC_SELECT: begin
                            if (bcast != 2'b10)
                                check_static = `NPU_ST_BAD_FIELD;
                            else if (f_src1 > `NPU_FMT_FP32)
                                check_static = `NPU_ST_BAD_FIELD;
                            else if (p4 != 0 || p5[31:8] != 0)
                                check_static = `NPU_ST_BAD_FIELD;
                            else if (c[87:80] == p5[7:0])
                                check_static = `NPU_ST_BAD_FIELD;
                        end
                        `NPU_VEC_CONVERT: begin
                            if (bcast != 2'b00 || c[79:72] != 0 || p2 != 0)
                                check_static = `NPU_ST_BAD_FIELD;
                            else if (f_src1 != `NPU_FMT_FP16)
                                check_static = `NPU_ST_BAD_FIELD;
                            else if (p4 != 0 || p5 != 0 || p6 != 0)
                                check_static = `NPU_ST_BAD_FIELD;
                        end
                        `NPU_VEC_MUL_SCALAR, `NPU_VEC_ADD_SCALAR: begin
                            if (bcast != 2'b00 || c[79:72] != 0 || p2 != 0)
                                check_static = `NPU_ST_BAD_FIELD;
                            else if (f_src1 != `NPU_FMT_FP16)
                                check_static = `NPU_ST_BAD_FIELD;
                            else if (p5 != 0 || p6 != 0)
                                check_static = `NPU_ST_BAD_FIELD;
                        end
                        `NPU_VEC_COMPARE_GT: begin
                            if (bcast != 2'b00 || f_dst != `NPU_FMT_FP16)
                                check_static = `NPU_ST_BAD_FIELD;
                            else if (f_src1 > `NPU_FMT_FP32)
                                check_static = `NPU_ST_BAD_FIELD;
                            else if (p4 != 0 || p5 != 0 || p6 != 0)
                                check_static = `NPU_ST_BAD_FIELD;
                            else if (c[25] == 1'b1)
                                check_static = `NPU_ST_BAD_FIELD;
                        end
                        `NPU_VEC_ROTATE_PAIR: begin
                            if (bcast != 2'b00 || c[79:72] != 0 || p2 != 0)
                                check_static = `NPU_ST_BAD_FIELD;
                            else if (f_src1 != `NPU_FMT_FP16)
                                check_static = `NPU_ST_BAD_FIELD;
                            else if (p4 != 0 || p5 != 0 || p6 != 0)
                                check_static = `NPU_ST_BAD_FIELD;
                            else if (p0[0] != 0)
                                check_static = `NPU_ST_BAD_FIELD;
                            else if ((V_LANES < 2) || ((V_LANES % 2) != 0))
                                check_static = `NPU_ST_UNSUPPORTED;
                            else if (c[25] == 1'b1)
                                check_static = `NPU_ST_BAD_FIELD;
                        end
                        default: check_static = `NPU_ST_BAD_FIELD;
                        endcase

                        if (check_static == `NPU_ST_OK) begin
                            // in-place legality
                            if (c[25] == 1'b1) begin
                                if (!(vsub == `NPU_VEC_CONVERT ||
                                      vsub == `NPU_VEC_ADD ||
                                      vsub == `NPU_VEC_SUB ||
                                      vsub == `NPU_VEC_MUL ||
                                      vsub == `NPU_VEC_MAX ||
                                      vsub == `NPU_VEC_MUL_SCALAR ||
                                      vsub == `NPU_VEC_ADD_SCALAR))
                                    check_static = `NPU_ST_BAD_FIELD;
                                else if (c[87:80] != c[71:64] || p1 != p3)
                                    check_static = `NPU_ST_BAD_FIELD;
                                else if (vsub == `NPU_VEC_CONVERT &&
                                         f_src0 != f_dst)
                                    check_static = `NPU_ST_BAD_FIELD;
                                else if ((bcast != 2'b00) &&
                                         (c[87:80] == c[79:72]))
                                    check_static = `NPU_ST_BAD_FIELD;
                            end else begin
                                if (c[87:80] == c[71:64])
                                    check_static = `NPU_ST_BAD_FIELD;
                                if ((vsub != `NPU_VEC_MUL_SCALAR) &&
                                    (vsub != `NPU_VEC_ADD_SCALAR) &&
                                    (vsub != `NPU_VEC_CONVERT) &&
                                    (vsub != `NPU_VEC_ROTATE_PAIR) &&
                                    (bcast == 2'b00) &&
                                    (c[87:80] == c[79:72]))
                                    check_static = `NPU_ST_BAD_FIELD;
                            end
                        end
                    end
                end

                // ------------------------------------------------------
                `NPU_OP_REDUCE: begin
                    if (sub[6:1] != 6'd0)
                        check_static = `NPU_ST_BAD_FIELD;
                    else if (c[103:96] >= EVENT_SLOTS)
                        check_static = `NPU_ST_RANGE;
                    else if (p0 == 0)
                        check_static = `NPU_ST_BAD_FIELD;
                    else if (c[79:72] != 0)
                        check_static = `NPU_ST_BAD_FIELD;
                    else if (f_src0 > `NPU_FMT_FP32)
                        check_static = `NPU_ST_BAD_FIELD;
                    else if (f_acc != `NPU_FMT_FP32 || f_dst != `NPU_FMT_FP32)
                        check_static = `NPU_ST_BAD_FIELD;
                    else if (sub[7] == 1'b0 && (p5 != 0 || p6 != 0))
                        check_static = `NPU_ST_BAD_FIELD;
                    else if (sub[7] == 1'b1 && p5[31:8] != 0)
                        check_static = `NPU_ST_BAD_FIELD;
                    else if (c[25] == 1'b1)
                        check_static = `NPU_ST_BAD_FIELD;
                    else if (p2 != 0 || p4 != 0 ||
                             (p7|p8|p9|p10|p11) != 0)
                        check_static = `NPU_ST_BAD_FIELD;
                end

                // ------------------------------------------------------
                `NPU_OP_SFU: begin
                    if (sub > 8'd8)
                        check_static = `NPU_ST_BAD_FIELD;
                    else if (c[103:96] >= EVENT_SLOTS)
                        check_static = `NPU_ST_RANGE;
                    else if (p0 == 0)
                        check_static = `NPU_ST_BAD_FIELD;
                    else if (c[79:72] != 0 || c[25] == 1'b1)
                        check_static = `NPU_ST_BAD_FIELD;
                    else if (f_src0 > `NPU_FMT_FP32 ||
                             f_dst > `NPU_FMT_FP32)
                        check_static = `NPU_ST_BAD_FIELD;
                    else if (f_acc != `NPU_FMT_FP16) // accumulator unused: 0
                        check_static = `NPU_ST_BAD_FIELD;
                    else if (p4[31:2] != 0 || p4[1:0] > 2'd1)  // mode 0/1 only
                        check_static = `NPU_ST_BAD_FIELD;
                    else if ((p5|p6|p7|p8|p9|p10|p11) != 0)
                        check_static = `NPU_ST_BAD_FIELD;
                    else if (sub == 8'd3) begin
                        if (p2 == p3)
                            check_static = `NPU_ST_BAD_FIELD;
                    end else if (p2 != 0) begin
                        check_static = `NPU_ST_BAD_FIELD;
                    end
                end

                // ------------------------------------------------------
                `NPU_OP_LAYOUT: begin
                    if (sub == `NPU_LAYOUT_TRANSPOSE) begin
                        if (c[103:96] >= EVENT_SLOTS)
                            check_static = `NPU_ST_RANGE;
                        else if (p0 == 0 || p1 == 0)
                            check_static = `NPU_ST_BAD_FIELD;
                        else if (c[71:64] == c[87:80] || c[79:72] != 0)
                            check_static = `NPU_ST_BAD_FIELD;
                        else if (f_src0 != f_dst || f_src0 > `NPU_FMT_FP32 ||
                                 f_acc != `NPU_FMT_FP16)
                            check_static = `NPU_ST_BAD_FIELD;
                        else if (c[25] == 1'b1)
                            check_static = `NPU_ST_BAD_FIELD;
                        else if (({32'd0, p4} <
                                  ({32'd0, p1} * `NPU_ELEM_BYTES(f_src0))) ||
                                 ({32'd0, p5} <
                                  ({32'd0, p0} * `NPU_ELEM_BYTES(f_dst))))
                            check_static = `NPU_ST_BAD_FIELD;
                        else if ((`NPU_ELEM_BYTES(f_src0) == 4 &&
                                  (p2[1:0] != 0 || p3[1:0] != 0 ||
                                   p4[1:0] != 0 || p5[1:0] != 0)) ||
                                 (`NPU_ELEM_BYTES(f_src0) == 2 &&
                                  (p2[0] != 0 || p3[0] != 0 ||
                                   p4[0] != 0 || p5[0] != 0)))
                            check_static = `NPU_ST_ALIGN;
                        else if ((p6|p7|p8|p9|p10|p11) != 0)
                            check_static = `NPU_ST_BAD_FIELD;
                        else begin
                            tmp64 = {32'd0, p0} * {32'd0, p1};
                            if (tmp64 > 64'h0000_0000_FFFF_FFFF)
                                check_static = `NPU_ST_RANGE;
                        end
                    end else if (sub == `NPU_LAYOUT_PACK_W16) begin
                        lay_g   = p5[7:0];
                        s_steps = (p0 + 32'd3) >> 2;
                        if (c[103:96] >= EVENT_SLOTS)
                            check_static = `NPU_ST_RANGE;
                        else if (p0 == 0 || p1 == 0)
                            check_static = `NPU_ST_BAD_FIELD;
                        else if (c[71:64] == c[87:80] || c[79:72] != 0)
                            check_static = `NPU_ST_BAD_FIELD;
                        else if (f_src0 != f_dst || f_src0 > `NPU_FMT_BF16 ||
                                 f_acc != `NPU_FMT_FP16)
                            check_static = `NPU_ST_BAD_FIELD;
                        else if (c[25] == 1'b1)
                            check_static = `NPU_ST_BAD_FIELD;
                        else if (lay_g < 1 || lay_g > 8 || p5[31:8] != 0)
                            check_static = `NPU_ST_BAD_FIELD;
                        else if (p1 > 4*lay_g)
                            check_static = `NPU_ST_RANGE;
                        else if (s_steps == 0 || s_steps > WBUF_K_STEPS)
                            check_static = `NPU_ST_RANGE;
                        else if (p4 < 2*p1)
                            check_static = `NPU_ST_BAD_FIELD;
                        else if (p2[0] != 0 || p3[3:0] != 0)
                            check_static = `NPU_ST_ALIGN;
                        else if ((p6|p7|p8|p9|p10|p11) != 0)
                            check_static = `NPU_ST_BAD_FIELD;
                    end else begin
                        check_static = `NPU_ST_BAD_FIELD;
                    end
                end

                // ------------------------------------------------------
                `NPU_OP_FENCE: begin
                    if (sub != 0)
                        check_static = `NPU_ST_BAD_FIELD;
                    else if (c[103:96] >= EVENT_SLOTS)
                        check_static = `NPU_ST_RANGE;
                    else if ((c[71:64]|c[79:72]|c[87:80]|c[95:88]) != 0)
                        check_static = `NPU_ST_BAD_FIELD;
                    else if (c[25] == 1'b1)
                        check_static = `NPU_ST_BAD_FIELD;
                    else if ((p0|p1|p2|p3|p4|p5|p6|p7|p8|p9|p10|p11) != 0)
                        check_static = `NPU_ST_BAD_FIELD;
                end

                // ------------------------------------------------------
                `NPU_OP_RELEASE: begin
                    if (sub != 0 || c[103:96] != `NPU_SLOT_NONE)
                        check_static = `NPU_ST_BAD_FIELD;
                    else if ((c[71:64]|c[79:72]|c[87:80]|c[95:88]) != 0)
                        check_static = `NPU_ST_BAD_FIELD;
                    else if (c[25] == 1'b1)
                        check_static = `NPU_ST_BAD_FIELD;
                    else if (p0[7:0] >= EVENT_SLOTS || p0[31:24] != 0)
                        check_static = `NPU_ST_RANGE;
                    else if ((p1|p2|p3|p4|p5|p6|p7|p8|p9|p10|p11) != 0)
                        check_static = `NPU_ST_BAD_FIELD;
                end

                default: check_static = `NPU_ST_BAD_OPCODE;
                endcase
            end
        end
    endfunction

    // ------------------------------------------------------------------
    // queue storage
    // ------------------------------------------------------------------
    reg [528:0] entry [0:CMD_FIFO_DEPTH-1]; // {identity,error,cmd}
    reg [15:0]  entry_job   [0:CMD_FIFO_DEPTH-1];
    reg [CMD_FIFO_DEPTH-1:0] entry_valid;

    reg [AW-1:0] wp, rp;
    reg [AW:0]   count;

    wire full  = (count == CMD_FIFO_DEPTH[AW:0]);
    wire empty = (count == {(AW+1){1'b0}});

    assign cmd_ready = accept_enable && !full;
    assign issue_valid = !empty;
    assign issue_cmd = entry[rp][511:0];
    assign issue_error = entry[rp][527:512];
    assign issue_identity_error = entry[rp][528];
    assign fifo_level = {{(31-LW){1'b0}}, count[LW-1:0]};

    // ------------------------------------------------------------------
    // enqueue-time identity comparison
    // ------------------------------------------------------------------
    reg        dup;
    integer    i;
    always @* begin
        dup = 1'b0;
        if (held_valid && (held_job == cmd_data[47:32]))
            dup = 1'b1;
        for (i = 0; i < CMD_FIFO_DEPTH; i = i + 1) begin
            if (entry_valid[i] && (entry_job[i] == cmd_data[47:32]))
                dup = 1'b1;
        end
    end

    wire        push = cmd_valid && cmd_ready;
    wire        pop  = issue_valid && issue_ready;
    wire [15:0] init_err = check_static(cmd_data);

    always @(posedge clk_core or negedge rst_core_n) begin
        if (!rst_core_n) begin
            wp             <= {AW{1'b0}};
            rp             <= {AW{1'b0}};
            count          <= {(AW+1){1'b0}};
            entry_valid    <= {CMD_FIFO_DEPTH{1'b0}};
            accepted_count <= 64'd0;
            rejected_count <= 64'd0;
        end else begin
            if (push) begin
                entry[wp]       <= {dup,
                                    dup ? `NPU_ST_BAD_FIELD : init_err,
                                    cmd_data};
                entry_job[wp]   <= cmd_data[47:32];
                entry_valid[wp] <= 1'b1;
                wp              <= wp + 1'b1;

                if (accepted_count != {64{1'b1}})
                    accepted_count <= accepted_count + 64'd1;
                if ((init_err != `NPU_ST_OK) || dup) begin
                    if (rejected_count != {64{1'b1}})
                        rejected_count <= rejected_count + 64'd1;
                end
            end

            if (pop) begin
                entry_valid[rp] <= 1'b0;
                rp              <= rp + 1'b1;
            end

            case ({push, pop})
                2'b10:   count <= count + 1'b1;
                2'b01:   count <= count - 1'b1;
                default: count <= count;
            endcase
        end
    end

endmodule

`default_nettype wire

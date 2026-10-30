//=============================================================================
// activation_packer : scratchpad A tile -> gemv_stream A beats (core domain)
//
//   Follows docs/npu_integration/08_mpu_engine.md section 4.
//
//   A job's M tile has 1..4 rows and the full K reduction.  For every N tile
//   the same A tile is re-read from the scratchpad, so the packer simply walks
//   tile -> step -> row and emits one 1024-bit beat per (tile, step):
//
//     W4  (mode[0]=0): K_step = 16, element i -> a_data[256*m+16*i +:16]
//     W16 (mode[0]=1): K_step =  4, element i -> a_data[256*m+16*i +:16]
//
//   Element j of a row sits at byte A_base + m*A_row_stride + 2*(s*K_step+j).
//   Only the bytes of the valid K prefix are read; the K tail and the rows
//   past m_len stay zero.  A row-step spans at most two B-byte-aligned memory
//   words, and the packer has at most one read in flight (client rule).
//=============================================================================
`timescale 1ns/1ps
`default_nettype none

`include "npu_defs.vh"

module activation_packer #(
    parameter MEM_DATA_W = 256
)(
    input  wire        clk_core,
    input  wire        rst_core_n,

    // job configuration (one pulse per job, from the task adapter)
    input  wire        cfg_valid,
    output wire        cfg_ready,
    input  wire [511:0] cfg_cmd,

    // scratchpad read channel (client MPU_A)
    output wire        mem_req_valid,
    input  wire        mem_req_ready,
    output wire [`NPU_REQ_W(MEM_DATA_W, MEM_DATA_W/8)-1:0] mem_req,
    input  wire        mem_rsp_valid,
    output wire        mem_rsp_ready,
    input  wire [`NPU_RSP_W(MEM_DATA_W)-1:0] mem_rsp,

    // A beat stream
    output wire        a_valid,
    input  wire        a_ready,
    output wire [1023:0] a_data,
    output wire [15:0] a_tile,
    output wire [15:0] a_step,
    output wire        a_last,

    output wire        pack_done,
    output wire [15:0] error
);

    localparam DW    = MEM_DATA_W;
    localparam BW    = MEM_DATA_W / 8;
    localparam REQ_W = `NPU_REQ_W(MEM_DATA_W, MEM_DATA_W/8);
    localparam RSP_W = `NPU_RSP_W(MEM_DATA_W);

    localparam P_IDLE   = 4'd0;
    localparam P_SETUP  = 4'd1;
    localparam P_REQ0   = 4'd2;
    localparam P_WAIT0  = 4'd3;
    localparam P_REQ1   = 4'd4;
    localparam P_WAIT1  = 4'd5;
    localparam P_GATHER = 4'd6;
    localparam P_HOLD   = 4'd7;
    localparam P_DONE   = 4'd8;
    localparam P_ERR    = 4'd9;

    reg [3:0]  state;

    // job configuration
    reg [2:0]  m_len;
    reg [3:0]  n_grp;
    reg [1:0]  mode;
    reg [15:0] tiles;
    reg [15:0] k_steps;
    reg [31:0] k_valid;
    reg [31:0] a_base;
    reg [31:0] a_row_stride;
    reg [7:0]  src_buf;
    reg [15:0] k_step;

    reg [15:0] t_r, s_r, m_r;

    reg [255:0] w0_r, w1_r;
    reg [1023:0] a_buf;
    reg        done_r;
    reg [15:0] error_r;

    // ------------------------------------------------------------------
    // command decode
    // ------------------------------------------------------------------
    wire [2:0]  d_m_len  = cfg_cmd[130:128];
    wire [1:0]  d_mode   = cfg_cmd[145:144];
    wire [15:0] d_tiles  = cfg_cmd[175:160];
    wire [15:0] d_ksteps = cfg_cmd[191:176];
    wire [31:0] d_kvalid = cfg_cmd[223:192];
    wire [31:0] d_abase  = cfg_cmd[287:256];
    wire [31:0] d_astride = cfg_cmd[319:288];
    wire [7:0]  d_srcbuf = cfg_cmd[71:64];

    // ------------------------------------------------------------------
    // per-row-step address computation
    // ------------------------------------------------------------------
    // 64-bit intermediates: the byte address is only truncated to 32 bits
    // when it is handed to the memory subsystem.
    wire [63:0] k_base   = {32'd0, k_valid};
    wire [63:0] s_off    = {48'd0, s_r} * {48'd0, k_step};
    wire [63:0] row_off  = {32'd0, a_row_stride} * {48'd0, m_r};
    wire [63:0] start_a  = {32'd0, a_base} + row_off + (s_off << 1);

    wire [63:0] k_left   = (k_base > s_off) ? (k_base - s_off) : 64'd0;
    wire [15:0] k_eff    = (k_left >= {48'd0, k_step}) ? k_step
                            : (k_left == 64'd0) ? 16'd0 : k_left[15:0];
    wire [5:0]  byte_off = start_a[4:0];
    wire [6:0]  nbytes   = {1'b0, k_eff} << 1;
    wire        need2    = (byte_off + nbytes > 7'd32);
    wire [31:0] w0_addr  = {start_a[31:5], 5'd0};
    wire [31:0] w1_addr  = w0_addr + 32'd32;

    // response field split
    wire [DW-1:0] rsp_data   = mem_rsp[DW-1:0];
    wire [7:0]    rsp_tag    = mem_rsp[DW+7:DW];
    wire          rsp_err    = mem_rsp[DW+8];
    wire [7:0]    rsp_reason = mem_rsp[DW+16:DW+9];

    wire rsp_bad = rsp_err || (rsp_reason != `NPU_MEMR_OK);

    assign mem_req = {8'h00, {BW{1'b0}}, {DW{1'b0}},
                      ((state == P_REQ1) ? w1_addr : w0_addr),
                      src_buf, 1'b0};
    assign mem_req_valid = (state == P_REQ0) || (state == P_REQ1);
    assign mem_rsp_ready = (state == P_WAIT0) || (state == P_WAIT1);

    // ------------------------------------------------------------------
    // gather: copy the k_eff elements out of the two words
    // ------------------------------------------------------------------
    // Byte-wise extraction: an element may straddle the two words, so each
    // byte picks its own word.  Byte indices then never exceed a word.
    integer j;
    reg [6:0] bo;
    reg [7:0] b_lo, b_hi;
    always @(posedge clk_core) begin
        if (state == P_GATHER) begin
            for (j = 0; j < 16; j = j + 1) begin
                bo = byte_off + 2*j;
                b_lo = (bo < 7'd32) ? w0_r[8*bo +: 8] : w1_r[8*(bo - 7'd32) +: 8];
                b_hi = (bo + 7'd1 < 7'd32) ? w0_r[8*(bo + 7'd1) +: 8]
                                           : w1_r[8*(bo + 7'd1 - 7'd32) +: 8];
                if (j < k_eff)
                    a_buf[256*m_r + 16*j +: 16] <= {b_hi, b_lo};
                else
                    a_buf[256*m_r + 16*j +: 16] <= 16'd0;
            end
        end
    end

    assign a_data  = a_buf;
    assign a_valid = (state == P_HOLD);
    assign a_tile  = t_r;
    assign a_step  = s_r;
    assign a_last  = (s_r + 16'd1 == k_steps);

    assign pack_done = done_r;
    assign error     = error_r;
    assign cfg_ready = (state == P_IDLE);

    // ------------------------------------------------------------------
    // FSM
    // ------------------------------------------------------------------
    always @(posedge clk_core or negedge rst_core_n) begin
        if (!rst_core_n) begin
            state    <= P_IDLE;
            m_len    <= 3'd0;
            n_grp    <= 4'd0;
            mode     <= 2'd0;
            tiles    <= 16'd0;
            k_steps  <= 16'd0;
            k_valid  <= 32'd0;
            a_base   <= 32'd0;
            a_row_stride <= 32'd0;
            src_buf  <= 8'd0;
            k_step   <= 16'd0;
            t_r      <= 16'd0;
            s_r      <= 16'd0;
            m_r      <= 16'd0;
            w0_r     <= 256'd0;
            w1_r     <= 256'd0;
            a_buf    <= 1024'd0;
            done_r   <= 1'b0;
            error_r  <= `NPU_ST_OK;
        end else begin
            done_r <= 1'b0;

            case (state)
            P_IDLE: begin
                if (cfg_valid) begin
                    m_len        <= d_m_len;
                    n_grp        <= cfg_cmd[139:136];
                    mode         <= d_mode;
                    tiles        <= d_tiles;
                    k_steps      <= d_ksteps;
                    k_valid      <= d_kvalid;
                    a_base       <= d_abase;
                    a_row_stride <= d_astride;
                    src_buf      <= d_srcbuf;
                    k_step       <= d_mode[0] ? 16'd4 : 16'd16;
                    t_r          <= 16'd0;
                    s_r          <= 16'd0;
                    m_r          <= 16'd0;
                    a_buf        <= 1024'd0;
                    error_r      <= `NPU_ST_OK;
                    state        <= P_SETUP;
                end
            end
            P_SETUP: begin
                if (k_eff == 16'd0)
                    state <= P_GATHER;      // tail step: zero row, no read
                else
                    state <= P_REQ0;
            end
            P_REQ0: begin
                if (mem_req_ready) state <= P_WAIT0;
            end
            P_WAIT0: begin
                if (mem_rsp_valid) begin
                    w0_r <= rsp_data;
                    if (rsp_bad) begin
                        error_r <= `NPU_ST_MEM_ACCESS;
                        state   <= P_ERR;
                    end else if (need2) begin
                        state <= P_REQ1;
                    end else begin
                        state <= P_GATHER;
                    end
                end
            end
            P_REQ1: begin
                if (mem_req_ready) state <= P_WAIT1;
            end
            P_WAIT1: begin
                if (mem_rsp_valid) begin
                    w1_r <= rsp_data;
                    if (rsp_bad) begin
                        error_r <= `NPU_ST_MEM_ACCESS;
                        state   <= P_ERR;
                    end else begin
                        state <= P_GATHER;
                    end
                end
            end
            P_GATHER: begin
                if (m_r + 16'd1 < {13'd0, m_len}) begin
                    m_r   <= m_r + 16'd1;
                    state <= P_SETUP;
                end else begin
                    m_r   <= 16'd0;
                    state <= P_HOLD;
                end
            end
            P_HOLD: begin
                if (a_ready) begin
                    a_buf <= 1024'd0;
                    if (s_r + 16'd1 < k_steps) begin
                        s_r   <= s_r + 16'd1;
                        m_r   <= 16'd0;
                        state <= P_SETUP;
                    end else begin
                        s_r <= 16'd0;
                        m_r <= 16'd0;
                        if (t_r + 16'd1 == tiles) begin
                            state <= P_DONE;
                        end else begin
                            t_r   <= t_r + 16'd1;
                            state <= P_SETUP;
                        end
                    end
                end
            end
            P_DONE: begin
                done_r <= 1'b1;
                state  <= P_IDLE;
            end
            P_ERR: begin
                done_r <= 1'b1;
                state  <= P_IDLE;
            end
            default: state <= P_IDLE;
            endcase
        end
    end

endmodule

`default_nettype wire

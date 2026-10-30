//=============================================================================
// dbg_wrap : npu_top plus internal probes, for the Verilator harness
//
//   A flattened design is opaque to the C++ side, so the probes are
//   hierarchical references collected here into top-level ports.  The
//   wrapper's port list is a superset of npu_top's, which lets the existing
//   harness compile against either module (DUT_CLASS).
//
//   Only build this for debug: the probes are meant to be printed per cycle.
//=============================================================================
`timescale 1ns/1ps
`default_nettype none

`include "npu_defs.vh"

module dbg_wrap #(
    parameter MEM_DATA_W   = 256,
    parameter IO_DATA_W    = 128,
    parameter EXT_WR_W     = 128,
    parameter EVENT_SLOTS  = 16,
    parameter RESULT_SLOTS = 4,
    parameter WBUF_K_STEPS = 256,
    parameter V_LANES      = 4,
    parameter BANKS        = 8,
    parameter BANK_DEPTH   = 1024,
    parameter GROUPS       = 8
)(
    input  wire        clk_core,
    input  wire        clk_io,
    input  wire        reset_n,

    input  wire        platform_isolated,
    input  wire        run_req,

    input  wire        cmd_valid,
    output wire        cmd_ready,
    input  wire [511:0] cmd_data,

    output wire        cpl_valid,
    input  wire        cpl_ready,
    output wire [127:0] cpl_data,

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
    input  wire        wr_rsp_error,

    // ---- probes ----
    output wire [3:0]  p_dma_state,
    output wire [63:0] p_dma_addr,
    output wire [31:0] p_dma_bytes,
    output wire        p_dma_wvalid,
    output wire        p_dma_wready,
    output wire        p_dma_wlast,
    output wire [3:0]  p_wl_state,
    output wire        p_wl_sel,
    output wire        p_wl_dready,
    output wire [1:0]  p_wl_wren,
    output wire [15:0] p_wl_waddr,
    output wire [2:0]  p_wl_wgrp,
    output wire        p_wl_whalf,
    output wire        p_wl_wkind,
    output wire        p_wl_wnz,
    output wire [1:0]  p_wl_publish,
    output wire [1:0]  p_wl_fill,
    output wire [2:0]  p_b0_io,
    output wire [1:0]  p_b0_core,
    output wire        p_b0_dv,
    output wire [2:0]  p_b1_io,
    output wire [1:0]  p_b1_core,
    output wire        p_b1_dv,
    output wire [3:0]  p_rdr_state,
    output wire        p_b_valid,
    output wire        p_b_ready,
    output wire        p_b_nz,
    output wire        p_wlo_nz,
    output wire        p_scl_nz,
    output wire [3:0]  p_mpu_state,
    output wire        p_mpu_avalid,
    output wire        p_mpu_aready,
    output wire        p_mpu_anz,
    output wire        p_mpu_bvalid,
    output wire        p_mpu_bready,
    output wire        p_mpu_accnz,

    output wire [31:0] p_pack_lo,
    output wire        p_b_wgt_nz,
    output wire [31:0] p_rlat_lo,
    output wire        p_scl_lat_nz,
    output wire        p_core_av,
    output wire        p_core_done,
    output wire        p_core_anz,
    output wire        p_core_bwz,
    output wire [31:0] p_core_a_lo,
    output wire [31:0] p_core_b_lo,
    output wire [15:0] p_core_kk,
    output wire [15:0] p_core_ocnt,

    output wire [31:0] p_ha_lo,
    output wire [31:0] p_hb_lo,
    output wire [31:0] p_hb_scl,
    output wire        p_clu_ynz,
    output wire [31:0] p_clu_y0,
    output wire [31:0] p_pe0_lo,
    output wire [31:0] p_clu_acc0
);

    npu_top #(
        .MEM_DATA_W(MEM_DATA_W), .IO_DATA_W(IO_DATA_W),
        .EXT_WR_W(EXT_WR_W), .EVENT_SLOTS(EVENT_SLOTS),
        .RESULT_SLOTS(RESULT_SLOTS), .WBUF_K_STEPS(WBUF_K_STEPS),
        .V_LANES(V_LANES), .BANKS(BANKS), .BANK_DEPTH(BANK_DEPTH),
        .GROUPS(GROUPS)
    ) u_dut (
        .clk_core(clk_core), .clk_io(clk_io), .reset_n(reset_n),
        .platform_isolated(platform_isolated), .run_req(run_req),
        .cmd_valid(cmd_valid), .cmd_ready(cmd_ready), .cmd_data(cmd_data),
        .cpl_valid(cpl_valid), .cpl_ready(cpl_ready), .cpl_data(cpl_data),
        .cfg_valid(cfg_valid), .cfg_ready(cfg_ready), .cfg_write(cfg_write),
        .cfg_addr(cfg_addr), .cfg_wdata(cfg_wdata),
        .cfg_rsp_valid(cfg_rsp_valid), .cfg_rsp_ready(cfg_rsp_ready),
        .cfg_rdata(cfg_rdata), .cfg_error(cfg_error),
        .irq(irq), .top_state(top_state), .fatal(fatal),
        .fatal_code(fatal_code),
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
        .wr_rsp_tag(wr_rsp_tag), .wr_rsp_error(wr_rsp_error)
    );

    // ---- DMA ----
    assign p_dma_state  = u_dut.u_dma.io_state;
    assign p_dma_addr   = u_dut.u_dma.i_addr;
    assign p_dma_bytes  = u_dut.u_dma.i_row_bytes;
    assign p_dma_wvalid = u_dut.dma_wdata_valid;
    assign p_dma_wready = u_dut.dma_wdata_ready;
    assign p_dma_wlast  = u_dut.dma_wdata_last;

    // ---- loader ----
    assign p_wl_state  = u_dut.u_wload.state;
    assign p_wl_sel    = u_dut.u_wload.sel;
    assign p_wl_dready = u_dut.u_wload.data_ready;
    assign p_wl_wren   = u_dut.u_wload.bank_wr_en;
    assign p_wl_waddr  = u_dut.u_wload.wr_addr;
    assign p_wl_wgrp   = u_dut.u_wload.wr_grp;
    assign p_wl_whalf  = u_dut.u_wload.wr_half;
    assign p_wl_wkind  = u_dut.u_wload.wr_kind;
    assign p_wl_wnz    = |u_dut.u_wload.wr_data;
    assign p_wl_publish = u_dut.u_wload.bank_publish;
    assign p_wl_fill    = u_dut.u_wload.bank_fill_start;

    // ---- banks ----
    assign p_b0_io   = u_dut.u_wbank0.io_state;
    assign p_b0_core = u_dut.u_wbank0.core_state;
    assign p_b0_dv   = u_dut.u_wbank0.desc_valid_r;
    assign p_b1_io   = u_dut.u_wbank1.io_state;
    assign p_b1_core = u_dut.u_wbank1.core_state;
    assign p_b1_dv   = u_dut.u_wbank1.desc_valid_r;

    // ---- reader ----
    assign p_rdr_state = u_dut.u_wreader.state;
    assign p_b_valid   = u_dut.rd_b_valid;
    assign p_b_ready   = u_dut.rd_b_ready;
    assign p_b_nz      = |u_dut.rd_b_data;
    assign p_wlo_nz    = |u_dut.wbank_wlo_hi[0 +: GROUPS*256];
    assign p_scl_nz    = |u_dut.wbank_scale[0 +: GROUPS*128];

    // ---- MPU ----
    assign p_mpu_state  = u_dut.u_mpu.state;
    assign p_mpu_avalid = u_dut.u_mpu.a_valid_p;
    assign p_mpu_aready = u_dut.u_mpu.a_ready_p;
    assign p_mpu_anz    = |u_dut.u_mpu.a_data_p;
    assign p_mpu_bvalid = u_dut.u_mpu.w_b_valid;
    assign p_mpu_bready = u_dut.u_mpu.w_b_ready;
    assign p_mpu_accnz  = |u_dut.u_mpu.acc_data;

    // ---- deeper: packer, reader latch, core front end ----
    assign p_pack_lo    = u_dut.u_mpu.a_data_p[31:0];
    assign p_b_wgt_nz   = |u_dut.rd_b_data[2047:0];      // weights only
    assign p_rlat_lo    = u_dut.u_wreader.w_lat[31:0];
    assign p_scl_lat_nz = |u_dut.u_wreader.sc_lat;
    assign p_core_av    = u_dut.u_mpu.core_acc_valid;
    assign p_core_done  = u_dut.u_mpu.core_done;
    assign p_core_anz   = |u_dut.u_mpu.u_core.a_din;
    assign p_core_bwz   = |u_dut.u_mpu.u_core.b_din[2047:0];
    assign p_core_a_lo  = u_dut.u_mpu.u_core.a_din[31:0];
    assign p_core_b_lo  = u_dut.u_mpu.u_core.b_din[31:0];
    assign p_core_kk    = u_dut.u_mpu.u_core.kk;
    assign p_core_ocnt  = u_dut.u_mpu.u_core.ocnt;

    // ---- what gemv_cluster actually consumes / produces ----
    // a_head/b_head are the FIFO outputs; the earlier ca/cb probes watched the
    // FIFO write side, one stage upstream of the array.
    assign p_ha_lo    = u_dut.u_mpu.u_core.a_head[31:0];
    assign p_hb_lo    = u_dut.u_mpu.u_core.b_head[31:0];
    assign p_hb_scl   = u_dut.u_mpu.u_core.b_head[2048 +: 32];
    assign p_clu_ynz  = |u_dut.u_mpu.u_core.u_cluster.y_all;
    assign p_clu_y0   = u_dut.u_mpu.u_core.u_cluster.y_all[31:0];
    assign p_pe0_lo   = u_dut.u_mpu.u_core.u_cluster.g_row[0].g_col[0]
                            .u_unit.g_pe[0].u_pe.out_v[31:0];
    assign p_clu_acc0 = u_dut.u_mpu.u_core.acc_out[31:0];

endmodule

`default_nettype wire

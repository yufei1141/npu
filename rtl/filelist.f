//=============================================================================
// filelist.f : NPU peripheral RTL (docs/npu_integration)
//
//   The two existing cores are referenced from their own trees and are not
//   modified by this peripheral:
//     tensorcore/rtl/*.v   gemv_stream and below (MPU datapath)
//     sfu/sfu/*.v          sfu_top (SFU adapter)
//
//   Include paths: -I rtl/common   (npu_defs.vh)
//                  -I sfu/sfu     (sfu_fp32.vh)
//   Language:     iverilog -g2001  (Verilog-2001, `default_nettype none)
//
//   Order inside rtl/ does not matter for iverilog, but is grouped by role.
//=============================================================================

// common
rtl/common/sync_fifo.v
rtl/common/async_fifo.v
rtl/common/cdc_handshake.v
rtl/common/format_converter.v

// command / scheduling / completion
rtl/cmd/command_frontend.v
rtl/sched/access_decode.v
rtl/sched/task_scheduler.v
rtl/completion/completion_manager.v

// scratchpad
rtl/mem/sram_wrapper.v
rtl/mem/memory_subsystem.v

// DMA
rtl/dma/dma_engine.v

// weights
rtl/weight/weight_bank.v
rtl/weight/weight_loader.v
rtl/weight/weight_reader.v

// MPU adapter
rtl/mpu/activation_packer.v
rtl/mpu/mpu_engine.v

// result writeback
rtl/wb/result_writeback.v

// vector / SFU
rtl/vpu/fp32_alu.v
rtl/vpu/tensor_layout.v
rtl/vpu/vector_engine.v
rtl/sfu/sfu_adapter.v

// top
rtl/top/npu_top.v

//-----------------------------------------------------------------------------
// cores from the existing trees (referenced, never modified)
//-----------------------------------------------------------------------------
// gemv_stream and below
tensorcore/rtl/mul11x4.v
tensorcore/rtl/unpack_fmt.v
tensorcore/rtl/pe.v
tensorcore/rtl/op_fifo.v
tensorcore/rtl/fp32_acc.v
tensorcore/rtl/fp32_reduce4.v
tensorcore/rtl/gemv_unit.v
tensorcore/rtl/gemv_stream.v
tensorcore/rtl/gemv_cluster.v

// sfu_top and below (sfu_fp32.vh is an include, found with -I sfu/sfu)
sfu/sfu/fp32_add.v
sfu/sfu/sfu_slice.v
sfu/sfu/sfu_microseq.v
sfu/sfu/sfu_top.v

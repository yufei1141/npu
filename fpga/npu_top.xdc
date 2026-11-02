# npu_top - ZCU102 (XCZU9EG) timing constraints, first pass.
#
# Both domains are constrained at the same frequency to start; the user asked
# for a 150 MHz attempt.  clk_core and clk_io are independent by construction
# (the top only synchronizes reset release and the io-ready indication through
# the CDCs inside dma_engine / weight_bank), so they are asynchronous groups.
#
# I/O delays are a 2 ns placeholder on both sides, which is roughly what a
# 150 MHz board-level interface would budget.  They are here so the memory and
# host paths are timed rather than reported as unconstrained; the real numbers
# have to come from the board-level design.

create_clock -name clk_core -period 6.667 [get_ports clk_core]
create_clock -name clk_io   -period 6.667 [get_ports clk_io]

set_clock_groups -asynchronous \
    -group [get_clocks clk_core] \
    -group [get_clocks clk_io]

set_false_path -from [get_ports reset_n]
set_false_path -from [get_ports platform_isolated]

# external memory side (clk_io)
set_input_delay  -clock clk_io 2.0 [get_ports {rd_valid rd_data* rd_keep rd_last rd_tag* rd_error}]
set_input_delay  -clock clk_io 2.0 [get_ports {wr_ready wr_rsp_valid wr_rsp_tag* wr_rsp_error}]
set_output_delay -clock clk_io 2.0 [get_ports {rd_req_valid rd_req_addr* rd_req_bytes* rd_req_tag* rd_ready}]
set_output_delay -clock clk_io 2.0 [get_ports {wr_req_valid wr_req_addr* wr_req_bytes* wr_req_tag*}]
set_output_delay -clock clk_io 2.0 [get_ports {wr_valid wr_data* wr_keep wr_last wr_tag* wr_rsp_ready}]

# host side (clk_core)
set_input_delay  -clock clk_core 2.0 [get_ports {cmd_valid cmd_data* cpl_ready run_req}]
set_input_delay  -clock clk_core 2.0 [get_ports {cfg_valid cfg_write cfg_addr* cfg_wdata* cfg_rsp_ready}]
set_output_delay -clock clk_core 2.0 [get_ports {cmd_ready cpl_valid cpl_data* irq top_state* fatal fatal_code*}]
set_output_delay -clock clk_core 2.0 [get_ports {cfg_ready cfg_rsp_valid cfg_rdata* cfg_error}]

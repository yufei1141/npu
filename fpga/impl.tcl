# npu_top implementation (opt / place / route) from the post-synthesis checkpoint.
#
# No pin assignment yet: this is a resource-and-timing probe, not a board
# build, so Vivado places the I/O where it likes.  Timing is what the 150 MHz
# constraint says it is.

set root F:/desktop/npu_1029

open_checkpoint $root/build/synth/post_synth.dcp

opt_design
place_design
phys_opt_design
route_design

write_checkpoint -force $root/build/synth/post_route.dcp
report_utilization -hierarchical -file $root/build/synth/util_impl_hier.rpt
report_timing_summary -delay_type max -max_paths 20 \
    -file $root/build/synth/timing_impl.rpt
report_timing -delay_type max -max_paths 10 -path_type summary \
    -file $root/build/synth/timing_impl_top.rpt
report_clock_utilization -file $root/build/synth/clocks.rpt

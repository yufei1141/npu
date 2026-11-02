# npu_top synthesis for ZCU102 / XCZU9EG-2FFVB1156E
#
# Synthesis only: gives the resource picture and a pre-place timing estimate.
# Implementation (opt/place/route) is a separate script so this stays quick
# enough to iterate on while the RTL is still moving.

set root F:/desktop/npu_1029
set part xczu9eg-ffvb1156-2-e
set top  npu_top

set files [concat \
    [glob -nocomplain $root/rtl/*/*.v] \
    [glob -nocomplain $root/tensorcore/rtl/*.v] \
    [glob -nocomplain $root/sfu/sfu/*.v]]

puts "npu_top: [llength $files] source files"

# npu_defs.vh and sfu_fp32.vh are include files, not compilation units, so
# they are reached through the search path that synth_design takes
read_verilog $files
read_xdc $root/fpga/npu_top.xdc

# rebuilt flattens the whole design and needs a lot of memory during
# technology mapping; none keeps the hierarchy and survives on this machine
#
# At this design size "rebuilt" does not merely run out of memory: the
# Vivado 2021.2 JVM dies with EXCEPTION_ACCESS_VIOLATION part way through
# technology mapping (hs_err_pid*.log in the repo root), which surfaces as
# rc=7 with no error line in the log.  "none" completes the same phase in
# ~1.5 h at 7.1 GB peak, so it is the default here.  FLATTEN=rebuilt still
# works as an override if a flatter netlist is ever needed.
if {[info exists ::env(FLATTEN)]} {
    set flatten $::env(FLATTEN)
} else {
    set flatten none
}
puts "npu_top: flatten_hierarchy = $flatten"

# out_of_context keeps the top-level ports as plain internal nets: no I/O
# buffers are inserted and no package pins are required.  That matters here
# because npu_top has ~1280 port bits (a 512-bit command bus, 128-bit
# completion, 128-bit read/write data), far more than the FFVB1156 package
# has user I/O, so a default-mode place_design could not place the design at
# all.  The question this run answers is "does the logic fit and how fast is
# it", not "can it be pinned out"; the board build will add real AXI pins.
# MODE=default restores the normal flow.
if {[info exists ::env(MODE)]} {
    set mode $::env(MODE)
} else {
    set mode out_of_context
}
puts "npu_top: mode = $mode"

synth_design -top $top -part $part -mode $mode -flatten_hierarchy $flatten \
    -include_dirs [list $root/rtl/common $root/sfu/sfu]

write_checkpoint -force $root/build/synth/post_synth.dcp
report_utilization -hierarchical -file $root/build/synth/util_hier.rpt
report_utilization -file $root/build/synth/util.rpt
report_timing_summary -delay_type max -max_paths 20 \
    -file $root/build/synth/timing_synth.rpt
report_design_analysis -complexity -file $root/build/synth/complexity.rpt

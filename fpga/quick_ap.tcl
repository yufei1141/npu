# Quick out-of-context synthesis of activation_packer alone.
#
# Used to check that the a_buf single-driver restructure really removed the
# "multi-driven net on pin din" critical warnings without paying for a full
# npu_top run.  Same part and include path as the real build.

set root F:/desktop/npu_1029

read_verilog [list $root/rtl/mpu/activation_packer.v]

synth_design -top activation_packer -part xczu9eg-ffvb1156-2-e \
    -mode out_of_context -flatten_hierarchy none \
    -include_dirs [list $root/rtl/common]

puts "QUICK_AP_DONE"

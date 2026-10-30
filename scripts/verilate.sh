#!/usr/bin/env bash
#=============================================================================
# verilate.sh : build a Verilator simulation of the peripheral RTL
#
#   usage:  scripts/verilate.sh <top-module> <harness.cpp> [obj_dir]
#
#   iverilog stays the reference simulator for the protocol testbenches
#   (they rely on Verilog delays).  Verilator is used for the long numeric
#   runs -- real weights, thousands of vectors -- where iverilog is too slow.
#   The harness is plain C++: it drives the DUT ports and prints results.
#
#   The mingw64 perl cannot load Pod::Usage, so verilator is launched through
#   the msys perl that can.
#=============================================================================
set -eu

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"

TOP=${1:?usage: verilate.sh <top-module> <harness.cpp> [obj_dir]}
HARNESS=${2:?usage: verilate.sh <top-module> <harness.cpp> [obj_dir]}
OBJ=${3:-build/vsim/$TOP}

if [ -d /e/app_install_file/msys64/mingw64/bin ]; then
    export PATH="/e/app_install_file/msys64/mingw64/bin:$PATH"
fi

# a perl that has the core modules verilator's driver script needs
PERL=
for p in /e/app_install_file/msys64/usr/bin/perl "$(command -v perl || true)"; do
    [ -n "$p" ] || continue
    if "$p" -MPod::Usage -e1 >/dev/null 2>&1; then PERL=$p; break; fi
done
[ -n "$PERL" ] || { echo "verilate.sh: no perl with Pod::Usage found" >&2; exit 1; }

mkdir -p "$OBJ"

# Verilator writes the harness path into the generated makefile unchanged, and
# make runs with -C "$OBJ"; a relative source path breaks there.
HARNESS=$(cd "$(dirname "$HARNESS")" && pwd)/$(basename "$HARNESS")

# Verilator drives the C++ compile through make without a -j flag; exporting
# MAKEFLAGS lets the sub-make use every core (serial builds take 25+ minutes).
JOBS=${JOBS:-$(nproc 2>/dev/null || echo 4)}
export MAKEFLAGS="-j$JOBS"

# One tensorcore file is substituted.  fp32_acc.v accumulates with a delayed
# assignment to an array inside a for loop, which the simulator cannot
# schedule: with BLKLOOPINIT suppressed it builds, but the assignment is
# dropped and the accumulator never leaves its reset value.  tb/vsim/
# fp32_acc_sim.v is an elaboration-identical replacement (same fadd, one
# always block per lane).  tensorcore/ itself is not modified, and iverilog
# -- the reference simulator -- keeps compiling the original.
SIM_SHIM=tb/vsim/fp32_acc_sim.v
SHIMMED=tensorcore/rtl/fp32_acc.v
SRC=
for f in rtl/*/*.v tensorcore/rtl/*.v sfu/sfu/*.v; do
    [ "$f" = "$SHIMMED" ] || SRC="$SRC $f"
done
SRC="$SRC $SIM_SHIM"

# -Wno-fatal: the slices carry Verilog-2001 idioms that Verilator flags as
# width/timescale warnings but elaborates correctly.  64-bit is needed by the
# address arithmetic in the memory path.
"$PERL" "$(command -v verilator)" \
    --cc --exe --build -O3 \
    -Wno-fatal -Wno-WIDTH -Wno-TIMESCALEMOD -Wno-UNOPTFLAT -Wno-CASEINCOMPLETE \
    -Wno-LATCH -Wno-MULTIDRIVEN -Wno-BLKLOOPINIT \
    -Irtl/common -Isfu/sfu -Itensorcore/rtl \
    --top-module "$TOP" \
    --Mdir "$OBJ" \
    -CFLAGS "-O2 -std=c++11" \
    -LDFLAGS "-static" \
    $SRC \
    "$HARNESS"

echo "built: $OBJ/V$TOP"

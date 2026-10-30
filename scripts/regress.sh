#!/usr/bin/env bash
#=============================================================================
# regress.sh : compile and run every peripheral testbench with iverilog/vvp
#
#   usage:  scripts/regress.sh            run all testbenches
#           scripts/regress.sh tb_mpu     run the ones whose name matches
#
#   The MSYS2 mingw64 toolchain is prepended to PATH so iverilog is found on
#   Windows; on Linux/macOS an installed iverilog is used as-is.
#=============================================================================
set -u

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"

if [ -d /e/app_install_file/msys64/mingw64/bin ]; then
    export PATH="/e/app_install_file/msys64/mingw64/bin:$PATH"
fi

OUT=build/reg
mkdir -p "$OUT"

# peripheral RTL + the two unmodified cores it instantiates
RTL_SRC=$(ls rtl/*/*.v)
CORE_SRC=$(ls tensorcore/rtl/*.v)
SFU_SRC=$(ls sfu/sfu/*.v)
TB_SUPPORT="tb/engine_stub.v tb/extmem_model.v"

FILTER=${1:-tb_}
fail=0

for tb in tb/*.v; do
    name=$(basename "$tb" .v)
    case "$name" in
        engine_stub|extmem_model|dbg_wrap|fifo_wrap) continue ;;
    esac
    case "$name" in
        *"$FILTER"*) ;;
        *) continue ;;
    esac

    # -I rtl/common : npu_defs.vh
    # -I sfu/sfu    : sfu_fp32.vh (included by the SFU slice)
    if ! iverilog -g2001 -I rtl/common -I sfu/sfu -o "$OUT/$name.vvp" \
            $RTL_SRC $CORE_SRC $SFU_SRC $TB_SUPPORT "$tb" 2> "$OUT/$name.err"; then
        echo "$name: COMPILE FAIL"
        head -5 "$OUT/$name.err"
        fail=1
        continue
    fi

    # a hung testbench is killed after 180 s; vvp writes the log first
    timeout -s KILL 180 vvp "$OUT/$name.vvp" > "$OUT/$name.log" 2>&1
    rc=$?
    line=$(grep -E "^(PASS|FAIL|RESULT:)" "$OUT/$name.log" | tail -1)
    # some end-to-end benches report "RESULT: PASS/FAIL ..." instead
    case "$line" in
        RESULT:\ PASS*) line="PASS: (testbench verdict)" ;;
        RESULT:\ FAIL*) line="FAIL: (testbench verdict)" ;;
    esac
    if [ -z "$line" ]; then
        line="FAIL: (no verdict, rc=$rc)"
    fi
    echo "$name: $line"
    case "$line" in
        PASS:*) ;;
        *) fail=1 ;;
    esac
done

exit $fail

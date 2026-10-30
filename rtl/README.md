# NPU peripheral RTL

Platform-independent integration of the MiniCPM-1B text NPU described in
`docs/npu_integration/`. The peripheral owns the command interface, scratchpad,
scheduling, DMA, weight staging, completion/fault reporting — and reaches into
the two pre-existing datapath cores through adapters:

| core | tree | used by |
|---|---|---|
| `gemv_stream` (+ `gemv_unit`, `pe`, `fp32_acc`, `fp32_reduce4`) | `tensorcore/rtl/` | `rtl/mpu/mpu_engine.v` |
| `sfu_top` (+ `sfu_slice`, `sfu_fp32.vh`) | `sfu/sfu/` | `rtl/sfu/sfu_adapter.v` |

**`tensorcore/` is never modified.** Every interface it needs is produced by
`rtl/mpu/mpu_engine.v`, which does not restart the core, reach into its
sequencer, or drive its internal state.

`sfu/sfu/` was a placeholder skeleton; its FP32 datapath is now implemented
from `sfu/arch/spec.md` and `sfu/arch/microarch.md` (see `sfu/sfu/sfu_slice.v`), while
the module ports and `sfu_top`'s two-slice structure are unchanged. The
adapter in `rtl/sfu/` still treats it as an external core through that port.

## Building

Verilog-2001 only (`-g2001`); the include files are `rtl/common/npu_defs.vh`
and `sfu/sfu/sfu_fp32.vh`, so both directories go on the include path.

```sh
iverilog -g2001 -I rtl/common -I sfu/sfu -c rtl/filelist.f -o build/top.vvp \
         tb/engine_stub.v tb/extmem_model.v tb/tb_npu_top.v
vvp build/top.vvp
```

Full regression (every testbench, compile + run, PASS/FAIL line each):

```sh
scripts/regress.sh              # all
scripts/regress.sh tb_sfu       # one
```

`scripts/regress.sh` prepends the MSYS2 mingw64 toolchain when it exists, so
the same script works from Git Bash on Windows and from a POSIX shell.

## Directory map

```
rtl/
  common/   npu_defs.vh, sync_fifo, async_fifo, cdc_handshake, format_converter
  cmd/      command_frontend        512-bit command FIFO + static decode
  sched/    access_decode           buffer-resource half of the CHECK state
            task_scheduler          in-order issue, deps, locks, slot allocation
  completion/ completion_manager    event slots, dependency lookup, CPL FIFO
  mem/      sram_wrapper            fixed-latency synchronous bank model
            memory_subsystem        5 clients -> 8 banks, arbitration, checks
  dma/      dma_engine              scratchpad <-> external memory, cross-clock
  weight/   weight_loader           external memory -> weight banks (io domain)
            weight_bank             double-buffered tile storage, CDC
            weight_reader           tile -> gemv_stream B beats
  mpu/      activation_packer       scratchpad A tile -> gemv_stream A beats
            mpu_engine              MATMUL task adapter around gemv_stream
  vpu/      fp32_alu                combinational FP32 add/mul/max/gt
            tensor_layout           LAYOUT: TRANSPOSE2D, PACK_W16
            vector_engine           VECTOR / REDUCE / LAYOUT tasks
  sfu/      sfu_adapter             SFU task adapter around sfu_top
  wb/       result_writeback        accumulator -> scratchpad, one job at a time
  top/      npu_top                 pin-level peripheral, config, top FSM
```

## Hierarchy

```
npu_top
├── command_frontend                (core)
├── task_scheduler                  (core) ── access_decode (comb.) (core)
├── completion_manager              (core)
├── memory_subsystem                (core) ── sram_wrapper × BANKS
├── dma_engine                      (core ⇄ io, own CDC)
├── weight_loader                   (io)
├── weight_bank × 2                 (io write / core read, own CDC)
├── weight_reader                   (core)
├── mpu_engine                      (core)
│   ├── activation_packer
│   └── gemv_stream  …              (unmodified core)
├── result_writeback                (core)
├── vector_engine                   (core) ── fp32_alu × lanes
│                                            └─ tensor_layout
├── sfu_adapter                     (core)
└── sfu_top  …                      (unmodified core)
```

## Clock and reset

Two independent clocks, with no integer frequency ratio required
(`docs/npu_integration/14_npu_top.md`). Every crossing uses a four-phase
handshake or a Gray-pointer FIFO, neither of which depends on a ratio.

* **core** — everything except the io-side halves listed above.
* **io** — `dma_engine` command/row state machine, `weight_loader`,
  `weight_bank` write port.

`npu_top` synchronizes only reset release (3-stage per domain) and the io-ready
indication (2 flip-flops). All data crossing a clock boundary travels through
`cdc_handshake` or the Gray-pointer FIFOs; there are no unsynchronized data
paths between domains.

## Interface conventions

**Handshakes** are valid/ready on both directions, payload stable while valid.
A pulse-style interface (e.g. `weight_reader` select) is not used where
backpressure is possible.

**Command word** (512 bit, `docs/npu_integration/02_command_format.md`):

| bits | field | bits | field |
|---|---|---|---|
| 7:0 | version | 95:88 | formats `{acc[7:6], dst[5:4], src1[3:2], src0[1:0]}` |
| 15:8 | opcode | 103:96 | event slot |
| 23:16 | subop | 111:104 | dependency slot |
| 31:24 | flags (0 DEP_EN, 1 INPLACE, 2 IRQ) | 127:112 | reserved |
| 47:32 | job_id | 159:128 | P0 |
| 71:64 | src0 buffer | 191:160 | P1 |
| 79:72 | src1 buffer | 223:192 | P2 |
| 87:80 | dst buffer | 255:224 | P3 … P11 at 128+32·i |

Opcodes: LOAD `0x01`, STORE `0x02`, MATMUL `0x10`, VECTOR `0x20`, REDUCE
`0x21`, SFU `0x22`, LAYOUT `0x23`, FENCE `0x30`, RELEASE `0x31`.
Formats: FP16 `0`, BF16 `1`, FP32 `2`, INT4 `3`.

**Completion word** (128 bit, `docs/npu_integration/13_completion_and_fault.md`):
`{aux[32], units[32], detail[16], status[16], opcode[8], slot[8], job[16]}`.
A task is finished when its completion record has been handed to the host;
`RELEASE` retires the dependency slot.

**Memory requests** use `mem_req_t` / `mem_rsp_t` from `npu_defs.vh`
(`{tag, wstrb, wdata, byte_offset, buf_id, write}` / `{reason, error, tag,
rdata}`). Offsets are byte offsets **inside a logical buffer**, word-aligned for
reads. A client has at most one outstanding request; every accepted request
receives exactly one response, including error responses.

## Scratchpad model

8 physical banks × 1024 × 256 bit (8 KiB each, 8 B-interleaved words). Logical
buffers are mapped to a bank, word base and byte size by the configuration
port. Five fixed clients: `0 MPU_A`, `1 VPU`, `2 SFU`, `3 RESULT_WB`,
`4 DMA_CORE`. Each bank arbitrates round-robin, one access in flight.

Access rules enforced in hardware (`memory_subsystem`, plus the static check in
`access_decode`):

* unknown / disabled buffer, unaligned offset, or an access crossing the buffer
  end → error response, the RAM is not touched;
* within one command, a read range and a write range in the **same physical
  bank** must not overlap, and a read and a write of the **same logical
  buffer** is rejected outright as `BAD_FIELD`;
* `INPLACE` drops the destination buffer from the read set, so the destination
  may also hold a source; the other source must still be a different buffer.

## Configuration map (`npu_top`)

| address | register |
|---|---|
| `0x0000` | VERSION (RO, `0x0001_0000`) |
| `0x0008` | STATUS (RO) |
| `0x0010` | CAPABILITY (RO) |
| `0x0018` | WIDTHS (RO) |
| `0x0020` | LIMITS (RO) |
| `0x0028` | FATAL_INFO (RO) |
| `0x0030` | COUNTER_SNAPSHOT (WO) |
| `0x0038` | SNAPSHOT_STATUS (RO) |
| `0x0100 + 0x20·b` | buffer `b`: `+0x00` map `{bank[15:8], enable[0]}`, `+0x08` base word, `+0x10` size bytes, `+0x18` diagnostic |
| `0x1000 + 8·i` | counter `i` (RO) |

Configuration writes are accepted only in IDLE. A write with reserved bits set,
or to a read-only word, is refused with `cfg_error` and has no effect.

## Top-level states

`0 RESET → 1 BOOT → 2 IDLE → 3 RUN → 4 DRAIN → 2 IDLE`, plus `5 FAULT`.
`run_req` gates acceptance of new commands (`RUN` only); a rising edge is not
required. Lowering `run_req` in RUN enters DRAIN, which returns to IDLE when the
command FIFO is empty, no task is issuing or executing, no memory request is
outstanding and no writeback is pending. `fatal` latches in any state and stops
command acceptance; the platform recovers by reset.

## Verification

`scripts/regress.sh` runs 17 testbenches, all passing:

| testbench | covers |
|---|---|
| `tb_command_frontend` | FIFO, static decode, identity errors, drain |
| `tb_control_plane` | config map, start/stop, fault latch |
| `tb_completion_manager` | slots, dependencies, RELEASE, CPL backpressure |
| `tb_memory_subsystem` | arbitration, routing, error responses |
| `tb_dma_engine` | LOAD/STORE both directions, cross-clock |
| `tb_weight_path` | loader → bank → reader, tile double buffering |
| `tb_mpu_engine` | MATMUL end-to-end against `gemv_stream` |
| `tb_result_writeback` | accumulator drain, FP32 store, summary |
| `tb_format_converter` | FP16/BF16/FP32 RNE, FTZ, flags |
| `tb_fp32_alu` | add/sub/mul/max/gt numeric contract |
| `tb_vector_engine` | VECTOR / REDUCE over the real scratchpad model |
| `tb_tensor_layout` | TRANSPOSE2D / PACK_W16 addresses, padding, strobes, rejections |
| `tb_sfu_adapter` | capability gating, lane padding, batching, SINCOS two-frame, numeric faults |
| `tb_sfu_slice` | FP32 datapath against a double precision reference: 9 ops, ULP budget, specials, FTZ, BF16/FP16 in and out, fast/accurate mode |
| `tb_npu_top` | reset → BOOT → IDLE, config readback/rejection, RUN, VECTOR, LAYOUT and SFU through the full pipeline, EVENT_RELEASE + slot reuse, DRAIN |
| `tb_gemv_cluster` | `gemv_cluster` alone: all four modes, K=16/K=4 beats, m_len/n_grp masking |
| `tb_top_gemv` | LOAD → LOAD_W → MATMUL → STORE end to end, K=1024 of 1.0 → 1024.0 |

## Verilator and real weights

`iverilog` stays the reference simulator for the protocol testbenches, which
rely on Verilog delays. The numeric runs — real weights, thousands of vectors —
go through `scripts/verilate.sh`, which builds a C++ harness against the same
hierarchy:

```sh
scripts/verilate.sh npu_top tb/vsim/npu_top_main.cpp   # -> build/vsim/npu_top/Vnpu_top
build/vsim/npu_top/Vnpu_top --mem image.bin --script cmds.txt
```

One tensorcore file is substituted in that build: `fp32_acc.v` accumulates with
a delayed assignment to an array inside a `for` loop, which Verilator cannot
schedule (BLKLOOPINIT). Suppressing the lint message still builds, but the
assignment is dropped and the accumulator never leaves its reset value.
`tb/vsim/fp32_acc_sim.v` is an elaboration-identical replacement (`fadd()` is
copied verbatim; only the per-lane update moves to one `always` block per lane).
`tensorcore/` is not modified and `iverilog` keeps compiling the original. The
mapping lives in `scripts/verilate.sh` (`SHIMMED` → `SIM_SHIM`).

Real-weight validation drives one projection of one layer of the actual
Qwen3-0.6B AWQ checkpoint, with the reference being the real fp32 forward pass
(`scripts/qwen3_ref.py`), not a hardware model:

```sh
python scripts/real_gemv.py --proj q --layer 0 --cols 32 --token -1
python scripts/real_gemv.py --proj down --layer 0 --cols 32 --token -1
```

All seven projections return bit-identical results to an exact fp32 matmul over
the packed weights, so the remaining error is the int4 + E8M0 requantization
(5–31 % of RMS depending on the projection), not the RTL. Results and the two
bugs this found are recorded in
`docs/npu_integration/15_validation_and_sequences.md`.

## SFU accuracy

`tb_sfu_slice` drives every op through the slice port and compares against a
double precision reference; worst case over 6 000 to 15 000 random vectors per
op, with the ULP taken from the result's own exponent:

| op | worst | target | op | worst | target |
|---|---|---|---|---|---|
| EXP2 | 0.95 | 2 | SIN | 1.37 | 4 |
| LOG2 | 0.50 | 2 | COS | 1.17 | 4 |
| RCP | 1.36 | 4 | TANH | 3.05 | 4 |
| RSQRT | 1.52 | 4 | SQRT | 2.01 | 4 |

The datapath has no FMA and rounds once per `sf_add` / `sf_mul`, so exact
cases are not guaranteed to the last bit: `1/1.0` and `1/2.0` land one ULP low
(a Newton fixed point in fp32), `cos(pi/2)` is 2 ULP from the exact -4.37e-8,
and `tanh(pi)` is 1 ULP from the correctly rounded value. The documented
deviations from `sfu/arch/microarch.md` are FTZ (subnormal inputs and results
flush to zero) and SIN/COS arguments above 2^20 (invalid).

## Not built yet

* Nothing outstanding on the SFU: `P4[1:0]` now drives the core `mode` (0 fast,
  1 accurate; other values are `BAD_FIELD`), and all nine subops
  (`0` EXP2, `1` RCP, `2` RSQRT, `3` SINCOS, `4` LOG2, `5` SQRT, `6` SIN,
  `7` COS, `8` TANH) map onto the verified datapath.

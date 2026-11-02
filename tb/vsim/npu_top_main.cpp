//=============================================================================
// npu_top_main.cpp : Verilator platform harness for npu_top
//
//   The DUT is driven at its real pins: a 512-bit command stream in, 128-bit
//   completion records out, a config port, and the two external memory
//   channels.  External memory lives here in C++ and is preloaded from a
//   binary image, so a test can move real weight and activation bytes.
//
//   usage: Vnpu_top --mem image.bin --script cmds.txt
//
//   script lines:
//     CMD <128 hex digits>        one 512-bit command word
//     WAIT <job>                  run until that job completes
//     DUMP <addr> <len> <path>    write external memory to a file
//     DELAY <cycles>
//
//   every completion is printed as
//     CPL <job> <slot> <opcode> <status> <detail> <units> <aux>
//
//   The buffer map is set up the same way tb_npu_top does it: buffers 0..7
//   on banks 0..7, base 0, 8 KiB each.
//=============================================================================
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <string>
#include <vector>
#include <deque>
#include <set>
#include <fstream>
#include <iostream>
#include <sstream>

// DUT_CLS selects the generated model name, so the same harness drives either
// npu_top or the probe wrapper in tb/dbg_wrap.v.
#if defined(DUT_IS_DBG)
#include "Vdbg_wrap.h"
#ifndef DUT_CLS
#define DUT_CLS Vdbg_wrap
#endif
#else
#include "Vnpu_top.h"
#ifndef DUT_CLS
#define DUT_CLS Vnpu_top
#endif
#endif
#include "verilated_vcd_c.h"
#include "verilated.h"
#ifdef MPU_PEEK
// the peek below walks generated internals, so it needs the root class body
#include DUT_ROOT_HDR
#endif

static DUT_CLS *top;
static vluint64_t cyc = 0;
static std::vector<uint8_t> mem;

// ---------------------------------------------------------------------------
// handshake snapshot for one cycle.  Values are taken in the low phase, which
// is exactly what the following rising edge samples.
// ---------------------------------------------------------------------------
struct Hs {
    bool cmd_ready, cfg_ready, cfg_rsp_valid;
    bool cpl_valid;
    bool rd_req_valid, rd_ready, rd_valid;
    bool wr_req_valid, wr_valid, wr_rsp_ready;
    // the write beat must be taken here, in the low phase: the FIFO that
    // drives wr_data/wr_keep/wr_last pops on the same edge the handshake
    // completes, so reading them after the edge gives the *next* beat.
    uint32_t wr_data[4];
    bool     wr_keep[16];
    bool     wr_last;
};

static Hs hs;
static uint32_t cpl_words[4];
static std::deque<std::string> cpl_log;
// WAIT must not lose a completion that arrived while an earlier script line
// was running, so every job that has been reported stays known.
static std::set<unsigned> cpl_done;

// DPROBE=1 prints the dbg_wrap probe ports once per cycle.  The probe names
// only exist in that model, so the whole block compiles out otherwise.
#ifdef DUT_PROBES
static bool dprobe_on = false;
// DPROBE_SUM=1 narrows the printout to MPU state changes plus a heartbeat,
// so a full-block run can be watched without a per-cycle flood.
static bool dprobe_sum = false;
static unsigned long long dprobe_from = 0, dprobe_to = ~0ULL;
static unsigned long long dprobe_beat = 0;
static unsigned dprobe_mpu = ~0U;
static void print_probes(void)
{
    if (cyc < dprobe_from || cyc > dprobe_to) return;
    if (dprobe_sum) {
        unsigned m = (unsigned)top->p_mpu_state;
        bool changed = (m != dprobe_mpu);
        if (!changed && (cyc - dprobe_beat) < 20000) return;
        dprobe_mpu  = m;
        dprobe_beat = cyc;
    }
    printf("P %llu dma=%u addr=%016llx bytes=%u wv=%d wr=%d | "
           "wl=%u sel=%d dr=%d wen=%u wa=%04x wg=%u wh=%d wk=%d wnz=%d "
           "fill=%u pub=%u | b0=%u/%u/%d b1=%u/%u/%d | "
           "rdr=%u bv=%d br=%d bnz=%d wlo=%d scl=%d | "
           "mpu=%u av=%d ar=%d anz=%d bv=%d br=%d acc=%d\n",
           (unsigned long long)cyc,
           (unsigned)top->p_dma_state, (unsigned long long)top->p_dma_addr,
           (unsigned)top->p_dma_bytes, (int)top->p_dma_wvalid,
           (int)top->p_dma_wready,
           (unsigned)top->p_wl_state, (int)top->p_wl_sel,
           (int)top->p_wl_dready, (unsigned)top->p_wl_wren,
           (unsigned)top->p_wl_waddr, (unsigned)top->p_wl_wgrp,
           (int)top->p_wl_whalf, (int)top->p_wl_wkind, (int)top->p_wl_wnz,
           (unsigned)top->p_wl_fill, (unsigned)top->p_wl_publish,
           (unsigned)top->p_b0_io, (unsigned)top->p_b0_core,
           (int)top->p_b0_dv,
           (unsigned)top->p_b1_io, (unsigned)top->p_b1_core,
           (int)top->p_b1_dv,
           (unsigned)top->p_rdr_state, (int)top->p_b_valid,
           (int)top->p_b_ready, (int)top->p_b_nz,
           (int)top->p_wlo_nz, (int)top->p_scl_nz,
           (unsigned)top->p_mpu_state, (int)top->p_mpu_avalid,
           (int)top->p_mpu_aready, (int)top->p_mpu_anz,
           (int)top->p_mpu_bvalid, (int)top->p_mpu_bready,
           (int)top->p_mpu_accnz);
    printf("Q %llu pack=%08x bwgt=%d rlat=%08x scl=%d | "
           "cav=%d cdn=%d canz=%d cbwz=%d ca=%08x cb=%08x kk=%u ocnt=%u\n",
           (unsigned long long)cyc,
           (unsigned)top->p_pack_lo, (int)top->p_b_wgt_nz,
           (unsigned)top->p_rlat_lo, (int)top->p_scl_lat_nz,
           (int)top->p_core_av, (int)top->p_core_done,
           (int)top->p_core_anz, (int)top->p_core_bwz,
           (unsigned)top->p_core_a_lo, (unsigned)top->p_core_b_lo,
           (unsigned)top->p_core_kk, (unsigned)top->p_core_ocnt);
    printf("R %llu ha=%08x hb=%08x scl=%08x | ynz=%d y0=%08x pe0=%08x acc0=%08x\n",
           (unsigned long long)cyc,
           (unsigned)top->p_ha_lo, (unsigned)top->p_hb_lo,
           (unsigned)top->p_hb_scl,
           (int)top->p_clu_ynz, (unsigned)top->p_clu_y0,
           (unsigned)top->p_pe0_lo, (unsigned)top->p_clu_acc0);
    printf("S %llu aacc=%u bacc=%u trel=%u rcap=%u pdseen=%d bfeed=%d "
           "tile=%u btcnt=%u twait=%d | pack=%u ptile=%u pstep=%u\n",
           (unsigned long long)cyc,
           (unsigned)top->p_mpu_aacc, (unsigned)top->p_mpu_bacc,
           (unsigned)top->p_mpu_trel, (unsigned)top->p_mpu_rcap,
           (int)top->p_mpu_pdseen, (int)top->p_mpu_bfeed,
           (unsigned)top->p_mpu_tile, (unsigned)top->p_mpu_btcnt,
           (int)top->p_mpu_twait,
           (unsigned)top->p_pack_state, (unsigned)top->p_pack_ttile,
           (unsigned)top->p_pack_sstep);
    fflush(stdout);
}
#endif

// TRACE=1 prints every change of the bus request/valid pins with its cycle,
// so a stall can be attributed to the DUT or to this model of the far side.
static bool trace_on = false;
static int  tr_rd_req = -1, tr_rd_val = -1, tr_wr_req = -1, tr_wr_val = -1,
            tr_wr_rsp = -1, tr_state = -1;

// VCD tracing, enabled with VCD=<path>; VCD_START / VCD_END bound the window
// in harness cycles so a single test does not produce a multi-GB file.  Only
// the --trace build has the generated trace() method, so the feature compiles
// out of the fast (untraced) simulator.
#if VM_TRACE
static VerilatedVcdC *vcd = nullptr;
static uint64_t vcd_start = 0, vcd_end = ~0ULL;
static void close_vcd(void) { if (vcd) { vcd->close(); vcd = nullptr; } }
#endif

// read side registers
static int      r_state = 0;              // 0 idle, 1 data
static uint64_t r_addr = 0;
static uint32_t r_rem = 0;
static uint16_t r_tag = 0;
static uint8_t  r_beat[16];

// write side registers
static int      w_state = 0;              // 0 idle, 1 data, 2 response
static uint64_t w_addr = 0;
static uint32_t w_rem = 0;
static uint16_t w_tag = 0;
static int      w_delay = 0;

// ---------------------------------------------------------------------------
// DDR model.  The far side sustains `ddr_bpc` bytes per core cycle out of a
// token bucket, and returns the first beat of a request `ddr_lat` cycles
// after accepting it.  Bandwidth must be given per *cycle*, so a GB/s figure
// only means something together with the core clock: 10 GB/s at 250 MHz is
// 40 B/cycle, which a single 128 bit port (16 B per beat, one beat per
// cycle) can never exceed -- the limit only binds when it is set below that.
// Reads and writes share the bucket, as they would share the DRAM.
// ---------------------------------------------------------------------------
static double ddr_bpc = 0.0;              // 0 = unlimited
static double ddr_credit = 0.0;
static double ddr_burst = 256.0;          // burst depth, bytes
static long   ddr_lat = 0;                // cycles, request -> first beat
static long   r_lat_left = 0, w_lat_left = 0;
static double fcore_mhz = 0.0;            // only used to report GB/s
static unsigned long long rd_bytes = 0, wr_bytes = 0;
// Guard against a runaway run, not a budget: one position of the full
// 28-layer model is ~48M cycles, so the old 40M cap cut a real run off
// mid-position.  Raise it with --max-cycles / MAX_CYCLES.
static unsigned long long cyc_max = 400000000ULL;

static void set_beat(uint32_t *dst, const uint8_t *src, int n)
{
    for (int i = 0; i < 4; i++) {
        uint32_t v = 0;
        for (int b = 0; b < 4; b++) {
            int k = 4 * i + b;
            if (k < n) v |= (uint32_t)src[k] << (8 * b);
        }
        dst[i] = v;
    }
}

static void read_beat_data(void)
{
    uint32_t n = (r_rem >= 16) ? 16 : r_rem;
    memset(r_beat, 0, sizeof(r_beat));
    for (uint32_t i = 0; i < n; i++) {
        uint64_t a = r_addr + i;
        r_beat[i] = (a < mem.size()) ? mem[a] : 0;
    }
}

// is the far side able to hand over nbytes right now?
static bool ddr_ok(double nbytes)
{
    return ddr_bpc <= 0.0 || ddr_credit >= nbytes;
}

static void drive_read(void)
{
    uint32_t nb = (r_rem >= 16) ? 16 : r_rem;
    top->rd_req_ready = (r_state == 0);
    top->rd_valid     = (r_state == 1) && (r_lat_left == 0) && ddr_ok(nb);
    top->rd_last      = (r_state == 1) && (r_rem <= 16);
    top->rd_tag       = r_tag;
    top->rd_keep      = (r_rem >= 16) ? 0xFFFF : (uint16_t)((1u << r_rem) - 1u);
    top->rd_error     = 0;
    uint32_t words[4];
    set_beat(words, r_beat, 16);
    for (int i = 0; i < 4; i++) top->rd_data[i] = words[i];
}

static void drive_write(void)
{
    top->wr_req_ready = (w_state == 0);
    top->wr_ready     = (w_state == 1) && (w_lat_left == 0) && ddr_ok(16.0);
    top->wr_rsp_valid = (w_state == 2);
    top->wr_rsp_tag   = w_tag;
    top->wr_rsp_error = 0;
}

// ---------------------------------------------------------------------------
static void tick(void)
{
    if (ddr_bpc > 0.0) {
        ddr_credit += ddr_bpc;
        if (ddr_credit > ddr_burst) ddr_credit = ddr_burst;
    }
    if (r_state == 1 && r_lat_left > 0) r_lat_left--;
    if (w_state == 1 && w_lat_left > 0) w_lat_left--;

    top->clk_core = 0; top->clk_io = 0;
    top->eval();

    hs.cmd_ready     = top->cmd_ready;
    hs.cfg_ready     = top->cfg_ready;
    hs.cfg_rsp_valid = top->cfg_rsp_valid;
    hs.cpl_valid     = top->cpl_valid;
    hs.rd_req_valid  = top->rd_req_valid;
    hs.rd_valid      = top->rd_valid;
    hs.rd_ready      = top->rd_ready;
    hs.wr_req_valid  = top->wr_req_valid;
    hs.wr_valid      = top->wr_valid;
    hs.wr_rsp_ready  = top->wr_rsp_ready;
    for (int i = 0; i < 4; i++) hs.wr_data[i] = top->wr_data[i];
    for (int k = 0; k < 16; k++) hs.wr_keep[k] = (top->wr_keep >> k) & 1;
    hs.wr_last       = top->wr_last;

    if (trace_on) {
        int v[6] = {hs.rd_req_valid, hs.rd_valid, hs.wr_req_valid,
                    hs.wr_valid, hs.wr_rsp_ready, (int)top->top_state};
        int *p[6] = {&tr_rd_req, &tr_rd_val, &tr_wr_req,
                     &tr_wr_val, &tr_wr_rsp, &tr_state};
        const char *n[6] = {"rd_req", "rd_val", "wr_req", "wr_val",
                            "wr_rsp_rdy", "state"};
        for (int i = 0; i < 6; i++) {
            if (v[i] != *p[i]) {
                *p[i] = v[i];
                printf("T %llu %s=%d\n", (unsigned long long)cyc, n[i], v[i]);
            }
        }
        fflush(stdout);
    }

    if (hs.cpl_valid && top->cpl_ready) {
        for (int i = 0; i < 4; i++) cpl_words[i] = top->cpl_data[i];
        char buf[160];
        uint16_t job = cpl_words[0] & 0xFFFF;
        uint8_t  slot = (cpl_words[0] >> 16) & 0xFF;
        uint8_t  op = (cpl_words[0] >> 24) & 0xFF;
        uint16_t st = cpl_words[1] & 0xFFFF;
        uint16_t det = (cpl_words[1] >> 16) & 0xFFFF;
        uint32_t units = cpl_words[2];
        uint32_t aux = cpl_words[3];
        snprintf(buf, sizeof(buf), "CPL %u %u %u %04x %04x %u %u",
                 job, slot, op, st, det, units, aux);
        cpl_log.push_back(buf);
        cpl_done.insert(job);
        printf("%s\n", buf);
        fflush(stdout);
    }

    top->clk_core = 1; top->clk_io = 1;
    top->eval();
    cyc++;

    // ---- external read channel ----
    if (r_state == 0) {
        if (hs.rd_req_valid) {
            r_addr  = top->rd_req_addr;
            r_rem   = top->rd_req_bytes;
            r_tag   = top->rd_req_tag;
            if (trace_on)
                printf("RREQ addr=%016llx bytes=%u tag=%u\n",
                       (unsigned long long)r_addr, r_rem, r_tag);
            read_beat_data();
            r_lat_left = ddr_lat;
            r_state = 1;
        }
    } else {
        if (hs.rd_valid && hs.rd_ready) {
            uint32_t nb = (r_rem >= 16) ? 16 : r_rem;
            if (ddr_bpc > 0.0) ddr_credit -= nb;
            rd_bytes += nb;
            if (r_rem <= 16) {
                r_state = 0;
                r_rem   = 0;
            } else {
                r_addr += 16;
                r_rem  -= 16;
                read_beat_data();
            }
        }
    }
    drive_read();

    // ---- external write channel ----
    if (w_state == 0) {
        if (hs.wr_req_valid) {
            w_addr  = top->wr_req_addr;
            w_rem   = top->wr_req_bytes;
            w_tag   = top->wr_req_tag;
            w_lat_left = ddr_lat;
            w_state = 1;
        }
    } else if (w_state == 1) {
        if (hs.wr_valid) {
            uint32_t nb = (w_rem >= 16) ? 16 : w_rem;
            if (ddr_bpc > 0.0) ddr_credit -= nb;
            wr_bytes += nb;
            for (int k = 0; k < 16; k++) {
                if (!hs.wr_keep[k]) continue;
                uint64_t a = w_addr + k;
                if (a < mem.size())
                    mem[a] = (hs.wr_data[k / 4] >> (8 * (k % 4))) & 0xFF;
            }
            if (hs.wr_last) {
                w_delay = 2;
                w_state = 2;
            } else {
                w_addr += 16;
                w_rem  -= 16;
            }
        }
    } else {
        if (hs.wr_rsp_ready) {
            w_state = 0;
        } else if (w_delay > 0) {
            w_delay--;
        }
    }
    drive_write();
#ifdef DUT_PROBES
    if (dprobe_on) print_probes();
#endif
#if VM_TRACE
    if (vcd && cyc >= vcd_start && cyc <= vcd_end) {
        vcd->dump(cyc);
    }
#endif
    if (cyc > cyc_max) {
        fprintf(stderr, "cycle limit reached (%llu)\n",
                (unsigned long long)cyc_max);
        exit(9);
    }
}

// ---------------------------------------------------------------------------
static void reset_dut(int n)
{
    top->reset_n = 0;
    top->platform_isolated = 1;
    top->run_req = 0;
    top->cmd_valid = 0;   top->cmd_data[0] = 0;
    top->cpl_ready = 1;
    top->cfg_valid = 0;   top->cfg_write = 0; top->cfg_addr = 0;
    top->cfg_wdata = 0;   top->cfg_rsp_ready = 0;
    r_state = 0; w_state = 0; w_delay = 0;
    drive_read();
    drive_write();
    for (int i = 0; i < n; i++) tick();
}

static void cfg_write(uint16_t addr, uint64_t data, bool expect_err)
{
    top->cfg_addr  = addr;
    top->cfg_wdata = data;
    top->cfg_write = 1;
    top->cfg_valid = 1;
    // hs holds the values the edge in this tick samples, so the tick that
    // reports ready is the tick that completes the handshake
    do { tick(); } while (!hs.cfg_ready);
    top->cfg_valid = 0;
    while (!hs.cfg_rsp_valid) tick();
    if ((bool)top->cfg_error != expect_err) {
        fprintf(stderr, "cfg %04x: error %d expected %d\n",
                addr, (int)top->cfg_error, (int)expect_err);
    }
    top->cfg_rsp_ready = 1;
    tick();
    top->cfg_rsp_ready = 0;
}

static void send_cmd(const uint32_t w[16])
{
    for (int i = 0; i < 16; i++) top->cmd_data[i] = w[i];
    top->cmd_valid = 1;
    do { tick(); } while (!hs.cmd_ready);
    top->cmd_valid = 0;
}

// ---------------------------------------------------------------------------
// Activation-B peek.  b_frame_reader and mem_arb2 sit inside mpu_engine, so a
// MATMUL whose B operand is a scratchpad frame can stall without changing a
// single DUT pin.  Walking the generated model's internals says which side of
// the handshake is idle and how far the reader got.
// ---------------------------------------------------------------------------
#ifdef MPU_PEEK
#define PK(f) ((unsigned long long)top->rootp->f)
static void peek_mpu(void)
{
    fprintf(stderr,
            "[peek %llu] mpu st=%llu job=%llu act=%llu wait=%llu fdone=%llu "
            "t=%llu/%llu cnt=%llu/%llu aacc=%llu bacc=%llu exp=%llu "
            "pdone=%llu ftl=%llu rel=%llu\n"
            "           core run=%llu done=%llu kk=%llu | "
            "pack st=%llu s=%llu/%llu dn=%llu | "
            "bfr st=%llu buf=%llu base=%llu str=%llu g=%llu/%llu "
            "s=%llu/%llu t=%llu/%llu dn=%llu err=%llu\n"
            "           arb g=%llu pref=%llu | pak v=%llu r=%llu "
            "rsp v=%llu r=%llu\n",
            (unsigned long long)cyc,
            PK(npu_top__DOT__u_mpu__DOT__state),
            PK(npu_top__DOT__u_mpu__DOT__job_r),
            PK(npu_top__DOT__u_mpu__DOT__b_act_r),
            PK(npu_top__DOT__u_mpu__DOT__tile_waiting),
            PK(npu_top__DOT__u_mpu__DOT__b_feed_done),
            PK(npu_top__DOT__u_mpu__DOT__t_r),
            PK(npu_top__DOT__u_mpu__DOT__tiles_r),
            PK(npu_top__DOT__u_mpu__DOT__b_tile_cnt),
            PK(npu_top__DOT__u_mpu__DOT__ksteps_r),
            PK(npu_top__DOT__u_mpu__DOT__a_accepted),
            PK(npu_top__DOT__u_mpu__DOT__b_accepted),
            PK(npu_top__DOT__u_mpu__DOT__exp_beats),
            PK(npu_top__DOT__u_mpu__DOT__pack_done_seen),
            PK(npu_top__DOT__u_mpu__DOT__fatal_r),
            PK(npu_top__DOT__u_mpu__DOT__tiles_released),
            PK(npu_top__DOT__u_mpu__DOT__u_core__DOT__running),
            PK(npu_top__DOT__u_mpu__DOT__u_core__DOT__done_r),
            PK(npu_top__DOT__u_mpu__DOT__u_core__DOT__kk),
            PK(npu_top__DOT__u_mpu__DOT__u_pack__DOT__state),
            PK(npu_top__DOT__u_mpu__DOT__u_pack__DOT__s_r),
            PK(npu_top__DOT__u_mpu__DOT__u_pack__DOT__k_steps),
            PK(npu_top__DOT__u_mpu__DOT__u_pack__DOT__done_r),
            PK(npu_top__DOT__u_mpu__DOT__u_bframe__DOT__state),
            PK(npu_top__DOT__u_mpu__DOT__u_bframe__DOT__buf_r),
            PK(npu_top__DOT__u_mpu__DOT__u_bframe__DOT__base_r),
            PK(npu_top__DOT__u_mpu__DOT__u_bframe__DOT__stride_r),
            PK(npu_top__DOT__u_mpu__DOT__u_bframe__DOT__g_r),
            PK(npu_top__DOT__u_mpu__DOT__u_bframe__DOT__ngrp_r),
            PK(npu_top__DOT__u_mpu__DOT__u_bframe__DOT__s_r),
            PK(npu_top__DOT__u_mpu__DOT__u_bframe__DOT__ksteps_r),
            PK(npu_top__DOT__u_mpu__DOT__u_bframe__DOT__t_r),
            PK(npu_top__DOT__u_mpu__DOT__u_bframe__DOT__tiles_r),
            PK(npu_top__DOT__u_mpu__DOT__u_bframe__DOT__done_r),
            PK(npu_top__DOT__u_mpu__DOT__u_bframe__DOT__error_r),
            PK(npu_top__DOT__u_mpu__DOT__u_arb__DOT__grant),
            PK(npu_top__DOT__u_mpu__DOT__u_arb__DOT__pref_b),
            PK(npu_top__DOT__u_mpu__DOT__pak_req_valid),
            PK(npu_top__DOT__u_mpu__DOT__pak_req_ready),
            PK(npu_top__DOT__u_mpu__DOT__pak_rsp_valid),
            PK(npu_top__DOT__u_mpu__DOT__pak_rsp_ready));
    fprintf(stderr,
            "           fe rp=%llu wp=%llu cnt=%llu | "
            "sched st=%llu kind=%llu job=%llu/%llu lok=%llu "
            "lv=%llu lr=%llu lw=%llu eng=%llu dep=%llu busy=%llu "
            "rm=%llu wm=%llu\n",
            PK(npu_top__DOT__u_frontend__DOT__rp),
            PK(npu_top__DOT__u_frontend__DOT__wp),
            PK(npu_top__DOT__u_frontend__DOT__count),
            PK(npu_top__DOT__u_sched__DOT__state),
            PK(npu_top__DOT__u_sched__DOT__kind),
            PK(npu_top__DOT__u_sched__DOT__job_c),
            PK(npu_top__DOT__u_sched__DOT__job_d),
            PK(npu_top__DOT__u_sched__DOT__locks_ok),
            PK(npu_top__DOT__u_sched__DOT__lock_v_d),
            PK(npu_top__DOT__u_sched__DOT__lock_r_d),
            PK(npu_top__DOT__u_sched__DOT__lock_w_d),
            PK(npu_top__DOT__u_sched__DOT__engine_ready),
            PK(npu_top__DOT__u_sched__DOT__dep_en_r),
            PK(npu_top__DOT__u_sched__DOT__busy_d),
            PK(npu_top__DOT__u_sched__DOT__dec_read_mask),
            PK(npu_top__DOT__u_sched__DOT__dec_write_mask));
}
#endif

static bool wait_job(unsigned job, long limit)
{
    if (cpl_done.count(job)) return true;
#ifdef MPU_PEEK
    const bool peek = getenv("NPU_PEEK") != nullptr;
#endif
    for (long n = 0; n < limit; n++) {
#ifdef MPU_PEEK
        if (peek && n % 5000 == 0) peek_mpu();
#endif
        for (auto &s : cpl_log) {
            unsigned j = 0;
            if (sscanf(s.c_str(), "CPL %u", &j) == 1 && j == job) {
                cpl_log.clear();
                return true;
            }
        }
        cpl_log.clear();
        tick();
    }
#ifdef MPU_PEEK
    if (peek) peek_mpu();
#endif
    return false;
}

static void wait_state(int s, long limit)
{
    for (long n = 0; n < limit && top->top_state != s; n++) tick();
    if (top->top_state != s) {
        fprintf(stderr, "state %d not reached (now %d)\n", s, top->top_state);
        exit(3);
    }
}

static void dump_mem(uint64_t addr, uint32_t len, const char *path)
{
    std::ofstream f(path, std::ios::binary);
    for (uint32_t i = 0; i < len; i++) {
        uint64_t a = addr + i;
        char c = (a < mem.size()) ? (char)mem[a] : 0;
        f.put(c);
    }
}

int main(int argc, char **argv)
{
    const char *mem_path = nullptr, *script_path = nullptr;
    double ddr_gbps = 0.0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--mem") && i + 1 < argc) mem_path = argv[++i];
        else if (!strcmp(argv[i], "--script") && i + 1 < argc) script_path = argv[++i];
        else if (!strcmp(argv[i], "--ddr-bpc") && i + 1 < argc) ddr_bpc = atof(argv[++i]);
        else if (!strcmp(argv[i], "--ddr-gbps") && i + 1 < argc) ddr_gbps = atof(argv[++i]);
        else if (!strcmp(argv[i], "--fcore-mhz") && i + 1 < argc) fcore_mhz = atof(argv[++i]);
        else if (!strcmp(argv[i], "--ddr-lat") && i + 1 < argc) ddr_lat = atol(argv[++i]);
        else if (!strcmp(argv[i], "--max-cycles") && i + 1 < argc)
            cyc_max = strtoull(argv[++i], nullptr, 0);
    }
    if (const char *e = getenv("MAX_CYCLES")) cyc_max = strtoull(e, nullptr, 0);
    if (const char *e = getenv("DDR_GBPS"))   ddr_gbps   = atof(e);
    if (const char *e = getenv("FCORE_MHZ"))  fcore_mhz  = atof(e);
    if (const char *e = getenv("DDR_BPC"))    ddr_bpc    = atof(e);
    if (const char *e = getenv("DDR_LAT"))    ddr_lat    = atol(e);
    if (ddr_gbps > 0.0) {
        if (fcore_mhz <= 0.0) {
            fprintf(stderr, "--ddr-gbps needs --fcore-mhz: bandwidth is per cycle\n");
            return 2;
        }
        ddr_bpc = ddr_gbps * 1e9 / (fcore_mhz * 1e6);
    }
    if (!script_path) {
        fprintf(stderr, "usage: %s --mem image.bin --script cmds.txt\n", argv[0]);
        return 2;
    }
    trace_on = getenv("TRACE") != nullptr;
#ifdef DUT_PROBES
    dprobe_on  = getenv("DPROBE") != nullptr;
    dprobe_sum = getenv("DPROBE_SUM") != nullptr;
    if (const char *s = getenv("DPROBE_FROM")) dprobe_from = strtoull(s, nullptr, 0);
    if (const char *s = getenv("DPROBE_TO"))   dprobe_to   = strtoull(s, nullptr, 0);
#endif

    // The image carries the whole model a run needs: a layer's packed
    // weights are 8.85 MB, a chained run keeps several layers in one image
    // (LAYER_STRIDE in scripts/real_layer.py), and the autoregressive demo
    // adds an int4 lm_head -- about 120 MiB.  Size the memory from the image
    // itself, in MiB, with a 32 MiB floor, instead of refusing anything big.
    uint64_t memsz = 1u << 25;
    if (mem_path) {
        std::ifstream probe(mem_path, std::ios::binary | std::ios::ate);
        if (probe) {
            uint64_t n = (uint64_t)probe.tellg();
            if (n > memsz)
                memsz = (n + (1u << 20) - 1) & ~((uint64_t)(1u << 20) - 1);
        }
    }
    mem.assign((size_t)memsz, 0);
    if (mem_path) {
        std::ifstream f(mem_path, std::ios::binary);
        if (!f) { fprintf(stderr, "cannot open %s\n", mem_path); return 2; }
        f.read((char *)mem.data(), mem.size());
    }

    Verilated::commandArgs(argc, argv);
    top = new DUT_CLS;
#if VM_TRACE
    if (const char *vcd_path = getenv("VCD")) {
        Verilated::traceEverOn(true);
        vcd = new VerilatedVcdC;
        top->trace(vcd, 99);
        vcd->open(vcd_path);
        atexit(close_vcd);
        if (const char *s = getenv("VCD_START")) vcd_start = strtoull(s, nullptr, 0);
        if (const char *s = getenv("VCD_END"))   vcd_end   = strtoull(s, nullptr, 0);
    }
#endif
    reset_dut(6);
    top->reset_n = 1;
    for (int i = 0; i < 6; i++) tick();

    wait_state(2, 200);                       // IDLE

    for (int b = 0; b < 8; b++) {
        cfg_write(0x0100 + 0x20 * b, ((uint64_t)b << 8) | 1, false);
        cfg_write(0x0108 + 0x20 * b, 0, false);
        cfg_write(0x0110 + 0x20 * b, 0x2000, false);
    }

    top->run_req = 1;
    wait_state(3, 200);                       // RUN

    std::ifstream sf(script_path);
    if (!sf) { fprintf(stderr, "cannot open %s\n", script_path); return 2; }
    std::string line;
    while (std::getline(sf, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream is(line);
        std::string op;
        is >> op;
        if (op == "CMD") {
            std::string hex;
            is >> hex;
            if (hex.size() != 128) { fprintf(stderr, "CMD needs 128 hex digits\n"); return 2; }
            uint32_t w[16];
            for (int i = 0; i < 16; i++) {
                std::string t = hex.substr(128 - 8 * (i + 1), 8);
                w[i] = (uint32_t)strtoul(t.c_str(), nullptr, 16);
            }
            send_cmd(w);
        } else if (op == "WAIT") {
            unsigned job; is >> job;
            // WAIT_LIMIT overrides the 20M-cycle default so a hung job is
            // caught in minutes while bringing up a new path
            long lim = 20000000L;
            if (const char *e = getenv("WAIT_LIMIT")) lim = atol(e);
            if (!wait_job(job, lim)) {
                fprintf(stderr, "timeout waiting for job %u\n", job);
                return 4;
            }
        } else if (op == "DUMP") {
            unsigned long long addr; unsigned len; std::string path;
            is >> std::hex >> addr >> len >> path;
            dump_mem(addr, len, path.c_str());
        } else if (op == "CLEAR") {
            // job ids are 16 bit, so a stream past 65536 commands reuses
            // them; drop the finished window's completions before that
            cpl_done.clear();
            cpl_log.clear();
        } else if (op == "DELAY") {
            long n; is >> n;
            for (long i = 0; i < n; i++) tick();
        } else {
            fprintf(stderr, "unknown script op '%s'\n", op.c_str());
            return 2;
        }
    }

    if (top->fatal) {
        fprintf(stderr, "FAIL: fatal %04x\n", (unsigned)top->fatal_code);
        return 5;
    }
    printf("DONE cycles=%llu rd_bytes=%llu wr_bytes=%llu rd_Bpc=%.2f",
           (unsigned long long)cyc, rd_bytes, wr_bytes,
           cyc ? (double)rd_bytes / (double)cyc : 0.0);
    if (fcore_mhz > 0.0)
        printf(" fcore=%.0fMHz rd_GBps=%.3f",
               fcore_mhz,
               (double)rd_bytes / (double)(cyc ? cyc : 1)
               * fcore_mhz * 1e6 / 1e9);
    printf("\n");
    top->final();
    delete top;
    return 0;
}

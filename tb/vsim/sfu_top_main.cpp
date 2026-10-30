//=============================================================================
// sfu_top_main.cpp : Verilator stream harness for sfu_top
//
//   Reads requests from stdin, one per line:
//       <op> <fmt_in> <fmt_out> <mode> <din lane0..lane7, hex>
//   Drives sfu_top and writes one line per output beat:
//       R <req> <beat> <dout lane0..lane7 hex> <status hex>
//
//   Beat count is 2 for SINCOS (op 8) and 1 otherwise; the harness waits for
//   exactly that many beats, so a missing or extra write-back shows up as a
//   stall rather than being silently accepted.
//
//   The reference model lives in Python (scripts/sfu_ref.py); this file only
//   moves data.
//=============================================================================
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <iostream>
#include <sstream>

#include "Vsfu_top.h"
#include "verilated.h"

static Vsfu_top *top;
static vluint64_t cyc = 0;

static void tick()
{
    top->clk = 0; top->eval();
    top->clk = 1; top->eval();
    cyc++;
}

static void reset()
{
    top->rst_n = 0;
    top->flush = 0;
    top->in_valid = 0;
    top->out_ready = 1;
    top->op = 0;
    top->fmt_in = 0;
    top->fmt_out = 0;
    top->mode = 0;
    for (int i = 0; i < 8; i++) top->din[i] = 0;
    for (int i = 0; i < 4; i++) tick();
    top->rst_n = 1;
    for (int i = 0; i < 2; i++) tick();
}

int main(int argc, char **argv)
{
    Verilated::commandArgs(argc, argv);
    top = new Vsfu_top;
    reset();

    std::string line;
    long req = 0;
    while (std::getline(std::cin, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream is(line);
        unsigned op, fi, fo, md;
        unsigned long long d[8];
        is >> op >> fi >> fo >> md;
        for (int i = 0; i < 8; i++) is >> std::hex >> d[i] >> std::dec;
        if (!is) { fprintf(stderr, "bad line %ld: %s\n", req, line.c_str()); return 2; }

        top->op = op; top->fmt_in = fi; top->fmt_out = fo; top->mode = md;
        // din is VlWide<8>: word i is lane i, lane 0 in bits [31:0]
        for (int i = 0; i < 8; i++) top->din[i] = (uint32_t)d[i];
        top->in_valid = 1;
        int guard = 0;
        while (!top->in_ready) {
            if (++guard > 4000) { fprintf(stderr, "timeout in_ready\n"); return 3; }
            tick();
        }
        tick();                       // handshake
        top->in_valid = 0;

        int beats = (op == 8) ? 2 : 1;
        for (int b = 0; b < beats; b++) {
            guard = 0;
            while (!top->out_valid) {
                if (++guard > 4000) { fprintf(stderr, "timeout out_valid\n"); return 4; }
                tick();
            }
            printf("R %ld %d", req, b);
            for (int i = 0; i < 8; i++) {
                printf(" %08x", (uint32_t)top->dout[i]);
            }
            printf(" %02x\n", (unsigned)top->status);
            fflush(stdout);
            tick();                   // consume the beat
        }
        req++;
    }
    top->final();
    delete top;
    return 0;
}

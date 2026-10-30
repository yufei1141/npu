//=============================================================================
// fifo_main : Verilator harness for op_fifo at 1024 / 2304 bits
//
//   Pushes N distinct entries into both FIFOs, then drains them, checking
//   that the head presented on each read is the entry that was written.
//   A wide-array read path that silently returns zeros fails here.
//
//   usage: Vfifo_wrap [n]
//=============================================================================
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>

#include "Vfifo_wrap.h"
#include "verilated.h"

static Vfifo_wrap *top;

static void tick(void)
{
    top->clk = 0; top->eval();
    top->clk = 1; top->eval();
}

// word i of an N-word vector carries a value derived from (tag, i)
static uint32_t patt(unsigned tag, int i) { return 0xA5000000u ^ (tag * 0x01010101u) ^ (uint32_t)i * 0x00010001u; }

int main(int argc, char **argv)
{
    Verilated::commandArgs(argc, argv);
    top = new Vfifo_wrap;

    int n = (argc > 1) ? atoi(argv[1]) : 6;
    if (n > 8) n = 8;

    top->clk = 0;
    top->rst_n = 0;
    top->a_wr = 0; top->a_rd = 0;
    top->b_wr = 0; top->b_rd = 0;
    for (int i = 0; i < 1024 / 32; i++) top->a_din[i] = 0;
    for (int i = 0; i < 2304 / 32; i++) top->b_din[i] = 0;
    for (int i = 0; i < 4; i++) tick();
    top->rst_n = 1;
    tick();

    // ---- fill ----
    for (int t = 0; t < n; t++) {
        for (int i = 0; i < 1024 / 32; i++) top->a_din[i] = patt(t, i);
        for (int i = 0; i < 2304 / 32; i++) top->b_din[i] = patt(t + 0x40, i);
        top->a_wr = 1; top->b_wr = 1;
        tick();
    }
    top->a_wr = 0; top->b_wr = 0;
    tick();

    printf("after fill: a_empty=%d b_empty=%d a_full=%d b_full=%d\n",
           (int)top->a_empty, (int)top->b_empty, (int)top->a_full, (int)top->b_full);

    // ---- drain ----
    int bad = 0;
    for (int t = 0; t < n; t++) {
        int a_ok = 1, b_ok = 1;
        for (int i = 0; i < 1024 / 32; i++)
            if (top->a_dout[i] != patt(t, i)) a_ok = 0;
        for (int i = 0; i < 2304 / 32; i++)
            if (top->b_dout[i] != patt(t + 0x40, i)) b_ok = 0;
        printf("entry %d: a %s (a0=%08x want %08x)  b %s (b0=%08x want %08x)\n",
               t, a_ok ? "OK" : "BAD", top->a_dout[0], patt(t, 0),
               b_ok ? "OK" : "BAD", top->b_dout[0], patt(t + 0x40, 0));
        if (!a_ok || !b_ok) bad++;
        top->a_rd = 1; top->b_rd = 1;
        tick();
    }
    top->a_rd = 0; top->b_rd = 0;
    tick();

    printf("after drain: a_empty=%d b_empty=%d\n", (int)top->a_empty, (int)top->b_empty);
    printf("%s\n", (bad == 0 && top->a_empty && top->b_empty) ? "FIFO PASS" : "FIFO FAIL");
    top->final();
    delete top;
    return bad ? 1 : 0;
}

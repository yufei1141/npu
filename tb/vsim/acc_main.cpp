//=============================================================================
// acc_main : Verilator harness for fp32_acc alone
//
//   Loads lane 0 with 16.0, then accumulates 16.0 three more times.
//   Expected acc_out[0] = 16, 32, 48, 64 and acc_out[1] untouched.
//
//   usage: Vfp32_acc
//=============================================================================
#include <cstdio>
#include <cstdint>

#include "Vfp32_acc.h"
#include "verilated.h"

static Vfp32_acc *top;

static void tick(void)
{
    top->clk = 0; top->eval();
    top->clk = 1; top->eval();
}

int main(int argc, char **argv)
{
    Verilated::commandArgs(argc, argv);
    top = new Vfp32_acc;

    top->clk = 0;
    top->rst_n = 0;
    top->valid = 0;
    top->acc_en = 0;
    for (int i = 0; i < 4096 / 32; i++) top->y[i] = 0;
    for (int i = 0; i < 4; i++) tick();
    top->rst_n = 1;
    tick();

    top->y[0] = 0x41800000u;         // 16.0
    top->y[1] = 0x3F800000u;         // 1.0, never accumulated (lane 1)

    printf("reset : acc0=%08x acc1=%08x\n", top->acc_out[0], top->acc_out[1]);

    top->valid = 1;
    top->acc_en = 0;                 // load
    tick();
    printf("load  : acc0=%08x acc1=%08x  (want 41800000 3f800000)\n",
           top->acc_out[0], top->acc_out[1]);

    top->acc_en = 1;                 // accumulate on lane 1 only stays a load
    top->y[1] = 0;
    for (int i = 0; i < 3; i++) {
        tick();
        printf("acc%d  : acc0=%08x acc1=%08x\n", i + 1,
               top->acc_out[0], top->acc_out[1]);
    }

    int ok = (top->acc_out[0] == 0x42800000u);   // 64.0
    printf("%s\n", ok ? "ACC PASS" : "ACC FAIL");
    top->final();
    delete top;
    return ok ? 0 : 1;
}

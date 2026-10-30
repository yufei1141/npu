//=============================================================================
// gemv_main : Verilator harness for gemv_cluster alone
//
//   Reproduces tb_gemv_cluster case 1 without any Verilog delays: one beat,
//   A = 1.0 on all 16 K rows, every weight nibble 4, every E8M0 scale 125.
//   The expected fp32 result is 16.0 per column (16 * 4 * 2^-2).
//
//   usage: Vgemv_cluster [beats]
//=============================================================================
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>

#include "Vgemv_cluster.h"
#include "verilated.h"

static Vgemv_cluster *top;
static uint64_t cyc = 0;

static void tick(void)
{
    top->clk = 0; top->eval();
    top->clk = 1; top->eval();
    cyc++;
}

// one 256-bit group beat: 16 K rows x 4 columns x 4 bits, row major,
// column c at nibble 3-c; E8M0 per PE.
static void fill_group(uint8_t *g256, int nib)
{
    memset(g256, 0, 32);
    for (int r = 0; r < 16; r++) {
        uint16_t row = 0;
        for (int c = 0; c < 4; c++)
            row |= (uint16_t)(nib & 0xF) << (4 * (3 - c));
        g256[2 * r]     = row & 0xFF;
        g256[2 * r + 1] = row >> 8;
    }
}

int main(int argc, char **argv)
{
    Verilated::commandArgs(argc, argv);
    top = new Vgemv_cluster;

    int beats = (argc > 1) ? atoi(argv[1]) : 1;

    top->clk = 0;
    top->rst_n = 0;
    top->mode = 0; top->m_len = 1; top->n_grp = 1;
    top->beat_en = 0; top->first = 0; top->last = 0;
    for (int i = 0; i < 1024 / 32; i++) top->a_din[i] = 0;
    for (int i = 0; i < 2304 / 32; i++) top->b_din[i] = 0;
    for (int i = 0; i < 4; i++) tick();

    // A: 16 fp16 ones per row lane, all 4 lanes (1024 bits = 32 words)
    for (int m = 0; m < 4; m++)
        for (int k = 0; k < 16; k++) {
            int word = (256 * m + 16 * k) / 32;
            int sh   = (256 * m + 16 * k) % 32;
            top->a_din[word] |= (uint32_t)0x3C00u << sh;
        }

    // B: group 0 = nibble 4, scale 125; other groups idle (scale 127 fill)
    // 2304 bits = 72 words; byte j lives in word j/4 at shift 8*(j%4)
    uint8_t g[32];
    fill_group(g, 4);
    uint32_t bw[72];
    memset(bw, 0, sizeof(bw));
    for (int i = 0; i < 32; i++)
        bw[i / 4] |= (uint32_t)g[i] << (8 * (i % 4));
    for (int n = 1; n < 8; n++)
        for (int e = 0; e < 4; e++) {
            int j = 2048 + 4 * n + e;
            bw[j / 4] |= (uint32_t)127 << (8 * (j % 4));
        }
    for (int i = 0; i < 72; i++) top->b_din[i] = bw[i];

    top->rst_n = 1;
    for (int i = 0; i < 2; i++) tick();

    for (int b = 0; b < beats; b++) {
        top->beat_en = 1;
        top->first = (b == 0);
        top->last = (b == beats - 1);
        tick();
    }
    top->beat_en = 0; top->first = 0; top->last = 0;

    int valid_seen = 0;
    for (int i = 0; i < 40; i++) {
        tick();
        if (top->acc_valid) {
            valid_seen++;
            printf("acc_valid at cyc %llu\n", (unsigned long long)cyc);
            for (int c = 0; c < 4; c++)
                printf("  row0 col%d = %08x\n", c, top->acc_out[c * 32 / 32 * 0 + c]);
            break;
        }
    }
    // acc_out row0 columns 0..3 live in bits [31:0], [95:64]? print raw words
    for (int i = 0; i < 8; i++)
        printf("  word[%d] = %08x\n", i, top->acc_out[i]);
    if (!valid_seen) printf("NO acc_valid\n");
    top->final();
    delete top;
    return 0;
}

# SFU 规格（端侧 LLM 推理，共享资源型）

## 1. 定位

面向端侧大模型推理（batch=1，1B–7B 量级）的向量 SFU。
端侧约束：面积和能效优先，吞吐只要不成为瓶颈即可。

三条基本判断：

1. **I/O 16-bit，内部 32-bit**：输入输出 BF16/FP16，指数/范围缩减/多项式累加用 FP32，
   避免 BF16 在 softmax 和 LayerNorm 上的精度损失。
2. **算子不并发**：同一时刻只执行一种非线性函数，因此所有函数共享同一套运算资源，
   不需要为每个函数做独立流水。
3. **低 lane 数**：端侧 NPU 的向量宽度和算力有限，4 lanes 已足够；不需要 16/32 lanes。

## 2. 功能集

### 2.1 硬件原语（7 个）

| Opcode | 函数 | PTX 对应 | LLM 用途 |
|---|---|---|---|
| 0 | `EXP2(x)` | `ex2.approx.f32` | softmax / online softmax |
| 1 | `LOG2(x)` | `lg2.approx.f32` | log-sum-exp、pow |
| 2 | `RCP(x)` | `rcp.approx.f32` | softmax 归一化、sigmoid |
| 3 | `RSQRT(x)` | `rsqrt.approx.f32` | RMSNorm / LayerNorm |
| 4 | `SQRT(x)` | `sqrt.approx.f32` | norm 兜底 |
| 5 | `SIN(x)` | `sin.approx.f32` | RoPE |
| 6 | `COS(x)` | `cos.approx.f32` | RoPE |
| 7 | `TANH(x)` | `tanh.approx.f32` | GELU / tanh |
| 8 | `SINCOS(x)` | 双输出 | RoPE 专用：一次范围缩减同时出 sin/cos |

`exp / ln / sigmoid / SiLU / GELU / softmax / RMSNorm` 由软件组合，不占额外硬件。
`SINCOS` 不是新算术单元，只是复用 SIN/COS 的微码并把两个结果一起输出。

### RoPE 说明

RoPE 的旋转本身（4 次乘加/维度对）由向量 ALU 完成，SFU 只负责 `sin/cos`。
长上下文角度可达 ~3×10⁴ rad，在线计算的 FP32 范围缩减可能引入 ~0.002 rad 相位误差；
默认建议查表/缓存，`SINCOS` 作为兜底。

### 2.2 软件组合

| 算子 | 组合 |
|---|---|
| `exp(x)` | `EXP2(x * log2e)` |
| `ln(x)` | `LOG2(x) * ln2` |
| `sigmoid(x)` | `RCP(1 + EXP2(-x*log2e))` |
| `SiLU(x)` | `x * sigmoid(x)` |
| `tanh(x)` | 硬件 `TANH`，或 `1 - 2*RCP(EXP2(2x*log2e)+1)` |
| `GELU(x)` | `0.5x(1 + tanh(0.79788456(x + 0.044715x^3)))` |
| `softmax` | max → EXP2 → sum → RCP → 乘 |
| `RMSNorm` | mean(x²) → RSQRT → 乘 |

### 2.3 不支持

- `erf/erfc`（GELU 用 tanh 近似）
- 双曲函数、`atan2`、`pow`（软件组合）
- 完整 IEEE-754 所有舍入模式

## 3. 数据格式与精度

| 项 | 规格 |
|---|---|
| 输入 | BF16 / FP16（可选 FP8 E4M3） |
| 输出 | BF16 / FP16，或 FP32（`fmt_out` 选择） |
| 内部 | **FP32**：指数、范围缩减、舍入、累加 |
| 多项式乘法 | FP32；或 FP16/BF16 乘 + FP32 累加（面积模式） |
| 舍入 | RNE；FTZ 可配置 |
| 特殊值 | NaN / ±Inf / ±0 / 非规格化 |

精度目标（ULP）：

| 函数 | 输出 BF16/FP16 | 输出 FP32 |
|---|---|---|
| EXP2 / LOG2 | ≤ 1 | ≤ 2 |
| RCP / RSQRT / SQRT | ≤ 2 | ≤ 4 |
| SIN / COS | ≤ 2 | ≤ 4（`|x| ≤ π`） |
| TANH | ≤ 2 | ≤ 4 |

> 16-bit 输出时，最终误差主要来自输出量化（0.5 ULP）；内部 FP32 保证在量化前不丢精度。

## 4. 性能

| 指标 | 目标 |
|---|---|
| lanes | **4**（参数化 2/4/8） |
| 运算阵列 | **1 套 4-lane FP32 FMA**，所有函数共享 |
| 峰值 | 1 micro-op/cycle（4 lanes 并行执行同一条微指令） |
| 每函数周期 | 5–14 cycles / 4-element vector |
| 有效吞吐 | 0.3–0.8 vector-op/cycle（取决于函数） |
| 延迟 | ≤ 16 cycles |
| 启动间隔 (II) | = 微指令条数（不做多 op 交织） |
| 时钟 | 500–800 MHz（端侧典型） |
| 面积目标 | ≤ 5% NPU 面积 |

### 4.1 端侧吞吐估算

假设 7B 模型、每 token 约 100 万次非线性运算：

| 方案 | 有效吞吐 | 每 token 非线性耗时 @600 MHz |
|---|---|---|
| 4 lanes，1 FMA | ~0.5 op/cycle | ~3.3 ms |
| 8 lanes，1 FMA | ~1.0 op/cycle | ~1.7 ms |
| 4 lanes，2 FMA（P/N 各一） | ~1.0 op/cycle | ~1.7 ms |

若 token 总时间为 20 ms，4-lane 方案占 ~16%；对 1B–3B 模型占比更低。
**结论：4 lanes 可以接受；若目标 7B 且 decode 延迟敏感，选 8 lanes。**

## 5. 接口

```
sfu_top
  clk, rst_n
  in_valid, in_ready
  op[3:0], fmt_in[1:0], fmt_out[1:0], mode[1:0]
  din[LANES*DATA_W-1:0]      // 每 lane DATA_W bit；FP8 时两两打包
  out_valid, out_ready
  dout[LANES*ACC_W-1:0]      // 每 lane ACC_W bit；16-bit 输出时低 DATA_W 位有效
  flush, status[7:0]
```

完整的端口/参数/编码/存储定义见 `sfu/arch/microarch.md` 的「接口契约」一节。

- 所有 lane 共享 `op/fmt/mode`。
- `mode`：0=fast，1=accurate（多一次 Newton / 高阶多项式）。
- `status`：overflow / underflow / invalid / div0。
- 引擎忙时 `in_ready=0`；一次 vector op 占用引擎 5–14 cycles。

## 6. 异常与边界

| 条件 | 行为 |
|---|---|
| EXP2 上溢/下溢 | +Inf / +0，置标志 |
| LOG2(x≤0) | -Inf / NaN，置 invalid |
| RCP(0) | ±Inf，置 div0 |
| RSQRT(0) / RSQRT(x<0) | +Inf / NaN，置 div0/invalid |
| SQRT(x<0) | NaN，置 invalid |
| SIN/COS(Inf/NaN) | NaN，置 invalid |
| 非规格化 | 按 FTZ 配置 |

## 7. 验收标准

- [ ] BF16/FP16 下每个原语**穷举 65536 个输入位型**，满足 ULP 目标
- [ ] FP32 每函数 ≥ 100 万随机向量 + 特殊值
- [ ] 4 lanes 连续向量吞吐符合 4.1 表
- [ ] 延迟 ≤ 16 cycles
- [ ] 综合无 latch / 组合环 / 多驱动
- [ ] `chip-balance` 无离群
- [ ] 面积/功耗：NOT RUN（无 PDK）

## 8. 非目标

- 后端 PD、DFT、门级 STA、UVM
- 多 op 并发执行（同一时刻只跑一种函数）
- 与 NVIDIA 相同的 ISA 编码

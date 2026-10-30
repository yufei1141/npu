# SFU 微架构（端侧 D1：共享 FMA + 微码）

## 1. 顶层结构

```
 din[4×16] ──► unpack / fmt convert ──► special-case detect
                                            │
 op[3:0] ───────────────────────────────────┤
                                            ▼
                                  ┌───────────────────┐
                                  │  microcode FSM    │
                                  │  (op → uPC)       │
                                  └─────────┬─────────┘
                                            │ control
              ┌─────────────────────────────┼──────────────────────────┐
              │                             ▼                          │
              │                ┌──────────────────────────┐            │
              │                │  shared 4-lane FP32 FMA  │            │
              │                └──────────────────────────┘            │
              │                             ▲                          │
              │        ┌────────────────────┼───────────────────┐      │
              │        │  integer unit      │  coefficient ROM   │      │
              │        │  exponent/round    │  (broadcast)       │      │
              │        └────────────────────┼───────────────────┘      │
              │                             │                          │
              │        ┌────────────────────▼───────────────────┐      │
              │        │  4 × 64-entry LUT (RCP/RSQRT 初值)      │      │
              │        └────────────────────────────────────────┘      │
              │                             │                          │
              └─────────────────────────────▼──────────────────────────┘
                                            │
                                  round / pack / flags
                                            ▼
                                  dout[4×16] 或 dout[4×32]
```

## 1.1 接口契约（强制）

信号名、位宽、参数在 RTL 中必须逐字一致，不得改名或改宽。

### 顶层端口

| 信号 | 方向 | 位宽 | 时钟域 | 复位 | 说明 |
|---|---|---|---|---|---|
| `clk` | in | 1 | — | — | 时钟 |
| `rst_n` | in | 1 | `clk` | 异步低有效 | 复位 |
| `in_valid` | in | 1 | `clk` | — | 输入有效 |
| `in_ready` | out | 1 | `clk` | `rst_n` | 可接收 |
| `op` | in | 4 | `clk` | — | 见 opcode 表 |
| `fmt_in` | in | 2 | `clk` | — | 输入格式 |
| `fmt_out` | in | 2 | `clk` | — | 输出格式 |
| `mode` | in | 2 | `clk` | — | 0=fast, 1=accurate |
| `din` | in | `LANES*DATA_W` | `clk` | — | 每 lane `DATA_W` bit |
| `out_valid` | out | 1 | `clk` | `rst_n` | 输出有效 |
| `out_ready` | in | 1 | `clk` | — | 下游可接收 |
| `dout` | out | `LANES*ACC_W` | `clk` | `rst_n` | 每 lane `ACC_W` bit |
| `status` | out | 8 | `clk` | `rst_n` | 异常标志 |
| `flush` | in | 1 | `clk` | — | 清空流水 |

### 参数

| 参数 | 类型 | 默认 | 范围 | 说明 |
|---|---|---|---|---|
| `LANES` | int | 4 | 2/4/8 | 每拍处理的元素数 |
| `DATA_W` | int | 16 | 16 | I/O 位宽 |
| `ACC_W` | int | 32 | 32 | 内部/输出位宽 |
| `LUT_DEPTH` | int | 64 | 32/64/128 | RCP/RSQRT 初值表深度 |
| `LUT_W` | int | 16 | 16 | LUT 数据位宽 |
| `LUT_IDX_W` | int | `$clog2(LUT_DEPTH)` | 派生 | LUT 索引位宽 |
| `COEF_NUM` | int | 35 | — | 系数 ROM 项数（按实际需求，无预留） |
| `MICRO_ROM_DEPTH` | int | 112 | — | 微码 ROM 深度（按实际需求，无预留） |
| `UOP_W` | int | 32 | — | 微指令位宽 |
| `REGS_PER_LANE` | int | 6 | — | 每 lane 寄存器数 |
| `MUL_PREC` | enum | `FP32` | `FP32`/`FP16` | 乘法精度模式 |
| `FTZ` | bit | 0 | 0/1 | 非规格化数清零 |

### 子模块 `sfu_slice` 端口

`SLICE_LANES = 4`，`LANES` 必须是 `SLICE_LANES` 的整数倍。

| 信号 | 方向 | 位宽 | 说明 |
|---|---|---|---|
| `clk`, `rst_n` | in | 1 | 时钟/复位 |
| `uop` | in | `UOP_W` | 微指令 |
| `coef` | in | `ACC_W` | 广播系数 |
| `lut_idx` | in | `SLICE_LANES*LUT_IDX_W` | LUT 索引 |
| `din` | in | `SLICE_LANES*ACC_W` | 4 lane 输入 |
| `dout` | out | `SLICE_LANES*ACC_W` | 4 lane 输出 |
| `ready` | out | 1 | 本 slice 可接收 |

### 编码表

`op[3:0]`：

| 值 | 函数 |
|---|---|
| 0 | `EXP2` |
| 1 | `LOG2` |
| 2 | `RCP` |
| 3 | `RSQRT` |
| 4 | `SQRT` |
| 5 | `SIN` |
| 6 | `COS` |
| 7 | `TANH` |
| 8 | `SINCOS`（双输出） |
| 9–15 | 保留 |

`fmt_in[1:0]` / `fmt_out[1:0]`：

| 值 | 格式 |
|---|---|
| 0 | FP32 |
| 1 | BF16 |
| 2 | FP16 |
| 3 | FP8 E4M3 |

`mode[1:0]`：0=fast，1=accurate，2–3 保留。

`status[7:0]`：bit0 overflow，bit1 underflow，bit2 invalid，bit3 div0，bit4–7 保留。
所有 lane 的异常按位 OR 汇总到 `status`；需要逐 lane 诊断时由验证环境单独采集。

> `SINCOS` 的输出方式：使用**两个写回周期**，第 1 拍 `dout` 为 sin，第 2 拍为 cos，
> `out_valid` 连续有效两次；下游必须能接收两个 beat。若下游不能接受，再评估增加
> `dout_cos[LANES*ACC_W-1:0]` 端口（面积代价）。

### 存储规格

| 存储 | 深度 × 位宽 | 端口 | 说明 |
|---|---|---|---|
| LUT | `LUT_DEPTH × LUT_W` | 每 lane 1 读 | 每 slice 4 份 |
| 系数 ROM | `COEF_NUM × ACC_W` | 1 读（广播） | 全局共享 |
| 微码 ROM | `MICRO_ROM_DEPTH × UOP_W` | 1 读 | 全局共享 |
| 寄存器堆 | `REGS_PER_LANE × ACC_W` / lane | 2 读 1 写 | 每 lane |

## 2. 执行模型

- **一条微指令同时作用于 4 个 lane**：4 个元素共享同一条控制信号和系数。
- 不同函数通过不同的微程序实现；`op` 选择微程序入口。
- 同一时刻只执行一个 vector op；引擎忙时 `in_ready=0`。
- 不做多 op 交织（端侧不并发），所以控制简单、面积小。

### 2.1 微指令格式（示意）

| 字段 | 含义 |
|---|---|
| `uop` | FMA / MOV / LUT / ROUND / SHIFT / PACK / NOP |
| `src_a, src_b, src_c` | 寄存器索引 |
| `dst` | 寄存器索引 |
| `coef_addr` | 系数 ROM 地址 |
| `flags` | 舍入模式、lane 掩码 |

- 每个 lane 一组通用寄存器：`x, f, y, t, e, p`（6 × 32 bit）。
- 系数 ROM 广播；LUT 按 lane 独立索引。

## 3. 流水与延迟

| 阶段 | 内容 | 周期 |
|---|---|---|
| S0 | 解包、16→32 转换、特殊值检测 | 1 |
| S1 | 范围缩减 / LUT 读 / 指数提取 | 1 |
| EX | 微码循环（FMA / 整数 / 选择） | 5–14 |
| S2 | 舍入、32→16 打包、异常标志 | 1 |

- 总延迟：`2 + micro_ops + 1`，约 8–16 cycles。
- 启动间隔 II = `micro_ops`；不做 op 级交织。
- **假设 FMA 延迟 1 cycle**。若真实标准单元库需要 2–3 cycle，微码在依赖处插入等待，
  延迟和吞吐按比例增加；这一点必须在有工艺库后重新评估。

## 4. 各函数的微程序

### 4.1 EXP2（约 10 条微指令）

```
S0: check_special(x)
S1: k = round(x)            // 整数单元
EX: f = x - k               // FMA
    p = c5
    p = p*f + c4            // 4 次 Horner
    p = p*f + c3
    p = p*f + c2
    p = p*f + c1
    p = p*f + c0
S2: y = ldexp(p, k); pack
```

- BF16/FP16 输出：4 阶即可；FP32：5 阶。
- 上溢/下溢在 `ldexp` 阶段 clamp。

### 4.2 LOG2（约 12 条）

```
S1: 规格化 x = m·2^e, m ∈ [1,2)
EX: f = m - 1
    p = c5; p = p*f+c4; ...; p = p*f+c0
    y = e + p
S2: pack
```

- 用 `f = m-1` 而不是 `(m-1)/(m+1)`，避免除法；阶数提高 1–2 阶补偿。
- 若精度不足，改用 `(m-1)/(m+1)`，复用 RCP 微码。

### 4.3 RCP（约 9 条）

```
S1: y0 = LUT[top bits]
EX: t = x*y0
    y1 = y0*(2 - t)          // Newton 1
    t  = x*y1
    y2 = y1*(2 - t)          // Newton 2（FP32）
S2: pack
```

- BF16/FP16 输出：1 次 Newton；FP32 输出：2 次。

### 4.4 RSQRT（约 11 条）

```
S1: y0 = LUT[top bits]
EX: t = y0*y0
    t = x*t
    y1 = y0*(1.5 - 0.5*t)    // Newton 1
    t  = y1*y1
    t  = x*t
    y2 = y1*(1.5 - 0.5*t)    // Newton 2
S2: pack
```

### 4.5 SQRT（约 12 条）

```
RSQRT 微码
y = x * y_rsqrt
```

### 4.6 SIN / COS（各约 10 条）

```
S1: n = round(x * 2/pi)
EX: r = x - n*(pi/2)         // Cody-Waite，2–3 段
    p = sin/cos 多项式（5 阶）
    q = n mod 4
S2: 按 q 选择 sin/cos、符号；pack
```

- 支持 `|x| ≤ 2^20`；超出报 invalid。

### 4.7 TANH（约 21 条）

```
EXP2 微码(t = exp2(2x*log2e))
RCP 微码(r = 1/(t+1))
y = 1 - 2*r
```

- 复用 EXP2 与 RCP 的微码，不新增算术资源。
- 若 GELU 成为热点，增加一条融合微指令路径（仍共享 FMA）。

## 5. 共享资源与面积

| 资源 | 规格 | 说明 |
|---|---|---|
| FP32 FMA | 4 lanes | 所有函数共享；面积主体 |
| 整数单元 | 1 | 指数提取、round、ldexp、象限 |
| 系数 ROM | `COEF_NUM × ACC_W`，默认 35 × 32 bit | 广播，不随 lane 复制 |
| LUT | 4 × 64 × 16 bit | RCP/RSQRT 初值 |
| 寄存器 | 4 lanes × 6 × 32 bit | 微码寄存器 |
| 微码 ROM | `MICRO_ROM_DEPTH × UOP_W`，默认 112 × 32 bit | 用 Python 生成 |

- 预计 2–5 万门（generic Yosys），RTL 骨架后校准。
- 面积优化开关：`MUL_PREC=FP16`（FP16 乘 + FP32 累加），可再省 30–50%，需穷举验证精度。

### 5.1 系数 ROM（默认 35 × 32 bit）

存放多项式系数和范围缩减常量。所有 lane 广播同一份。

地址分配（示例）：

| 地址 | 内容 |
|---|---|
| 0–5 | EXP2：`c0..c5` |
| 6–12 | LOG2：`c0..c6` |
| 13–18 | SIN：`c0..c5` |
| 19–24 | COS：`c0..c5` |
| 25 | `log2(e)` |
| 26 | `ln(2)` |
| 27–28 | `2/π` 高/低段（Cody-Waite） |
| 29–30 | `π/2` 高/低段（Cody-Waite） |
| 31–33 | `1.0 / 0.5 / 1.5` |
| 34 | `2.0`（RCP Newton） |

共 **35 项**，不做预留。若将来 `accurate` 模式需要独立系数，再扩 ROM；当前 accurate
模式复用同一组系数，通过增加 Newton 迭代次数提升精度。

**数据来源**：

1. 用 Python `mpmath`/`numpy` 在缩减后的区间上做 minimax（Remez）拟合。
2. 量化到 FP32；`MUL_PREC=FP16` 时再量化一份。
3. 用高精度参考模型扫描整个区间，统计 max ULP；不达标就升阶或换区间。
4. 生成 `coef_rom.hex` + `coef_rom.json`（含版本号与误差统计）。

**使用方式**：

- 微指令的 `coef_addr` 选中一项，直接作为 FMA 的 C 操作数。
- 例：`FMA r_p, r_p, r_f, COEF(c4)`。
- 广播到 4 个 lane；所有 lane 用同一系数。

**为什么用 ROM**：系数随精度模式、阶数、区间变化；ROM 让调参不改 RTL。

### 5.2 微码 ROM（默认 112 × 32 bit）

存放每个 opcode 的微指令序列。

微指令字段（示意）：

| 字段 | 位宽 | 说明 |
|---|---|---|
| `uop` | 4 | FMA / MOV / LUT / ROUND / LDEXP / PACK / END |
| `src_a` | 3 | 寄存器索引 |
| `src_b` | 3 | 寄存器索引 |
| `src_c` / `coef_addr` | 6 | 寄存器或系数 ROM 地址 |
| `dst` | 3 | 寄存器索引 |
| `flags` | 4 | 舍入模式 / lane 掩码 / 双写回 |
| `end` | 1 | 序列结束 |
| 预留 | 8 | |

地址分配（示例）：

| opcode | 起始地址 | 条数 |
|---|---|---|
| EXP2 | 0 | 10 |
| LOG2 | 10 | 12 |
| RCP | 22 | 9 |
| RSQRT | 31 | 11 |
| SQRT | 42 | 12 |
| SIN | 54 | 10 |
| COS | 64 | 10 |
| TANH | 74 | 21 |
| SINCOS | 95 | 17 |

共 **112 条**，不做预留。起始地址由 `gen_microcode.py` 生成，上面是当前估算；
生成器输出的 `microcode.json` 才是最终地址表。

**数据来源**：

1. 用 Python 写一个微汇编器/DSL，把每个函数展开成微指令列表。
2. 生成 `microcode.hex` + `microcode.json`。
3. 用 Python 微码模拟器逐条执行，与双精度参考模型对比。
4. 生成版本号和 checksum，写入构建产物。

**使用方式**：

- `op` 查到起始地址，微程序计数器逐条取指。
- 每拍：读寄存器 → 读系数 ROM → FMA → 写回。
- `end` 置位后回 idle，`out_valid` 拉高。
- SINCOS 用 `flags` 标记两次写回（sin / cos）。

**为什么用 ROM**：同一套 FMA 支持所有函数；修 bug 或加函数只改微码，不改数据通路。

**实现方式**：仿真用 `$readmemh`；ASIC 用标准单元 ROM 或初始化寄存器阵列；
FPGA 用推断 ROM。hex 必须与 RTL 版本一起提交。

## 6. 精度预算

| 源 | 误差 |
|---|---|
| 16-bit 输入量化 | 0.5 ULP |
| 16→32 转换 | 0 |
| 范围缩减 | < 0.25 ULP |
| 多项式 | < 1 ULP |
| Newton（1–2 次） | < 0.5–1 ULP |
| 输出舍入 | 0.5 ULP |
| 合计 | ≤ 1.5–2.5 ULP |

- BF16/FP16 输出目标 ≤ 2 ULP（RCP/SQRT 类）或 ≤ 1 ULP（EXP/LOG）。
- 所有系数用 Python `mpmath`/`numpy` 拟合，定点化后再验证。

## 7. 验证钩子

- 顶层提供位型级接口，Python 生成 BF16/FP16 穷举向量。
- 微码 ROM 由 Python 生成，同时生成一份“微码级参考模型”，用于定位是算法错还是微码错。
- 吞吐测试：连续发送同 op 向量，测量 `out_valid` 间隔，验证 II 与设计值一致。
- 每个函数单独测：EXP2、LOG2、RCP、RSQRT、SQRT、SIN、COS、TANH。

## 8. 8-lane 扩展：实例化两次 4-lane，但只复制数据通路

**不要实例化两个完整的 SFU**（两套微码 FSM + 两套系数 ROM），那会浪费控制面积。
正确做法：

```
sfu_top (LANES = 8)
├── shared: op 解码 / 微码 FSM / 微码 ROM / 系数 ROM / 整数单元
├── sfu_slice[0]: 4×FMA + 4×寄存器 + 4×LUT
└── sfu_slice[1]: 4×FMA + 4×寄存器 + 4×LUT
```

- 把 4-lane 数据通路封装成 `sfu_slice`，顶层用 `generate` 实例化 `LANES/4` 个。
- 控制、系数 ROM、微码 ROM 共享；整数单元很小，可共享也可每 slice 一份。
- `in_ready = &slice_ready`；所有 slice 执行同一微指令。
- 面积约为 1.7–1.9×（不是 2×），吞吐翻倍（每 vector op 周期不变）。
- 特殊值走 lane 级 predication，不能让控制分叉。

> 只有当需要两个不同 op 同时执行时，才值得实例化两套完整 SFU。
> 端侧算子不并发，共享控制是更优解。

## 9. RoPE 支持

RoPE 对每一对维度 `(x0, x1)`、角度 `φ`：

```
y0 = x0*cos(φ) - x1*sin(φ)
y1 = x0*sin(φ) + x1*cos(φ)
```

SFU 能提供 `sin/cos`，但完整的 RoPE 还需要：

1. 角度生成 `φ = m * theta_i`（向量 ALU 或软件）
2. `sin/cos`（SFU）
3. 4 次乘加（向量 ALU）

### 推荐实现

- **增加 `SINCOS` 双输出微码**：一次范围缩减，同时产生 sin 和 cos，
  一个 vector op 出两个结果。RoPE 的 SFU 开销减半。
- 旋转的 4 次乘加交给向量 ALU，不要把 SFU 变成通用 ALU。
- 如果 RoPE 频率很高，再加一个融合 `ROPE` opcode：
  输入 `x0, x1, angle`，输出 `y0, y1`，复用同一套 FMA。

### 精度与范围问题

- 长上下文下 `φ = m * theta_i` 可达 ~3×10⁴ rad（32k context，base 10000）。
- FP32 Cody-Waite 在这么大幅度下约 0.002 rad 相位误差，可能影响 BF16/FP16 精度。
- 三种方案：
  1. **预计算 sin/cos 表/缓存**（推荐默认）：无范围缩减误差，但要占内存。
  2. **SFU 在线计算**：需要更高精度的范围缩减（3 段 Cody-Waite + 高位拆分，
     或 double-double 的 π/2 常量）。
  3. **融合 ROPE 微码**：适合 RoPE 是热点的场景。
- 建议：默认查表/缓存；`SINCOS` 作为动态位置或内存受限时的兜底，
  并在验证时专门测长上下文角度。

## 10. 与上一版（16 lanes）的差异

| 项 | 上一版 | 本版 D1 |
|---|---|---|
| lanes | 16 | 4 |
| 数据通路 | 2 条固定流水 | 1 套共享 FMA + 微码 |
| 内部精度 | BF16/FP16 多项式 + FP32 | 全 FP32（或 FP16 乘 + FP32 累加） |
| 吞吐 | 16 ops/cycle | 0.3–0.8 vector-op/cycle |
| 延迟 | 7 cycles | 8–16 cycles |
| 面积 | 大 | 约 1/3–1/2 |
| 适用 | 数据中心 GPU | 端侧 NPU |

## 11. 上层算子如何调用 SFU

SFU 是协处理器，不是主数据通路。以 4 lanes、每 vector op 处理 4 个元素为例：

| 算子 | SFU 原语 | SFU 调用次数 | 其余工作 |
|---|---|---|---|
| RMSNorm | RSQRT | **1 次 / group**（4 group 可拼一拍） | 平方、求和、均值、逐元素缩放 |
| LayerNorm | RSQRT | 1 次 / group | 均值/方差归约、缩放平移 |
| Softmax | EXP2 + RCP | N/4 次 EXP2 + 1 次 RCP | max、求和、缩放 |
| SiLU | EXP2 + RCP | 2 × N/4 | 乘 x |
| GELU（tanh） | TANH | N/4 | 乘加 |
| RoPE | SINCOS | pairs/4 | 每对 4 次 FMA |
| Attention scale | RSQRT | 1 次 / 常量 | 可预计算 |

关键结论：

- **RMSNorm 不是 SFU 密集算子**：每个归一化组只调用 1 次 RSQRT；主要开销是
  平方、归约、缩放，交给向量 ALU。
- **Softmax 和 GELU/SiLU 才是 SFU 密集算子**：按元素调用。
- 标量型调用（RSQRT/RCP）应把多个 head/token 拼到 4 个 lane，避免 lane 浪费。
- 不要把逐元素缩放塞进 SFU；否则 SFU 变成主 ALU，吞吐会崩。

### RMSNorm 数据流

```
x[N] ──► 平方 ──► 归约求和 ──► ×(1/N) ──► +eps ──► RSQRT（1 次）
                                                        │
x[N] ──► × rsqrt ──► × gamma ──► y[N] ◄─────────────────┘
```

- 4 个 token/head 同时归一化时，把 4 个 `rsqrt` 标量拼成一个 4-lane RSQRT。
- 只有一个 group 时，3 个 lane 空闲；这是正常的，RMSNorm 的瓶颈在归约，不在 SFU。

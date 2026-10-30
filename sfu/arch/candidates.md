# SFU 架构候选（端侧，共享资源）

> 上一版按 GPU/数据中心思路设计了 16 lanes 方案。根据端侧约束（16-bit I/O、FP32 内部、
> 算子不并发、面积优先）重新选型如下。

## 候选矩阵

| | **D1：共享 FMA 阵列（推荐）** | D2：双引擎 | D3：极简标量 |
|---|---|---|---|
| lanes | **4** | 4 | 2 |
| 运算资源 | **1 套 4-lane FP32 FMA** | 2 套（P 多项式 + N Newton） | 1 套 2-lane FMA |
| 函数共享 | **全部函数共享同一阵列** | 函数分组共享 | 全部共享 |
| 内部精度 | **FP32** | FP32 | FP32 |
| 控制 | 微码 FSM | 两条固定流水 | 微码 FSM |
| 每 vector op 周期 | 5–14 | 4–8 | 6–16 |
| 有效吞吐 | 0.3–0.8 op/cycle | 0.8–1.5 op/cycle | 0.15–0.4 op/cycle |
| 延迟 | 8–16 cycles | 7–10 cycles | 10–20 cycles |
| 相对面积 | **1.0（基准）** | ≈1.7 | ≈0.55 |
| 适用 | 1B–7B 端侧模型 | 7B+ / 延迟敏感 | MCU 级 NPU |

## 推荐：D1

理由：

1. **共享最彻底**：端侧同一时刻只跑一种非线性函数，没必要为 EXP/LOG/SIN/COS 和
   RCP/RSQRT 各做一套乘法器。一套 4-lane FP32 FMA 用微码串行执行所有函数，
   面积接近理论下限。
2. **4 lanes 匹配端侧向量宽度**：多数端侧 NPU 的 SIMD 是 64–128 bit，4×BF16 或 4×FP16
   正好一拍；再宽收益递减。
3. **FP32 内部满足精度要求**：16-bit I/O 量化误差 0.5 ULP，内部 FP32 保证 softmax /
   LayerNorm 不因中间舍入掉点。
4. **吞吐够用**：7B 模型每 token 约 100 万次非线性，4-lane 方案约 3.3 ms @600 MHz；
   若 token 总时间 20 ms，占比 ~16%。1B–3B 模型占比更低。

## 什么时候升级到 D2 / 8 lanes

满足任一条件时：

- 目标模型 ≥ 7B 且 decode 延迟是核心指标；
- profiling 显示 softmax / GELU 占 token 时间 > 20%；
- 需要同时跑 prefill 和 decode（prefill 的 softmax 是 N²）。

升级方式：保留 D1 的微码框架，把 FMA 阵列从 4 lanes 扩到 8 lanes，或增加第二套阵列。
RTL 参数化，不改控制逻辑。

**具体做法**：把 4-lane 数据通路封装成 `sfu_slice`，顶层用 `generate` 实例化 `LANES/4` 个。
**只复制数据通路（FMA/寄存器/LUT），共享微码 FSM、微码 ROM、系数 ROM、整数单元**。
面积约 1.7–1.9×，吞吐翻倍；不要实例化两套完整 SFU。

## D1 预算（写入 design_state.json）

| 参数 | 初值 | 依据 |
|---|---|---|
| `lanes` | 4 | 端侧 SIMD 宽度 |
| `fma_width` | FP32 × 4 | 内部精度 |
| `pipeline_stages` | 4（控制）+ 5–14（执行） | 微码串行 |
| `target_mhz` | 800 | 端侧典型 |
| `max_levels` | 30（暂定） | generic Yosys；需真实库校准 |
| `max_cells` | 50000（暂定） | 1 套 4-lane FP32 FMA + LUT + 控制 |
| `outlier_ratio` | 1.5 | 离群判据 |

> 这些是**待校准**的预算。RTL 骨架综合后必须用 `yosys_metrics.py` 更新。

## 关键设计决策

### D1：为什么用微码而不是固定流水

- 固定流水：每个函数一条专用路径，面积大。
- 微码：一套 FMA 阵列 + 一段微程序 ROM，按 opcode 跳转；面积小，函数可扩展。
- 代价：一次 vector op 需要 5–14 cycles，控制复杂度略高。
- 端侧算子不并发，这个代价可以接受。

### D2：为什么内部用 FP32，而不是 BF16

- Softmax 的 `exp` 动态范围大；BF16 只有 8-bit 尾数，累加和归一化容易掉精度。
- LayerNorm / RMSNorm 的方差计算对中间精度敏感。
- 端侧模型参数量小，精度损失更不可接受。
- 面积代价通过“一套共享 FMA + 4 lanes”抵消。

### D3：多项式乘法是否可以用 FP16 乘 + FP32 累加

- 可选面积模式：乘法器用 FP16/BF16，累加器用 FP32。
- 对多项式求值，输入在有限区间内，16-bit 乘法通常够用；误差需用穷举验证。
- 如果验证不达标，切回全 FP32。RTL 参数化 `MUL_PREC`。

### D4：TANH 是否需要专用硬件

- 第一版：`EXP2 → RCP` 两段微码，约 14 cycles。
- GELU 是 FFN 的热点；若 profiling 显示 TANH 占比高，再加一条融合微码路径，
  不需要新增乘法器。

### D5：SIN/COS 的范围

- 端侧 RoPE 通常预计算 sin/cos 表，SFU 只做兜底。
- 支持 `|x| ≤ 2^20` 的 Cody-Waite 范围缩减；超出报 invalid。
- 如果 RoPE 一定走 SFU，再评估 Payne-Hanek。

## 风险登记

| ID | 风险 | 概率 | 影响 | 分数 | 缓解 |
|---|---|---|---|---|---|
| R1 | 4 lanes 对 7B 模型吞吐不足 | 3 | 4 | 12 | 参数化到 8 lanes；先 profiling |
| R2 | FP32 FMA 面积超预算 | 3 | 4 | 12 | 提供 FP16 乘 + FP32 累加模式 |
| R3 | 微码控制复杂、验证量大 | 3 | 3 | 9 | 微码 ROM 用 Python 生成 + 穷举 |
| R4 | TANH 两段微码拖慢 GELU | 2 | 3 | 6 | 预留融合路径 |
| R5 | BF16/FP16 输出量化影响 softmax | 3 | 3 | 9 | 关键路径用 FP32 输出模式 |
| R6 | generic Yosys 级数与真实频率不匹配 | 4 | 3 | 12 | 等标准单元库；先用相对比较 |

## 验证策略

1. **BF16/FP16 穷举**：每个原语 65536 个位型，Python 双精度参考。
2. **FP32 随机**：每函数 ≥ 100 万向量 + 特殊值。
3. **ULP 统计**：max / p99 / 超阈值比例。
4. **吞吐/延迟**：4 lanes 连续向量，测量每函数 cycles 与 II。
5. **综合检查**：latch/loop/multi-driven = 0；depth 无离群。

## 下一步

1. Python 参考模型 + ULP 比较器（先验证精度预算）。
2. 微码 ROM 生成器（Python 输出每个函数的微指令序列）。
3. 参数化 RTL 骨架：先做 EXP2 的 4-lane 共享 FMA 版本，综合校准面积/级数。
4. 扩到 LOG2、RCP/RSQRT/SQRT、SIN/COS、TANH。

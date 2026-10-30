# NPU：LLM 推理加速器

平台无关的 NPU 外围 RTL，加上两套既有数据通路核（`tensorcore/` 矩阵核与
`sfu/` 非线性核）。目标单序列 **prefill + decode**。

**当前状态：外围 RTL 已实现并仿真通过，尚未综合、未做时序／板级实测，也未做整层或多 token 的精度验收。**
验证到“单个投影的硬件 GEMV 与真实网络的 fp32 前向比对”为止，见下文结果表。

## 结构

| 层 | 内容 |
|---|---|
| 外围 | 命令前端、调度、完成／故障、scratchpad、DMA、权重装载、结果回写、VPU、SFU 适配 |
| 矩阵核 | `tensorcore/`：`gemv_stream` → `gemv_cluster` → `gemv_unit` → `pe`，W4A16 / W16A16 四种 mode |
| 非线性核 | `sfu/`：`sfu_top` / `sfu_slice`，FP32 逐元素函数 |

**`tensorcore/` 从未修改**：外围只通过 `rtl/mpu/mpu_engine.v` 提供它需要的接口，
不重启核、不访问其内部状态机。`sfu/` 原本是占位骨架，其 FP32 数据通路已按
`sfu/arch/spec.md` 补全，端口与 `sfu_top` 的双 slice 结构保持不变。

模块划分、层级、命令／完成字格式、scratchpad 模型与配置寄存器见
[rtl/README.md](rtl/README.md) 与 [docs/npu_integration/](docs/npu_integration/README.md)。

## 目录

```
rtl/           外围 RTL（平台无关）
tensorcore/    既有矩阵核（只读）
sfu/           既有非线性核
tb/            iverilog testbench 与 Verilator 的 C++ 与包装
scripts/       回归、Verilator 构建、离线打包与真实模型参考
docs/          设计与验收文档
```

## 运行

```sh
scripts/regress.sh                                   # 全部 iverilog testbench
scripts/verilate.sh npu_top tb/vsim/npu_top_main.cpp # 构建快速模型
python scripts/real_gemv.py --proj q --layer 0 --cols 32 --token -1
```

`scripts/regress.sh` 在 MSYS2 mingw64 存在时自动前置工具链，因此同一脚本
在 Windows 的 Git Bash 与 POSIX shell 下都能用。协议类 testbench 用 iverilog
（依赖 Verilog 延迟）；长数值运行时用 Verilator。

需要 `downloads/qwen3-0.6b-awq`（AWQ 权重）才能跑 `real_gemv.py`；该目录不入库。

## 验证结果

| 项目 | 结果 |
|---|---|
| `scripts/regress.sh`（iverilog，17 个 testbench） | 17/17 通过 |
| 全 1 权重 K=1024（`tb_top_gemv`、`t_ones`） | 1024.0，与理想值一致 |
| 真实 Qwen3-0.6B AWQ 权重，七个投影（layer 0） | cos 0.953–0.999 |

七个投影中，RTL 输出与“用打包后权重做精确 fp32 矩阵乘”的结果打印位一致，
即残差全部来自 int4 + E8M0 再量化（占参考 RMS 的 5.5%–31%），不来自 RTL。
参考值是 `scripts/qwen3_ref.py` 的真实前向，不是硬件行为模型。逐项数据与
本轮修掉的两个缺陷记录在
[15 验收计划](docs/npu_integration/15_validation_and_sequences.md)。

## 已知边界

- 只验证了单投影单 token 的 GEMV，未做整层、多 token、KV 追加或端到端 logits 比对。
- 外围 RTL 未综合；无 STA、无 CDC 物理审查、无 ZCU102 板级带宽实测。
- `sfu/` 的 FP32 能力以 `tb_sfu_slice` 的 ULP 预算为准，未冻结为最终数值合同。
- Verilator 无法调度 `tensorcore/rtl/fp32_acc.v` 的写法，该文件在 Verilator
  构建中被 `tb/vsim/fp32_acc_sim.v` 替代（`tensorcore/` 本身不改）。细节见
  [rtl/README.md](rtl/README.md)。

## 文档

- [NPU 外围集成](docs/npu_integration/README.md)：协议、指令集、各模块合同、验收计划
- [RTL 说明](rtl/README.md)：模块划分、层级、时钟复位、寄存器与验证矩阵

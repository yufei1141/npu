# NPU：可扩展 LLM 加速器架构设计

> 新增：[MiniCPM-V-4.7-1B-GPTQ 纯文本 NPU spec 与规划](docs/minicpm_1b/README.md)，目标 ZCU102 裸金属、200MHz；含5种架构对照、递归状态引擎、完整模块合同及验证脚本。
>
> [查看新模型架构图](docs/minicpm_1b/diagrams/README.md) · [下载新模型 spec ZIP](https://github.com/yufei1141/npu/raw/refs/heads/main/downloads/minicpm_1b_npu_spec.zip)

基于 SpinalHDL 的 LLM 推理加速器设计，目标支持单序列 **prefill + decode**。

**当前交付是架构 spec、架构图和可执行建模脚本；尚未实现完整 RTL，也未完成 FPGA 综合、板级性能或模型精度验收。**

## 查看与下载

- [最新架构 spec 索引](docs/architecture/README.md)
- [全部架构图（GitHub 页面）](docs/architecture/diagrams/README.md)
- [下载架构图与 spec ZIP](https://github.com/yufei1141/npu/raw/refs/heads/main/downloads/npu_v1_architecture.zip)

ZIP 解压后，用浏览器打开 `npu_v1_architecture/diagrams/architecture.html`，即可离线查看全部9张图；SVG 可放大，DOT 源文件可编辑。HTML 在 GitHub 上显示源代码，需下载后打开。

## 结构

- **MPU**：128 tile × 8 产品位置；W4 专用共享激活产品路径，BF16 路径用于 attention 和 lm_head。
- **VPU**：32 物理 lane，MUL、ADD/compare、分段归约和 SFU。
- **DMA / MCU**：HBM 与片上权重、KV、激活的搬运、重排、打包和完成跟踪。
- **DCU / Acu**：宏指令、事件、资源依赖与在线 attention 调度。
- **存储**：权重、向量、结果、online state，以及两份 KV tile 缓冲；第三份 KV 缓冲是待评估候选。

![顶层架构](docs/architecture/diagrams/01_top.svg)

## 详细合同

| 文档 | 内容 |
|---|---|
| [协议与存储](docs/architecture/01_protocol_memory.md) | Bundle、握手、credit、bank 与 slot |
| [MPU](docs/architecture/02_mpu.md) | W4 产品、DOT8、group、scale、累加 |
| [VPU / SFU](docs/architecture/03_vpu.md) | 非线性、归约、Norm、RoPE、online attention |
| [DCU 调度](docs/architecture/04_dcu_schedule.md) | 宏指令、事件、prefill/decode、故障排空 |
| [架构模型与验证](docs/architecture/05_model_validation.md) | 有限缓冲模型、边界检查、验收门槛 |
| [DMA](docs/architecture/06_dma.md) | 读写引擎、队列、重排/pack、完成条件 |

## 运行架构检查

只需要 Python 3 标准库：

```bash
cd docs/architecture
python attention_schedule_model.py
python layout_validate.py
```

重新生成架构图需要 Python 3 与 Graphviz：

```bash
python docs/architecture/diagrams/render.py
```

这些脚本验证抽象调度、依赖和逻辑 bank 映射，不代替 RTL 仿真或实际器件资源/时序验证。

## 模型来源与待验证项

目标 checkpoint：[6block/Qwen3-8B-GPTQ](https://huggingface.co/6block/Qwen3-8B-GPTQ)。配置快照见 [docs/reference](docs/reference)，已经核对模型维度与量化声明；实际 compressed-tensors 权重打包、static act-order、scale dtype 和端到端误差尚待验证。

设计演进及先前瓶颈分析保留在 [docs/history](docs/history)，冲突时以最新 `docs/architecture` 实现合同为准。工具版本、浮点 IP 的延迟/II、物理 RAM 布局和 HBM 有效带宽尚未冻结。

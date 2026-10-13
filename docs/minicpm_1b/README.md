# MiniCPM-V-4.7-1B-GPTQ 纯文本 NPU：架构 v0.1

2026-10-13。目标：ZCU102、裸金属、200MHz、batch=1，支持纯文本prefill和decode；不加载/实现视觉编码器。未来视觉embedding入口预留，但当前不实现图片处理。此目录是新模型专用设计，不覆盖旧Qwen3-8B spec。

**交付状态：详细设计、对照架构、可执行模型与验证规划；没有完整SpinalHDL RTL、FPGA资源适配或实测token/s。**

## 设计结论

推荐 **输出并行的W4流式MPU + 可切换BF16/FP16 DOT通路 + FP32递归状态引擎 RPU + 小型VPU/SFU + 显式DMA与片上scratchpad**。不是缩小V80矩阵阵列，而是针对decode、原生输出词表和Gated DeltaNet分别组织数据流。

首版综合目标：MPU W4 64产品/拍、A16 32产品/拍；RPU 16lane FP32；VPU16lane、SFU4请求lane；可扩展32/64/128 W4配置。200MHz是目标，尚未完成时序。产品并行数不等于峰值TOPS达到或资源适配成功。

原生16bit lm_head在带宽上很昂贵；参考预算在6GB/s有效DDR带宽、4Kcontext及所选利用率假设下约 **6.1 token/s**。校准后额外量化lm_head的候选约9.6token/s，但改变了checkpoint，不能作为原模型性能。详细敏感性见02和自动生成结果。prefill独立报告TTFT，不以decode速度换算。

## 文档入口

| 文档 | 内容 |
|---|---|
| [01 模型与公开设计依据](01_model_and_sources.md) | 真正需要的算子、矩阵尺寸、RK3588/RKLLM、NVDLA、Gemmini参考边界 |
| [02 结构对照与性能模型](02_architecture_choices.md) | 5种候选、资源选择、decode/prefill预算及限制 |
| [03 MPU](03_mpu.md) | W4输出并行、DOT输出词表、分组与上下文、扩展数据流 |
| [04 RPU递归状态引擎](04_rpu.md) | FP32状态两遍处理、head缓存、事务提交及数值模型 |
| [05 VPU与SFU](05_vpu.md) | softplus/L2Norm、零中心Norm、短卷积、门控和partial RoPE |
| [06 DMA与存储](06_memory_dma.md) | PS/PL DDR角色、片上预算、bank、布局、backpressure |
| [07 宏指令与裸金属ABI](07_control_runtime.md) | descriptor格式、队列、CSR、cache维护、导出与恢复 |
| [08 调度与prefill](08_scheduling.md) | 两种layer流程、依赖、流水与head-major状态复用 |
| [09 验证与规划](09_validation_plan.md) | 数值/功能/资源门槛、分阶段交付与未决项 |
| [10 FP32多周期与存储端口](10_fp32_memory_timing.md) | RAM读写反馈、partial转置snapshot、RPU共享算术调度 |
| [架构图](diagrams/README.md) | 顶层、MPU、RPU、DMA/存储、layer调度 |

## 运行模型

```bash
cd docs/minicpm_1b
python models/design_space.py
python models/validate_recurrence.py
python models/validate_layout_and_buffers.py
python models/validate_fp32_feedback.py
```

分析模型用Python标准库；递归验证使用NumPy。输出结果属于本次运行，不能当作模型checkpoint精度验收。模型主参数取自保存的config及权重索引；MUL/ADD真实L/II与DDR有效带宽是待测输入。

## 三个首要门槛

1. 导出器必须核验GPTQ打包、实际16bit dtype与输出head绑定。source的tied-key声明与config/权重索引需在真实loader上确认。
2. 延迟scale、分组FP32累加和SFU近似必须通过端到端数值验收；不能把数学等价当成浮点位等价。
3. PS→PL有效读写带宽、RAM资源和FP32反馈延迟必须测量。预算中的4/6/8GB/s均是假设，不是ZCU102实测。

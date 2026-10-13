# SpinalHDL v1 实现合同索引

版本 v0.5，2026-10-13。适用范围：6block/Qwen3-8B-GPTQ 的单序列prefill与decode。本文档组优先于v0.4中存在冲突的通道宽度、256物理VPU、未明确归约结构等描述。当前交付是细化spec与可执行架构模型，没有宣称已完成RTL或SpinalHDL工具链。

## 实现基线

- MPU：128tile×8产品位置。W4专用共享激活路径；BF16通用乘法路径支持attention和lm_head。保持1024产品/拍目标，不扩MPU规模。
- VPU：32物理lane，独立FP32 MUL、ADD/compare、segment reducer；32请求lane共享exp/reciprocal/rsqrt资源。256元素只作为逻辑传输块。
- group处理：每tile保留16个DOT8结果，按8行/拍进入**64/32/16/8加法器四级group树**，8lane scale与total ADD。明确可实现的首版；16lane是后续参数候选，不当成已接受配置。
- KV：默认2slot、每slot128token K/V，总1MiB。首版融合QK max与PV状态写；跨tile lookahead/3slot通过模型比较，尚不冻结。
- 线性层输入/输出BF16，产品/group/total FP32；延迟scale明确为候选数值语义。尚未通过模型精度验收。
- 命令/事务统一Stream；固定延迟无ready算术IP用预留credit包裹；64bit地址、16bitcmdTag与16bitepoch。
- 宏指令采用v0.4 512bit ABI，支持prefill二维tensor。宿主配置前必须校验checkpoint的static act-order、实际scale dtype和compressed-tensors tensor布局。

## 文档

1. `01_protocol_memory.md`：Bundle字段、valid/ready、bank/slot/credit、KV读视图。
2. `02_mpu.md`：产品、DOT8、group树、scale/total、行映射与各FSM。
3. `03_vpu.md`：VPU资源、SFU、归约、Norm/SwiGLU/RoPE与在线attention。
4. `04_dcu_schedule.md`：顶层调度、事件、prefill/decode程序、故障/排空。
5. `05_model_validation.md`：有限KV slot的attention模型、覆盖范围与实现验收。
6. `06_dma.md`：DMA读写引擎、重排/打包、搬运、credit及完成条件。
7. `architecture_params.json`：可机器读取的冻结参数和待验证IP参数。
8. `attention_schedule_model.py`：模型脚本；运行生成`attention_schedule_results.json`。

## 实现顺序

先实现protocol/config、credit wrapper与一tile数值验证；随后group reducer与scale/total；再实现32lane VPU、小规模online attention和调度。单元latency/II、RAM端口与数值测试通过后扩完整128tile。board HBM shell、工具版本必须在工程创建时锁定。

不把group树共享归约、actor反压、head归约、标量操作或KV转置作为“之后自动解决”的问题。本组spec明确这些边界；实际Fmax、资源和吞吐仍需仿真、综合、板测。

## 顶层模块命名

`NpuTop = DCU + MPU + VPU + Acu + DMA + Stores + BoardShell`。DMA是显式子系统，可归于MCU访存控制层，职责是HBM↔片上搬运及受限LOCAL_COPY；host PCIe DMA由shell单独提供。所有DMA完成条件见06_dma.md，不以已发请求或开放端口代替搬运完成。

## 架构图

见 [架构图总览](diagrams/architecture.html)，包含顶层及MPU、W4产品、VPU、SFU、DMA、KV、DCU、online attention共9张图，附SVG/DOT源文件。

# 05 架构模型与实施验证门槛

## 1. 本次执行的模型

运行：

```
cd docs/architecture
python attention_schedule_model.py
python layout_validate.py
```

无第三方依赖，使用Python标准库。脚本写同目录JSON，保留数值参数和首24个事件，供重复核查。不是RTL测试，不需要Spinal/FPGA工具安装即可执行。

### Attention有限slot模型

显式建模：单MPU、单VPU阶段调度器、串行DMA填充；2或3KVslot；至多2score/p在飞；QK/PV共享MPU；next tile probability必须等待prev state commit；PV结束释放KV，而merge结束才提交state。每发QK先保留score空间。所有长度案例验证state顺序、PV依赖、slot保护及最多2score/p。

示例QK512拍、PV512拍、PREP272拍、MERGE136拍；PREP是exp/scale及固定drain的简化阶段，MERGE为状态合并；max视作融合，未包含实际MUL占用、score subtract、head归约、alpha/l标量、FP IP延迟/特殊值、AXI乱序、真实bank冲突。与实现spec的所有操作并非一一对应，不能将其视作完整吞吐预测。

模型tail仍按完整128token计算QK/PV，S1/127/129案例验证控制边界而非有效tail性能。DMA duration=40+ceil(validTokens*4096/BW)；一次只填一个slot，没有并发burst，因此高估某些DMA填充固定开销。

## 2. 4K attention实验结果

| 有效bytes/拍 | 2slot串行 cycles | 2slot lookahead cycles | 3slot lookahead cycles |
|---:|---:|---:|---:|
| 512 | 46,889 | 42,673 | 36,129 |
| 716 | 46,598 | 44,382 | 34,347 |
| 896 | 46,451 | 41,883 | 34,053 |

BW716情况下MPU利用率约70.3%→73.8%→95.4%。这是该阶段模型的利用率，不是全模型、FPGA或20tok/s实测。3slot额外512KiB仅在bank独立且接口能执行对应调度时有意义。

2slot贪心lookahead的716结果反而差于512：更快KV到达会改变“MPU空闲时优先PV否则立即QK”的选取时刻，使QK占用期间错过即将ready的PV。这是调度策略敏感性，不能把模型结果硬拟合成单调带宽曲线。后续比较有界wait/ready时间估计或单纯serial，优化必须证明不影响正确性。

boundary cases：1、127、128、129、4096token；均完成，无依赖/有限空间断言失败。模型不证明复杂RTL不存在死锁，也不覆盖prefill全部loop。

## 3. bank映射验证结果

layout_validate.py检查每份K/V各131,072个BF16元素具有唯一(bank,row,element)地址；decode QK、prefill QK、decode/prefill PV的每microstep同logical bank只要求一个row读取。验证基于新版PV 4qh×32dim映射及K32position宽word。

groupStore双slot逻辑容量16KiB、group tree120ADD位置算术检查通过。

这是logical word contract，不证明physicalRAM端口、bytewrite、pack吞吐、器件块碎片或布线。需要仿真RAM请求trace、综合RAM映射及板级验证。

## 4. 未冻结参数与收敛流程

- 浮点IP：实际MUL/ADD/SFU L、II、FTZ/NaN/subnormal支持、FMA与否、资源。配置JSON中null表示未测，simulation_defaults_only不能用于宣称已验证。
- W4 checkpoint：实际scale dtype、static act-order语义、compressed-tensors header/weight packing；对导出后q/scale逐元素抽样比较。
- SFU：分段/阶数/coeff/ROM端口及误差门限。稳定sigmoid实现3MUL/元素已进入合同。
- GroupStore和KV bank：logical→physical映射、pack buffer 80KiB、write throughput、完整128tile综合面积。
- 2/3KVslot与8/16scale lane：2/8作为可实现功能baseline，优化参数需要资源+cycle验证后选择。
- board shell、HBM可持续效率、Spinal/Scala/sbt/JDK工具版本：创建工程及板级集成时锁定。

## 5. SpinalHDL首个可验收里程碑

不直接从全NPU跑Qwen开始。第一个工程交付应包含NpuConfig/Bundle、CreditFpAdapter、W4ProductLane/Cluster、BF16ProductWrapper、Dot8Tree、单tile GroupCollector及reference。所有值都带tag/mask与随机反压；测试检查数值、完成计数及未消费返回空间。

第二里程碑实现128行的8row/拍group reader/reducer、8bank total、prefillTotal；第三实现VPU SFU prototype、segment reducer、SwiGLU/Norm；第四连接小规模Acu+KV两slot与DCU，从仿真事件trace对照模型，发现缺失阶段后更新模型而非修改测试期望掩盖stall。

对实现测试要求：可执行测试数非零、固定seed、超时失败、边界/随机覆盖、仿真vs综合vs板测分开报告。验证命令保留原退出状态。onboarding通过的open_tc FP22/FP8测试不能代替本FP32/BF16算术测试。

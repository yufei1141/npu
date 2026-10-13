# Qwen3-8B 结构与瓶颈复核

> 最新实现合同见 `../architecture/`。稳定sigmoid负支增加一次MUL，当前固定3MUL/元素；本文§3早期2MUL算术表属于历史估计。最新脚本已更新。有限slot调度实验见`../architecture/05_model_validation.md`。

2026-10-13；基线：1024乘法位置，200MHz，32物理VPU lane，32 SFU lane，14个逻辑512-bit HBM读口。范围固定为选定Qwen3-8B checkpoint所需的prefill/decode，不扩展通用算子。这里是静态资源、阶段及依赖分析，尚非RTL/板级性能验证。

## 1. 已核验的checkpoint配置

来源：https://huggingface.co/6block/Qwen3-8B-GPTQ/resolve/main/config.json ，本次已成功读取，原始副本为checkpoint_config.json。H4096、F12288、36layers、Nq32/Nkv8、D128、vocab151936、SiLU、RMSNorm eps=1e-6、rope_theta=1e6、default RoPE、full attention、无attention bias、无sliding window、embedding/lm_head不共享；lm_head在量化ignore列表。

量化为compressed-tensors/pack-quantized，4bit、group128、symmetric、actorder=static；并非从名称可直接推定的传统GPTQ qweight/qzeros文件ABI。scale_dtype=null不证明其为FP16；仍需检查实际tensor header及导出器语义。本次未下载多GB权重、未加载框架进行推理、未验证act-order排列。参数max_position_embeddings=40960不代表硬件HBM足够容纳该长度。

现有算子集合覆盖所声明的计算类型；精度、权重解释与prefill/decode完整正确性尚待实现验证。尤其不能将存在static act-order的checkpoint按“无g_idx无需置换”处理。

## 2. 结论和结构调整

1. **SwiGLU的平均吞吐不构成当前FFN主要瓶颈。** 32lane足够作为首版规模，但不能把其周期从性能表删掉。
2. **Attention是VPU最值得优先优化的地方。** QK/PV较快，max→exp→状态缩放→PV→状态合并的依赖会使MPU闲置；decode和prefill都存在这一问题。
3. **8lane group scale是满负荷单元。** 有任何持续吞吐损失就反压MPU，不能只优化乘法器。
4. **结果burst和共享bank可能让平均吞吐充足的VPU仍然卡住。** 应采用独立写入缓冲、独立attention状态bank与有credit的发射。
5. **HBM有效带宽与BF16 lm_head是更大系统风险。** 10口喂W4没有很多余量；80%效率下已不足，需按阶段重新分配全部14口。

建议优先调整：W4专用产品路径（保持1024产品/拍）；独立32lane MUL/ADD执行资源；32lane SFU（首版shared exp/recip/rsqrt，按实际II计数）；独立segment reducer；scale目标由8提高到16lane候选；attention状态与score/p独立bank；MPU结果至少双128元素缓冲；attention按两种调度方案进行有限缓冲cycle比较后决定KV slot数量。

## 3. FFN：非线性会不会追不上矩阵

### 3.1 相同资源计数才能比较

每元素SwiGLU：exp一次、reciprocal一次、ADD一次、MUL两次；如SFU内部有独立exp/recip子流水则另算。这里保守按exp/recip共享同一32lane请求资源，独立32lane MUL与ADD，各II=1。资源工作量不包括特殊值/范围规约额外操作与流水排空。

| 目标 | decode M1 | prefill M32 |
|---|---:|---:|
| gate/up MPU理想issue | 98,304拍 | 3,145,728拍 |
| down MPU理想issue | 49,152拍 | 1,572,864拍 |
| SFU工作量 | 768请求拍 | 24,576请求拍 |
| MUL工作量 | 768请求拍 | 24,576请求拍 |
| ADD工作量 | 384请求拍 | 12,288请求拍 |
| 连续任务资源负载下界 | 768拍=3.84µs | 24,576拍=122.88µs |
| 整算子串行阶段发射和 | 1,920拍=9.60µs | 61,440拍=307.20µs |

资源下界要求分块流水并保持各单元饱和，不是单指令完成时间保证；串行阶段发射和也不包含FP latency/访存。对比down，资源负载仅1.56%，整算子串行发射约3.91%。即使不重叠，也不是当前FFN的主要成本。

原描述将sigmoid当一次32lane SFU请求，384拍就完成SwiGLU，是有条件的：若sigmoid由共享exp与reciprocal多次经过同一SFU，至少768请求拍。不能把对外opcode数量当内部工作量。

### 3.2 但不能直接让down边消费边启动

现有W4 GEMV按输出行batch、group顺序读取，down的第一个输出tile需要访问全部F输入。只产生前128个activated元素并不意味着可以沿用现有循环开始完整down。

首版完整生成activated buffer后再down，正确性最简单，成本可接受。之后才考虑group-major down：保留全部4096输出的FP32部分和并改变weight预取顺序，在activated group ready后更新全部输出。prefill则需M32×4096 FP32总和=512KiB；改动显著，不应为隐藏几µs先做复杂全局调度。

gate/up可在离线打包中把相同输出tile配对，输出写各自buffer，成对ready后启动SwiGLU。但并行VPU处理必须使用与MPU写不同bank组；仅有不同逻辑地址不保证无端口冲突。

## 4. Attention：真实的非线性关键路径

### 4.1 decode每128token tile的阶段

32Qheads×128score=4096元素，online o为32×128=4096元素。

| 阶段 | 理想发射拍（32lane） | 依赖 |
|---|---:|---|
| QK MPU | 512 | KV、Q ready |
| head max扫描 | 128 | score可见，可与QK结果写入融合 |
| exp + 概率归约 | 128 | 每head当前m_new确定 |
| 旧o×alpha | 128 | 当前alpha和上一tile的o已提交 |
| PV MPU | 512 | p ready、V ready |
| contribution + scaled o | 128 | PV结果和旧o缩放完成 |

采用“每个宏阶段完整完成再启动下一个”时，发射和=1536拍/tile，较纯QK+PV的1024拍多50%。4K、36layer共有1152tile，额外512×1152/200MHz≈2.949ms。忽略首tilealpha=0快速路径会略高估工作量；忽略alpha exp、FP32标量l更新、树反馈/排空/最终除法会低估完成时间。因此不是完整预测或严格下界。

前版47.916ms理论预算剩2.084ms，加入这一阶段调度的估算就到约50.865ms，约19.66tok/s，尚未计其他开销。说明不能先声明20tok/s达标，再把softmax按0周期处理。

### 4.2 32lane是否太少

连续供给时MPU每512拍产生4096个QK score，平均8score/拍；32lane exp有4倍平均吞吐余量。问题主要是阶段依赖与资源复用，而非必须堆256 SFU。

改为64lane将上述独立vector扫描从512降为256拍，串行tile约1280拍，多25%；16lane则vector1024拍，串行tile2048拍，多100%。但先融合max/调度重叠，可能比直接扩lane更划算。

### 4.3 建议两阶段优化

**第一阶段：同tile内重叠，保持两KV slot。** MPU score输出时同步更新per-head max，消除额外全tile扫描；概率生成同时归约l；独立MUL管线处理旧o缩放。PV输出可直接与scaled o合并写状态，避免末尾再遍历完整o。

这需要输出带head/dim标记、segment reducer独立、scaled o bank独立和足够pipeline credit。PV输出仍是突发，FP32 ADD的II/feedback和写带宽需满足结果接收要求。具体收益由最后head到达、归约排空和max可见时刻决定，不直接写成“完全隐藏”。

**第二阶段：跨tile调度。** 在VPU处理tile0时，MPU先算tile1的QK，之后算tile0的PV；允许提前计算下一tile的score/max。下一tile的alpha及o更新必须等待上一tile的o提交，不能把state update乱序。

MPU候选序列 `QK0 → QK1 → PV0 → QK2 → PV1 → ...`。状态的m/l/o提交tag按tile递增；score buffer可提前ready，状态更新仍有依赖。

当前两KV slot可能同时被tile0等待PV、tile1等待消费占用，此时tile2无slot预取，PV0结束才释放slot0，QK2就会等HBM。**若此调度确有收益，考虑第三个512KiB KV slot（总1.5MiB），或分别管理K/V的生命周期后证明两slot可行。** 仅加score双缓冲不能解决被占用的KV slot。

prefetch2必须有物理bank独立于compute0/1；增加容量但共享单write端口同样会stall。两slot串行方案与三slot重叠方案均需cycle比较，三slot是候选，尚未冻结。

### 4.4 prefill同样存在这一问题

Bq32、Bk128、单qh：4096score，QK512issue、PV512issue，32lane max/exp/scale/combine各128拍，比例同decode聚合tile。因果边界的有效MPU计算少时，状态遍历相对成本更大。

为共享GQA的4qh KV，采用queryBlock→kvHead→keyTile→qh，保存4qh×32query×128dim FP32 o=64KiB。score/p按qh串行复用。若扩大同时在飞head数，必须相应增加在线状态、tag和credit，不能只改调度器队列深度。

长prefill的norm/SwiGLU算力比Linear仍有余量，但hidden/FFN spill、转置和scratch流量可能成为主瓶颈；短prefill、decode短上下文则FP latency、launch/drain、归约更显著。4K结论不能外推到所有长度。

## 5. MPU出口与VPU入口：平均吞吐足够不代表不会卡

128tile可能同时退休128 FP32结果。32lane接口需要4拍消化，producer每16issue拍形成一个QK micro-result batch，平均8元素/拍可接受。

但若MPU要求128结果同拍被接受、VPU仅接32，则一定反压或丢数。建议独立ResultStore，至少两个128元素FP32 bank（共1KiB），先并行捕获再4拍串化；实际容量还需覆盖IP不可撤销在飞返回。

若VPU连续blocked B拍，平均arrival8元素/拍，保守需求 `burst128 + ceil(B/16)*128 + outstandingReturn`；必须根据返回曲线/最坏在飞量计算。bank已满时阻止后续MPU发射，不能寄希望于未来ready。下游HBM/lm_head argmax路径与attention状态写共用时，应由公平仲裁保证有界阻塞。

prefill Linear结果可能在较长计算后突发M32×N128=4096元素；32lane串化128拍。完整输出tile计算K4096至少16384issue拍，平均很低，但结果buffer仍需容纳批量或具备可靠分批退休控制。

## 6. scale与累加：当前最紧的计算配套

group128：128行group sum/16issue拍=8个group结果/拍。8lane scale MUL、8lane Total ADD在理想II1时100%利用率，任何bubble无恢复空间。W4专用乘法再扩并行后此处会先堵。

建议把16lane scale/total更新作为综合比较点，GroupStore至少双缓冲，每buffer128 FP32。TotalStore按16bank映射且真正支持16更新/拍才有收益；仅增加乘法器不增加RAM端口没有用。decode同输出行每group更新间隔16拍，Ltotal必须≤16或有正确RAW forwarding。

DOT8归约group16仍是潜在资源/吞吐成本：每16拍每tile需要归约16个FP32 DOT结果。不能仅宣布“4级树”而不定义有多少树/如何时分。时分方案必须证明读端口、每group归约工作量和下一group覆盖前的完成期限。

## 7. 32lane VPU：需明确资源并行而非只有lane数

建议32lane独立MUL、32lane独立ADD/compare、独立SegmentReducer、32lane SFU；作为综合目标，FP IP是否能以所需II完成待测。不采用所有阶段竞争一个32lane“万能计算口”。

segment reducer在D128 head norm每4输入拍产生一个head partial；全局sum用交错partial context避免浮点feedback hazard。mean/eps/rsqrt、l更新、最终reciprocal等标量操作使用小ScalarUnit或保留SFU请求credit，不能造成长期vector占满时标量无法发出。

AttentionStateStore、Norm scalar、FFN、QKV和MPU出口至少分离物理bank组；每拍列出读/写访问量：
- SwiGLU：gate/up各32 BF16读，activated32 BF16写；旁路FIFO保留gate。
- o scale：32 FP32旧o读、32 FP32 scaled-o写。
- PV combine：32 FP32 contribution、32 FP32 scaled-o读、32 FP32 o写。
- RMSNorm第二遍：x/gamma各32 BF16读，32 BF16写。

同一个1R1W bank组不能凭不同base支持同拍双读。bank conflict时phase复用会降低II，需要计入模型。RoPE配对fetch和QK/PV不同KV视图也是端口瓶颈，不只是容量问题。

## 8. HBM敏感性

W4 Linear含scale需要528B/issue拍=105.6GB/s；10口理想128GB/s，效率须≥82.5%才可全速。80%有效效率需要至少11个独立口；70%需要12口。attention阶段要求约102.4GB/s KV有效带宽，8口完全没有余量，按效率重新分配。

| 假设有效效率 | W4所需口数 | BF16 lm_head，12口时间 |
|---|---:|---:|
| 100% | 9 | 8.103ms |
| 90% | 10 | 9.004ms |
| 80% | 11 | 10.129ms |
| 70% | 12 | 11.576ms |

口数只在每口独立物理带宽成立时有效。14口共享某个更窄NoC/HBM控制器的情形须受总物理上限约束。若权重仍只分配10口，80%时36layer Linear由33.915ms变为至少34.975ms；不要将此stall归类为非线性瓶颈。

BF16 lm_head通常是系统最大的单个带宽段；argmax输入平均每拍约896/2=448 BF16 weight对应的产品而非448 logits。每logit需4096权重，平均结果仅约0.1logit/拍，32lane argmax完全充足。需防的是输出行batch退休burst，不是平均argmax算力。

## 9. 后续必须跑的有限资源cycle模型

本次脚本npu_bottleneck_model.py是资源/阶段算术，可重复生成JSON；**它不是以下cycle仿真，也没有证明无死锁**。

最小cycle模型须含：真实L/II、每单元queue/FIFO credit、128元素返回burst、按bank的1R1W、AXI延迟/乱序/效率、2/3 KV slot、online状态依赖、score/p生命周期。至少比较：

- 串行baseline32lane。
- 32lane：max融合＋PV出口合并＋独立资源。
- 32lane：跨tile QK/PV调度，2slot与3slot。
- 64lane VPU及8/16lane scale。
- HBM效率70/80/90%，短上下文、4K、长上下文，prefill S1/32/128/4096。

输出：MPU idle按原因分类；VPU/SFU每资源busy；FIFO占用高水位；每bank冲突；HBM bytes及wait；每宏指令issue/start/lastReturn/commit。主要优化指标是MPU被VPU依赖阻塞的周期数，而不是单看SFU利用率。

按以上结构，优先解决attention依赖/缓存/bank和scale余量，再考虑增加VPU lane。矩阵MPU如果将来翻倍，必须同步重算scale、KV读取、结果出口和state更新，而不是只更新TOPS。

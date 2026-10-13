# 02 候选结构、参数与条件化性能

## 1. 比较五种组织

| 方案 | 结构 | 优点 | 瓶颈/放行条件 |
|---|---|---|---|
| A：统一小型计算核 | W4/A16各32产品，FP32状态通路8lane，普通VPU与state时分 | 复用多、适合先跑通 | 状态迭代、非线性与矩阵争资源；单核queue不能掩盖依赖 |
| **B：推荐异构流式** | W4输出并行64，A16 DOT32，RPU16，VPU16/SFU4，独立DMA | decode匹配有效DDR、原生词表连续读取、状态FP32不降精度 | 更多模块；资源与FIFO端口需综合；raw lm_head带宽仍占主导 |
| C：大systolic/mesh | 256 W4 /128 A16位置，DMA+独立RPU | 多token prefill权重复用潜力 | batch1空转、阵列fill/drain、DDR限制，ZCU102资源适配未证明 |
| D：B+输出head量化 | B结构，lm_head另做W4 | 显著减少每token原生词表流量 | 更改checkpoint；必须校准/端到端验收，shared embedding要保持原始lookup语义 |
| E：W4A8实验 | INT4×INT8整数MAC，FP32状态/非线性保留，headW4 | DSP48整数路径可节省浮点矩阵成本 | 新增activation量化；每group量化误差及head量化必须验收，不能声称原GPTQ直接兼容 |

方案B针对三种工作选择不同dataflow：W4 Linear输出并行与交错partial；native16 lm_head连续row-major DOT；GDN按FP32 state head-major缓存两遍更新。不是把同一DOT8阵列缩为64个位置。

C的利用率45%、A70%、B/D/E80%只是比较输入，未实测，不能从表推导C一定较差。没有综合资源与真实scheduler数据前，不把某架构排名当benchmark。模型没有定价FP32加法器LUT成本，也没有证明C能放进ZU9EG。

## 2. 推荐配置与扩展

| 参数 | ZCU102目标配置 | 扩展方式/约束 |
|---|---|---|
| W4产品 | 64/拍 | 32、64、128；同步扩weight供给、FP32 partial及bank |
| A16产品 | 32/拍 | 16/32/64，可配置DOT宽度；原生headDDR及row credit约束 |
| FP32 partial C | 16初始 | C>=ADD feedback latency，按IP/scoreboard证明 |
| group | 128 | 32/64/128 importer能力显式，不自动改checkpoint |
| RPU | 16 FP32 MUL/ADD lane | 8/16/32；state bank与DMA同时匹配 |
| VPU | 16 MUL/ADD lane，SFU4 | 8/16/32，标量和归约需有进展 |
| KV tile | 32token，2slot | 32/64，按D/head容量与bank验证 |
| state缓存 | 3×64KiB | 三role独立，head尺寸可随Dk/Dv配置，容量不足显式拒绝 |
| prefill | M16，4query×16row W4映射 | M4/8/16，按scratchpad可用性选择 |
| 本地工作存储 | 目标≤1.5MiB logical | RAM实际块碎片、端口、寄存另算，资源fit待综合 |
| context | 默认4096 | 1K/4K/8K/16K，内存与时延可行才声明支持 |

200MHz只有频率目标。对于PS→PL，4个128bit读取通路@200MHz理论12.8GB/s是总线上限示例，不是实测DDR速度或证明有4个独立物理带宽来源。实际可用端口/时钟/仲裁由BSP及board profile给出。

## 3. 用实际张量而不是1B标签算流量

W4矩阵497,025,024参数，含scale/serialized zero约258.22MB/token；输出head native16约508.10MB/token；FP32 GDN状态读+写37.75MB/token；full KV4K约50.33MB/token，另外small参数约数MB。视觉权重不加载，不计每token。

payload-only、canonical signed4、offline padded布局的真实byte count可能不同；容量由导出manifest计算，不按文件名估计。group size/scales/native head格式改变后必须重跑模型。

## 4. 参考6GB/s、4K context的模型结果

models/design_space.py按每projection/FFN分别取max(compute,memory)，再顺序加入RPU、attention、输出head及misc预算。不是整模型无限重叠roofline。推荐B的利用率输入矩阵80%、state80%；misc占位3ms+small参数读取。

| 方案 | 模型时间 | 模型decode |
|---|---:|---:|
| A | 239.61ms | 4.17tok/s |
| **B** | **163.62ms** | **6.11tok/s** |
| C | 147.20ms | 6.79tok/s |
| D，新增head量化 | 103.75ms | 9.64tok/s |
| E，新增W4A8+head量化 | 95.44ms | 10.48tok/s |

上述性能是该参数组合的分析结果，不是资源fit或数值验收后的承诺。特别是D/E必须先通过精度门槛；不能拿D/E速度宣传原生GPTQ模型。

B@6GB/s/4K：body约48.5ms，head约84.7ms，state约7.4ms，attention约19.7ms，加misc约3ms；输出head是最大的独立时间段。RPU若把FP32状态降到BF16，或代数重排降低MUL，不作为当前基线，必须重验精度。

## 5. Prefill不隐藏在decode数字里

推荐B的M16权重复用减少外存traffic，但并没有增加每秒产品数。源语义recurrent prefill对每token仍有state算术，按head缓存可减少DDR state搬运。4K纯文本prefill，采用同一利用率假设时core计算约269秒（约15prompt tok/s），尚不含scratch、Norm/conv、最后head等。

128prompt约7.2秒、512prompt约29.3秒同样只是core预算。它揭示了本FPGA配置的prefill成本，不应许诺即时4K首token。部署阶段可先短prompt，或额外主机precompute KV/state导入作为明确可选运行模式；导入不代替v1完整prefill能力。

C mesh、多核或W4A8可改善prefill，但模型算术、DDR和资源需要一起验证。优化TTFT与优化decode不同，采用独立性能门槛。

## 6. 优化顺序

先导出/数值合同和裸金属带宽测试→W4输出并行+native head DOT保持连续读取→RPU片上head两遍处理避免S临时写DDR→DMA queue/credit→融合VPU出口→只在实测瓶颈指向相应单元时扩规模。

不预先声称8/10/15tok/s，参考目标先6tok/s@4K（effective6GB/s假设）；若DDR只有4GB/s或FP32反馈导致bubble，模型自动下降。所有布局、流水和资源必须通过09门槛。

## 7. 推荐B的带宽/上下文敏感性

| 有效GB/s | context | 总预算ms | token/s |
|---:|---:|---:|---:|
| 4 | 1024 | 209.49 | 4.77 |
| 4 | 4096 | 224.23 | 4.46 |
| 4 | 8192 | 243.89 | 4.10 |
| 6 | 1024 | 148.88 | 6.72 |
| 6 | 4096 | 163.62 | 6.11 |
| 6 | 8192 | 183.28 | 5.46 |
| 8 | 1024 | 127.61 | 7.84 |
| 8 | 4096 | 142.36 | 7.02 |
| 8 | 8192 | 162.02 | 6.17 |

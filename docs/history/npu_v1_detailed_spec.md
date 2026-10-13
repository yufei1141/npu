# npu_v1 详细架构规范（SpinalHDL 实现基线）

> 最新实现合同：见 `../architecture/README.md`（v0.5），优先于本文早期模块规模、归约、VPU周期与KV计算视图描述。本文保留作为设计演进记录。

版本：v0.4-draft，2026-10-13。输入：用户提供的 `npu_v1_architecture.md` v0.3；参考实现：`/workspace/open_tc` 的 TensorCore CModel 与控制面组织。

**状态：可用于模块拆分、接口实现和验证模型开发；尚未经过 RTL 综合、时序收敛或目标模型端到端精度验证。** 本文中的“必须”是建议的 v1 实现合同，需评审冻结；初稿中的资源、吞吐与精度断言不自动成为已验证事实。

## 1. 范围与架构决定

### 1.1 默认范围

用户已确认：v1必须同时支持prefill与decode。实现单序列、batch=1、W4 group-128 Linear、BF16 KV、BF16 embedding/lm_head、greedy argmax；必须从prompt token序列完成全部层prefill、生成KV，再无缝进入逐token decode。decode可先实现用于调试，但prefill是v1放行的必选项。详细prefill合同见§14。

平台目标为 V80 类 Versal、200 MHz、1024 个 BF16 乘法位置；实际器件料号、HBM 容量、伪通道与用户 AXI 宽度、DSP58/IP 配置待核验。14 个 512-bit 读适配器是**逻辑目标**，不等于已有 14 个物理独立读通道。

运行时模型参数：H≤4096、F≤19200、层数≤48、D≤256；Q 头数、KV 头数和 H 必须满足 H=Nq·D、Nq mod Nkv=0；已由config.json确认的目标配置 Nq=32、Nkv=8、D=128、F=12288、层数=36、V=151936。checkpoint由用户指定为 https://huggingface.co/6block/Qwen3-8B-GPTQ/tree/main 。更新：本次复核已成功读取config.json，确认H4096/F12288/36layers/Nq32/Nkv8/D128/vocab151936；量化实际为compressed-tensors pack-quantized、W4 group128 symmetric、static act-order，scale dtype与具体tensor打包仍待检查。尺寸来自模型 manifest，不能仅按模型名称推断。上下文上限受实际内存预算约束，不保证所有最大参数能同时使用。

### 1.2 建议冻结的决定

| 项目 | v1 合同 | 原因 |
|---|---|---|
| 指令 | 512-bit 算子描述符 + 64B tensor descriptor | 字段足够表达形状、映射和事件，避免隐含地址约定 |
| 调度 | DCU 顺序取指，按依赖向单元队列分发 | 先建立可验证的 happens-before 关系 |
| MPU | 128 tile × 8 乘法位置，DOT8 + 分组部分和 | 避免默认假定 FP32 单累加器每拍反馈 |
| 数值 | 延迟 scale 为独立命名的 `W4_DELAYED` 模式 | 不声称与逐权重 BF16 反量化位等价 |
| KV | 物理 token-major、K/V 分区，8 KV 头共享给 32 Q 头 | 明确 GQA 的存储复用 |
| Attention | 128-token tile，FP32 online 状态 m/l/o | 无需保存完整上下文 logits 和概率 |
| UVC | 256 个向量位置，32 SFU，分段 head reduction | 全局归约不能混合不同 head |
| 存储 | 独立 Operand Store、Vector Store、KV Tile Store | 宽度与读写端口可实现，避免单一大 crossbar |
| 预处理 | 离线权重重排与格式规范化 | 运行时不做全矩阵转置 |

## 2. 初稿必须修正的事项

1. 延迟 scale 与 `BF16(q·s)` 再乘激活有不同舍入位置。不存在一般性的“输出偏差 <1 ulp”或“总是更精确”保证，抵消时相对误差可被放大。须通过逐层及端到端测试决定是否接受。
2. GPTQ 的 packed q/zero 编码不一定是有符号二补码、zp=0。`sym=True` 也不能直接证明文件中零点全零。导出器必须读取 checkpoint 的量化 metadata、qzeros 语义、g_idx/act-order 和打包次序。
3. 原 KV 公式 token stride=2048B、V offset=1024B 不适用于 BF16 K+V=4096B，存在地址重叠。本文统一为 stride=4096B、V offset=2048B。
4. 原每层 Linear MAC 实际为 192,937,984；加 4K attention 为 226,492,416，并非 327M。gate+up=100,663,296，down=50,331,648。
5. 1024 个乘法位置不自动代表 1024 MAC/拍。FP32 加法反馈延迟、归约吞吐、scale 单元、bank 端口和 HBM 排序均可能降低有效吞吐。
6. 1024 BF16 权重/拍需要 2048B/拍（409.6GB/s），14×512-bit@200MHz 仅 896B/拍（179.2GB/s）。BF16 lm_head 必然带宽受限。
7. 1024→256 BF16 输出串化是 4:1；FP32 输出同为 256 元素时仍是 4 拍。不能由位宽与元素数混用得出 8:1。
8. 每 token KV append=36×4096B=147,456B，在 20 tok/s 仅约 2.95MB/s 平均流量。12.8GB/s 是单 512-bit 写口理想容量，非实际 append 需求。
9. QK-Norm 对 32 Q 和 8 K 头分别归约；一棵全局 256→1 加法树不足以完成两个独立头的同拍归约。
10. 8KB 指令 RAM 与约750条带地址描述符不匹配。本方案按约64KB指令区预算；实际程序条数由汇编器报告。
11. 8GB 假设下，32K BF16 KV=4.832GB，加 W4 Linear≈3.581GB 已超8GB，再加独立 BF16 embedding/lm_head≈2.489GB 更不满足。地址位宽预留32K不等于支持32K容量。
12. 仓库未发现初稿所称 `mpu/`、`mcu/` Scala 骨架或 SpinalHDL 构建文件；当前 open_tc 是另一套 C++/RTL TensorCore 实现。应借鉴槽位、异步完成、仲裁方法，不能直接当作本 NPU 的实现。

## 3. 软件、模型与数值合同

### 3.1 模型 manifest

模型加载前必须提供：schemaVersion、H/F/D/Nq/Nkv/layers/vocab/maxContext、epsilon、ropeTheta、ropeScaling 类型和参数、Norm gamma tensor、是否存在各层 bias、embedding 与 lm_head 是否共享、权重格式、scale 格式、量化 groupSize、actOrder、张量地址与 byteLength、每个 artifact 的 SHA-256。任何不支持的组合在 RUN 前拒绝，不能静默套用默认值。

基础 v1 仅 groupSize=128、对称有效 q∈[-8,7]、无运行时 act-order。若 g_idx 是纯排列，导出器将权重与输入置换映射规范化并记录证据；不能消除的非连续 group 或非对称配置先拒绝。AWQ 的 zp 修正是后续扩展，不占用当前流字段。矩阵维度尾部填零，但有效 N/K 不改变；尾部值不能进入 argmax。

### 3.2 浮点语义

- BF16 为 IEEE 风格 1/8/7 位，FP32 为 1/8/23，FP16 scale 为 1/5/10；字节顺序 little-endian。
- 舍入默认 RNE；FP32→BF16 必须 round-to-nearest ties-to-even，不能截断低16位。BF16→FP32 是低16位补零，FP16→FP32 需要指数与 subnormal 转换逻辑。
- MUL 与 ADD 分离、各步 FP32 舍入；v1 不默认 FMA。累加次序按 §6 固定。替换为 FMA/IP 不得悄然改变 reference。
- 默认要求渐进下溢。若目标 IP 只能 FTZ/DAZ，需显式建立 `FP_MODE_FTZ` 并重新验收；软件模型必须相同。signed zero、NaN canonicalization、Inf 行为写入算术 wrapper 合同。
- 载入权重/scale 要求有限数；运行时 NaN/Inf 产生 sticky 数值标志，默认终止 token 并返回 NUMERIC_FAULT。softmax masked 元素作为无效元素处理，不送 `-Inf - -Inf` 给 exp。
- 延迟 scale：每个有效 group 的 `sum(q_i*x_i)` 经规定部分和树后乘 FP32(s)，再顺序累加 group。group和与总和均 FP32；仅最终写 BF16 dst 时舍入。
- GPU风格比较模型：先 `w_i=BF16(FP32(q_i)*FP32(s))`，再以其规定累加顺序计算。它是差异评估对象；不是当前硬件的位精确参考。

### 3.3 非线性与边界

RMSNorm：`r=rsqrt(sum_FP32(x²)/D + eps)`，`y=BF16(FP32(x)*r*FP32(gamma))`；中间两次乘法分别舍入。QK-Norm 每 head 独立，gamma按模型定义索引，不与 hidden RMSNorm 混用。

RoPE NeoX：对 i∈[0,D/2)，`y_i=x_i*c_i-x_(i+D/2)*s_i`，`y_(i+D/2)=x_(i+D/2)*c_i+x_i*s_i`，中间 FP32，出口BF16。position 为绝对 token 序号，非 tile index。v1 由主机按 manifest 生成当前 position 的 D/2 个 FP32 sin/cos，双缓冲最多2KB/position；可扩为硬件相位累加，但初稿4096点ROM不能独立保证所有ropeTheta/32K精度。

SwiGLU：`gate/up`先BF16；SiLU内部FP32计算，默认在乘 up 后才转 BF16，避免多余 BF16 中间舍入；该边界必须与软件 reference 一致。greedy argmax 比较 FP32，平局选择最小 tokenId，NaN按 NUMERIC_FAULT 处理；不把 argmax 称为随机采样。

SFU exp 使用 x≤0，建议 exp clamp下界=-80（更小输出0）；此截断为显式近似。rsqrt仅正有限输入。sigmoid对负/正输入用数值稳定形式，SiLU极端区间使用饱和旁路。精度验收分别看绝对/相对误差，接近0不能仅用相对误差；初稿统一≤2^-9为待验证目标，非实现事实。

## 4. 顶层端口与事务规则

### 4.1 顶层

| 端口 | 方向（相对NPU） | 合同 |
|---|---|---|
| ctrl | slave AXI4-Lite，32-bit data，16-bit addr | 配置、doorbell、状态、性能计数器 |
| hbm[j] | master AXI4，目标512-bit data、64-bit addr、6-bit ID | 逻辑端口0..13，物理宽度由adapter转换 |
| irq | out Bool | sticky completion/fault；软件W1C |
| clkCore/resetCore | in | 目标200MHz，同步逻辑reset |
| hbmClockDomains | shell决定 | 每接口CDC隔离，禁止直接跨域Stream |

AXI协议允许读写独立，但不得把 read-only 适配器宣称支持写。v1专用KV写适配器可绑定具有AW/W/B的物理口；其资源和映射由board shell确认。AXI burst不跨4KiB，INCR，按实际bus beat对齐；逻辑512-bit beat=64B。推荐16-beat burst（1KiB），尾部拆分。ID=6为wrapper目标，最终受shell约束，允许减少。

### 4.2 通用 Stream 规则

所有请求/数据/完成为 `Stream[T]`：仅 valid&&ready 接受；valid&&!ready 时payload及valid保持；计数器/游标/last均由fire推进。`last`表示当前命令/块最后一个有效beat，不靠空闲间隙识别。每条数据带cmdTag:16、epoch:16、tileId:16、beatIdx:16及mask（逐元素或逐字节）；非法mask/last产生PROTO_FAULT。

内部浮点IP若不支持反压：输入端先预留出口FIFO credit，再发射，latency条tag延迟线必须与数据同使能；不得只冻结tag而让IP继续跑。同步RAM读后到输出FIFO，credit覆盖全部在飞返回。跨时钟使用异步FIFO，不使用组合ready跨域。

### 4.3 主要子模块端口

| 模块 | 输入Stream | 输出Stream / 状态 |
|---|---|---|
| DCU | host提交、unitCompletion | MpuCmd、UvcCmd、DmaCmd、KvCmd；fault、tokenDone |
| McuReadCtrl[j] | ReadReq(addr64,bytes32,dst16,tag/epoch) | ReadBeat(data512,keep64,last,tag) |
| McuWriteCtrl | WriteReq、WriteBeat(data512,strb64,last) | WriteDone（必须在B响应后） |
| WeightUnpacker | ReadBeat、WeightBlockCfg | W4Slice(1024×4bit)、ScaleBlock(128×16bit) |
| WPU | W4Slice、laneMask、格式 | WOperand(128×8×16bit)、scaleMeta |
| MPU | MpuCmd、WOperand/KV operand、XOperand | ResultBeat(256×32bit,mask256,last) |
| MpuScaleUnit | GroupSum(128×FP32)、ScaleBlock | GroupResult / TotalResult；非任意拍无界接受 |
| VectorStore | BankReadReq、BankWriteReq | VectorBeat(256×16/32bit)、writeAck |
| UVC | UvcCmd、VectorBeat、AttentionTile | VectorResult、SoftmaxTile、UvcDone |
| SFU | SfuReq(32×FP32,op,mask32,tag) | SfuResp(32×FP32,mask32,tag) |
| KvCacheManager | AppendCmd、TileReadCmd | TileReady(slot,epoch)、AppendDone |
| SlotManager | allocate/release/markReady | slot state snapshot、credit |

1024个操作数不拉成跨芯片中心的大总线；128 tile按cluster布局，表中总位宽是逻辑总量。ScaleBlock可存metadata RAM后按8个/拍读取，不要求128个FP16同拍到跨片导线。模块命令接收只代表排队；完成由独立Stream发回。

## 5. 宏指令、控制寄存器与调度

### 5.1 控制寄存器（建议固定ABI）

| offset | 寄存器 | 语义 |
|---|---|---|
| 0x000 | VERSION | ABI版本/能力位，只读 |
| 0x004 | CONTROL | START/ABORT，脉冲；busy时START拒绝 |
| 0x008 | STATUS | idle/busy/done/fault |
| 0x00C | IRQ_STATUS | W1C |
| 0x010/014 | PROGRAM_BASE lo/hi | 64-bit指令区地址 |
| 0x018 | PROGRAM_COUNT | 指令数，不是字节数 |
| 0x020/024 | TENSOR_TABLE_BASE | 64-bit地址 |
| 0x028 | TENSOR_COUNT | 有效描述符数量 |
| 0x030 | CONTEXT_LEN | 运行前历史token数T；本轮attention读T+1 |
| 0x034 | POSITION | 当前绝对position，通常等于T |
| 0x038 | TOKEN_IN | embedding token id |
| 0x03C | TOKEN_OUT | 完成后的argmax |
| 0x040 | FAULT_CODE | 首个错误，sticky |
| 0x044 | FAULT_TAG | 首个错误关联指令 |
| 0x048/04C | CYCLES | 64-bit性能计数 |
| 0x080.. | PERF_SELECTOR/DATA | 单元忙、HBM等待、bank冲突、SFU等计数 |

配置写入shadow；START时原子锁存。完成前不得修改当前tensor table。host提交前保证DMA可见性。abort停止新发射并排空已发AXI，增加epoch，使过期内部完成失效；不能直接遗忘读响应或在AXI事务中重置接口。

### 5.2 512-bit指令格式

| bit | 字段 |
|---|---|
| 7:0 / 15:8 / 31:16 | opcode / flags / cmdTag |
| 47:32 / 63:48 / 79:64 / 95:80 | src0Id/src1Id/src2Id/dstId |
| 111:96 / 127:112 / 143:128 / 159:144 | M/N/K/T（16-bit，各自含义按opcode） |
| 191:160 / 223:192 | aux0/aux1（head、position或操作参数索引） |
| 255:224 / 287:256 | imm0/imm1（FP32标量或整数） |
| 319:288 / 351:320 | waitEvent0/waitEvent1（低8位slot，高24位generation） |
| 383:352 | signalEvent（同编码，0xFFFFFFFF为无事件） |
| 415:384 | repeatCount（循环体外层token/层数，不隐式重复当前指令） |
| 447:416 | extensionId（可选静态扩展描述符索引） |
| 511:448 | reserved，必须0 |

repeatCount仅 LOOP_BEGIN 使用，LOOP_END指定配对PC于aux0；v1可先展开程序，无loop能力时拒绝loop opcode。M/N/K/T均可表达32768，禁止溢出后按0解释；dim=0对算术指令非法。

64B tensor descriptor：byte0..7 base64；8..11 byteLength32；12..15 dim0；16..19 dim1；20..23 stride0Bytes；24..27 stride1Bytes；28 format8；29 space8；30 layout8；31 flags8；32..39 auxBase64（scale/metadata）；40..43 auxStride；44..47 capacityElements；48..55 layoutParameters；56..63 reserved=0。地址空间为 HBM/VectorStore/OperandStore/KvTileStore，HBM base绝对字节地址、片上base为该space的字节offset。大于4GiB的单tensor须分段或扩展descriptor，不能用byteLength32描述整个KV池。

### 5.3 指令集合

| opcode | 操作 | 完成条件 |
|---|---|---|
| 0x01 EMBED | token查表→hidden | VectorStore写入可见 |
| 0x02 GEMV_W4 | W4 delayed group Linear | 所有group和出口写完成 |
| 0x03 GEMV_BF16 | native BF16 lm_head等 | 最后结果被消费者接受 |
| 0x04 RMSNORM | hidden/per-head norm，aux指定分段 | 全部head出口写完成 |
| 0x05 ROPE | Q/K分别执行 | 全部配对出口写完成 |
| 0x06 KV_APPEND | 当前layer当前token K/V | HBM B响应成功 |
| 0x07 ATTN_TILE | QK→online update→PV，tile循环内部FSM | 所有tile和最终o/l写回完成 |
| 0x08 ADD | residual，支持受控原地 | 全部写确认 |
| 0x09 SWIGLU | gate/up→activated FFN | 全部输出写完成 |
| 0x0A ARGMAX | logits流/缓存→token | 全部有效词表元素归约 |
| 0x0B DMA | 通用搬运 | 目标空间写确认 |
| 0x0C FENCE | 等待此前操作/写事务 | 所有指定队列drained |
| 0x0D TOKEN_DONE | 提交本token上下文状态 | layer append均确认，输出token可见 |
| 0x0E/0x0F | LOOP_BEGIN/END | 仅声明该能力时支持 |
| 0x10 GEMM_W4 | prefill W4分组矩阵乘 | 所有query/output tile出口写完成 |
| 0x11 ATTN_PREFILL | 多query、因果online attention | 全query tile结果写完成 |
| 0x12 EMBED_BLOCK | prompt token序列查表 | 全block hidden写完成 |
| 0x13 KV_APPEND_BLOCK | 连续多个prompt token K/V | 全部B响应成功 |

ATTN_TILE是一个融合宏指令，内部发射有限的QK/PV/softmax微事务，不暴露每128-token tile给host。非法op、尺寸、地址溢出、格式、bank alias 在执行前报错。

### 5.4 依赖与资源

命令队列：MPU深度4、UVC深度4、DMA读每通道8、KV命令4、完成FIFO至少16（最终按最大在飞量和credit证明）。DCU按程序顺序尝试分发，单元内顺序执行；不实现任意乱序。事件共有64slot，generation每次复用加1；等待两个显式事件加资源scoreboard。编译器禁止等待将来的未提交事件，并做依赖图无环检查。

slot状态为 FREE→FILLING→READY→IN_USE→DRAINING→FREE；消费者完成且全部出口写可见后才释放。用上一拍已注册快照判定可发射，避免ready组合环。每slot携带epoch和tileId。MPU全局单占用；UVC与DMA仅在不同bank/slot上重叠。存在读写alias的两条命令串行，除明确支持的同索引ADD原地操作。

基线layer顺序：保存residual→attention RMSNorm→Q/K/V投影→Q/K per-head norm→RoPE Q/K→KV append→attention→O投影→ADD→FFN RMSNorm→gate/up→SwiGLU→down→ADD。Q/K完成、V完成、KV可见是attention依赖。初版等待append B再读本token；后续可通过当前token staging旁路，但必须按相同数据语义验证。先实现完整Q/K/V再attention，不采用未经证明的head-wise重叠。

## 6. MPU 微架构、映射与延迟

### 6.1 统一DOT8映射

每tile拥有8个BF16乘法位置、3层FP32 DOT8树；128 tile并行处理128输出行。每拍每tile读取8个weight和8个activation并形成一个DOT8。GEMV时activation向量8元素广播给所有tile；QK/PV时按head/tile广播或读取。

矩阵 y[N]=W[N,K]x[K]：tile t对应row=nBase+t；step u对应k=8u..8u+7；尾行与尾k掩码。理想乘法发射周期 `ceil(N/128)*ceil(K/8)`。此方案与1024 AXPY位置计算量相同，但权重布局、归约顺序和反馈结构不同。

### 6.2 group累加与feedback hazard

令 FP32累加器 latency=Lacc、II=1，选择 C=nextPow2(max(1,Lacc))；**目标C=16，必须Lacc≤16**。同一partial context两次更新至少间隔C个实际发射时钟；不能只按逻辑step距离在停顿时压缩时间。一个128输入group有16个DOT8，映射 `context=step mod C`；目标C=16时每context只接收一个DOT8，group归约为固定4层FP32树，无本group反馈依赖。

每tile的group归约树必须吞吐≥1 group/16拍；128tile同时完成的group sum写入双缓冲GroupStore。8个ScaleMul管线每拍处理8行，16拍覆盖128行；scale结果更新每行FP32 TotalStore，同行相邻group间隔至少16拍，要求总和ADD latency≤16，超出则stall或重新设计。TotalStore bank按row mod8，每bank一个更新/拍。scale+add在飞记录row/group/tag，RAW forwarding或scoreboard禁止读旧值。

双group缓冲在计算group g+1时缩放g；buffer未释放则MPU停发，不能覆盖。跨group总和按g递增相加，最后group后排空pipeline，读出结果。BF16连续矩阵若K>128也可使用同一group树但scale=1；明确每128输入有一次FP32总和相加的数值边界。

这不是“1024个FP32累加器约450LUT”的资源承诺。DOT8树包含128×7=896个FP32加法位置；group归约、scale、total add和bank都需综合测量。可以时分group树但须证明平均8行/拍和双缓冲时限。

### 6.3 延迟合同

| 单元 | 目标II | latency |
|---|---|---|
| BF16 MUL→FP32 | 1 | Lmul，IP配置后确定 |
| DOT8树 | 1/tile | 3·Ladd（不是固定3拍） |
| group16树 | 1 group/16拍/tile或等效时分 | 4·Ladd + serialization |
| scale FP32 MUL | 1/实例，8实例 | Lscale |
| TotalStore ADD | 1/实例，8实例 | Ltotal≤16为目标 |
| FP32→BF16 | 1/向量位置 | Lcast |
| RAM read | 1/端口 | 默认2拍（可配置） |

Spinal elaboration读取latency参数，在相应tag path生成pipeline。命令 latency=`fill+issue+drain+writeAck+stall`；仅 issue 项由MAC数量决定。对 K=128、N=128的W4 group，理想issue=16拍，不保证16拍完成。

### 6.4 QK与PV映射

QK tile长度B=128：tile编号t=`qh*4+pLane`，qh=0..31，pLane=0..3；posBatch=0..31；token=`tileStart+4*posBatch+pLane`。每tile DOT8覆盖dim块，16步形成一个token的score。每tile score乘 `1/sqrt(D)` 后保留FP32。GQA `kvh=floor(qh/4)`，K操作数在四个qh间共享；Q独立。理想QK=32*16=512 issue拍/tile。

PV：输出索引 `(qh,d)` 映射连续128行batch，128token以每step8token沿K归约；每qh的8个概率广播到其dim；V由相应kvh共享。输出32*128=4096行，32行批×16step=512issue拍/tile。结果是本tile未归一化o contribution。QK和PV共用MPU，不重叠执行。

K/V访存须提供转置的计算视图：QK读[head,position,dim]，PV读[head,dim,position]。由TileStore离线于fill阶段bank scatter或显式transpose阶段完成，不能假定token-major能直接每拍满足两种端口模式。冲突时记录stall，而不是读错数据。

## 7. 流式 Attention 与 UVC/SFU

### 7.1 精确在线状态合同

每qh维护 FP32 m=-Inf、l=0、o[D]=0。对有效tile scores `s_j=dot(Q,K_j)/sqrt(D)`：

```
m_new = max(m_old, max(valid scores))
alpha = (l_old == 0) ? 0 : exp(m_old-m_new)
p_j_fp32 = exp(s_j-m_new)  // invalid j -> 0
l_new = alpha*l_old + sum(p_j_fp32)
p_j_bf16 = BF16(p_j_fp32)
o_new[d] = alpha*o_old[d] + sum_FP32(p_j_bf16 * V[j,d])
final[d] = BF16(o[d] / l)
```

l用FP32概率、PV用BF16概率是明确的近似组合，须单独评估；若参考需要相同权重和归一化语义，可将l改为sum(FP32(p_bf16))并标记模式，不能混用结果。这里的p不是最终softmax概率；不能先对tile独立归一化再直接累加。

每次更新m都须缩放**全部D个旧o元素**，不只是m/l。全masked tile保持m/l/o不变；有效长度为T+1且当前token允许自注意力；prefill的每query因果mask另行实现。最后l=0为配置/数值错误。v1用SFU reciprocal或rsqrt替代算法时需独立精度验收，不能直接把rsqrt当reciprocal；可专门实现reciprocal LUT/Newton通路。

### 7.2 tile执行次序

KV tile ready→QK→head max reduction→exp与sum→旧o缩放→PV→o/l/m提交→释放tile。下一tile DMA可与当前计算重叠。score与p双缓冲支持后续流水优化，但同head状态提交必须按token tile顺序。SFU争用优先attention再norm再SwiGLU，公平计数防饥饿。

每tile4096个score，32 SFU的exp发射下界128拍；alpha最多32个另占1拍。UVC处理4096个o元素至少16拍/遍。不能将softmax净周期预先记为0：MPU QK末尾才有完整head max，数据依赖限制重叠。

### 7.3 UVC datapath

VectorStore分别提供srcA/srcB/srcGamma独立读bank组或phase复用；输出写到不同bank组。256位置处理BF16或FP32，有mask、segmentId、index。全局sum/max/argmax树8级逻辑层，latency为8·Lreduce；每拍吞吐256输入。per-head Norm使用2棵128输入树（D=128）；D=256为单树，D非支持集合先拒绝或走可验证分段归约。

RMSNorm两遍：读x平方/归约→除D+eps→SFU rsqrt→重新读x/gamma并归一化。issue下界2·ceil(H/256)=32拍，另加MUL/reduce/SFU/broadcast/drain。QK-Norm处理40head，共5120元素；每head128维，256位置一次2head，共20发射拍进行reduce，第二遍20拍，另加pipeline。不能固定说41拍含所有阶段。

RoPE每对需要4个乘法和2个加法。若256个位置只各有1乘法器，则每拍处理最多64对，Q+K共2560对需要≥40乘法发射拍；若增加乘法器才可提高吞吐。SwiGLU SFU发射≥ceil(F/32)=384拍，乘法与store可流式重叠，最终cycle由最慢stage和drain决定。

argmax lm_head输出直通UVC，只保留(blockMax,tokenId)，无需整词表logits RAM；出口按有效mask屏蔽padded词。初稿argmax594拍只指256lane满供给，不代表额外独立总时延。

## 8. 片上存储、bank 与端口预算

### 8.1 地址和并发

VectorStore用256个32-bit bank，逻辑元素e：bank=e mod256，row=floor(e/256)。BF16采用每槽低16bit存一个元素，禁止把两个不同并发元素压进同word造成read-modify-write。若使用16bit物理bank，FP32 tensor分配独立32bit bank组。每组默认1R1W，同地址read/write必须明确forwarding或阻止，不能依赖设备不确定的read-during-write行为。

统一地址空间不等于单物理端口：建议分为Hidden/Residual、QKV、FFN、AttentionState四组。srcA/srcB可分配到不同组；同组双源冲突则两拍读取缓存，不承诺256元素/拍。FFN gamma/hidden输入可从独立OperandStore广播。RoPE配对元素跨半向量bank映射，在D=128时安排pair fetch到局部FIFO。

### 8.2 容量基线（逻辑字节，不是直接器件块消耗）

| 数据 | 基线容量 | 备注 |
|---|---:|---|
| hidden A/B + residual | 24KiB | 3×4096×2B |
| normalized input | 8KiB | 可与生命周期已结束空间复用 |
| Q + current K/V | 12KiB | 8KiB+2KiB+2KiB |
| attention output BF16 | 8KiB | 4096元素 |
| gate/up/activated | 72KiB | 3×12288×2B；F=19200时112.5KiB |
| KV tile A/B | 1MiB | 2×128×4096B |
| FP32 scores A/B | 32KiB | 2×32×128×4B |
| BF16 p A/B | 16KiB | 2×32×128×2B |
| online o | 16KiB | 32×128×4B |
| m/l/alpha | <1KiB | FP32、双状态或commit寄存器 |
| W4 group A/B | 16.5KiB | 2×(8192B weights+256B scales) |
| MPU group/total/state | ≥数KiB | 128行group双buf、TotalStore、tag/credit |
| BF16 weight staging | ≥16KiB | 双缓冲，块大小按HBM返回与MAC消费率设计 |
| 指令RAM | 64KiB起 | 1024×64B，可从HBM refill |
| Norm gamma | ≈16.5KiB/当前layer | 2H+2D个BF16，按实际模型修正 |
| RoPE/SFU tables | 参数化 | 每函数ROM实际字宽含coeff，不沿用16KB总数 |

基线逻辑总量约1.4MiB，加DMA FIFO/重排和fragmentation，实际URAM/BRAM更高。8-bank W4宽读、256-bank VectorStore可能因最小RAM粒度显著浪费；综合报告必须同时列逻辑字节和物理块数。FFN三个buffer不能被2×8KB hidden缓冲代替。

### 8.3 KV TileStore具体计算视图

fill接收token-major data后，按kvh拆分。QK计算视图每kvh有4个position lane bank，每bank一个token的8dim向量（128bit），16个dim step；每拍8kvh×4pos×8dim=256 BF16读，GQA复制到1024乘法位置。PV视图按kvh/dim/token8存储，每拍读取所需128输出行对应的共享V向量；不同qh复用同kvh数据。要达到上述供给，可在fill时同时构建V转置bank视图或构建一个冲突可证明的bank排列；**1MiB预算只含一份K/V，不含复制品**。

实现首版建议K原视图、V显式转置，transpose阶段计入周期，保证一份数据无别名。确定bank宽度/数量后，用访问trace检查每cycle每bank≤1R1W；不满足则stall或扩大bank，不能把这一设计门槛留给自动综合解决。不同槽互不共享write bank组，fill A与compute B独立。

## 9. 权重格式、HBM地址与MCU

### 9.1 离线W4重排（取代未定义18-word布局）

文件标准格式 `W4_DOT8_G128_V1`：先按输出128行批 r，再group g存储。一个full block：4个64B scale beat +128个64B weight beat=8448B。scale[row]为FP16共256B，无zero字；weight每beat包含16行×8输入×4bit=64B。

```
block = (rowBatch * groupCount + group) * 8448
scaleAddr(row) = Wbase + block + 2*row
weightBeatAddr(step, rowChunk) = Wbase + block + 256 + 64*(step*8+rowChunk)
row = rowChunk*16 + localRow
nibbleIndex = localRow*8 + inputLane
q = signExtend4(nibble)
```

step=0..15、rowChunk=0..7。nibble0在byte0低4bit；nibble1在高4bit。q=-8到7通过小ROM转BF16（不是对任意值简单拼指数）。N尾行、K尾group离线零填，scale为1，mask屏蔽。N/K补齐后的字节数必须进入manifest。

W4Store有8bank，每bank512bit宽，每step并行读取8beat，形成1024个INT4。平均含scale供给528B/issue拍=105.6GB/s；10×512b理想128GB/s，仅17.5%余量，加入有效效率后未必充足。HBM分条须由地址router定义，例如64B逻辑beat按channel=beatIndex modC、localBeat=floor(beatIndex/C)，block基地址必须携带logicalRegion；不能把该虚拟映射地址直接当shell物理地址。

BF16格式同DOT8顺序，每rowChunk beat需要256B（16行×8×2B），split为4个64B beat；单issue2048B，总线不足则ready拉低，乘法阵列低于满利用率。embedding保留row-major，token随机读取连续H×2B，不用DOT8权重布局。

### 9.2 KV物理公式

对于token-major当前参数：

```
addr(layer,token,isV,kvh,d) = kvBase
  + layer*layerStride + token*tokenStride
  + isV*planeStride + kvh*headStride + d*2
headStride  = alignUp(D*2,64)               // D128 ->256
planeStride = Nkv*headStride               // 2048
 tokenStride = 2*planeStride               // 4096
layerStride = alignUp(maxContext*tokenStride,4096)
```

K与V各2048B，append整个4096B=64beat。所有地址用至少64bit中间乘加并检测溢出、range和alignment；layerStride来自capacity，不来自当前T。读tile尾部仅发实际有效token，不访问池外。T+1>maxContext返回CONTEXT_OVERFLOW；v1不做隐式ring覆盖，不假设sliding-window。

### 9.3 MCU状态与返回排序

McuReadCtrl：IDLE→ISSUE_AR→COLLECT_R→REORDER/FILL→DONE；可多burst在飞，但每ID保持原序，跨ID由(tag,logicalBeat)重排。AR发射前预留全burst空间；RRESP非OKAY产生BUS_FAULT并完成排空。读bytes未对齐时由明确的first/last keep处理，权重/KV正常请求按64B对齐。

每通道在飞深度由 `ceil(BW_channel * latency / burstBytes)` 确定。示例12.8GB/s、300ns、1KiB burst需要至少4在飞burst，推荐8作为初始参数；300ns是假设，必须实测。FIFO至少覆盖全部不可撤销返回 + skid。分段切用途先停止新AR并drain旧tag；不以“通道切为KV”丢弃尚未返回的weight。

KvCacheManager两个slot：FREE→FETCHING→READY→READING→FREE；只有前一slot release后允许重填。append确认维护每layer visibleLen，而全token committedLen只在TOKEN_DONE递增。失败时标记本token无效；不要求物理擦除未提交数据，但后续必须按 committedLen重写。

## 10. 延迟与性能的可验证预算

### 10.1 MAC issue下界（满利用率、无stall）

| 算子 | MAC | 理想issue拍 | 200MHz时间 |
|---|---:|---:|---:|
| Q proj | 16,777,216 | 16,384 | 81.92µs |
| K+V proj | 8,388,608 | 8,192 | 40.96µs |
| O proj | 16,777,216 | 16,384 | 81.92µs |
| gate+up | 100,663,296 | 98,304 | 491.52µs |
| down | 50,331,648 | 49,152 | 245.76µs |
| 每层Linear | 192,937,984 | 188,416 | 942.08µs |
| QK @4096 | 16,777,216 | 16,384 | 81.92µs |
| PV @4096 | 16,777,216 | 16,384 | 81.92µs |
| 每层合计 | 226,492,416 | 221,184 | 1105.92µs |
| 36层合计 | 8,153,726,976 | 7,962,624 | 39.81312ms |
| BF16 lm_head | 622,329,856 | 607,744 | 3.03872ms |

vocab在DOT128布局恰好151936/128=1187，无尾输出tile。旧AXPY1024输出行布局需要ceil(V/1024)=149批、610304拍；两种不同映射不能混用尾部周期。

lm_head BF16 byte=1,244,659,712，12通道@200MHz理想153.6GB/s，下界8.10325ms；14通道理想179.2GB/s，下界6.94565ms。表中保守延续12通道计算。embedding随机读、DMA延迟、pipeline、向量、softmax、transpose均未计入。

### 10.2 上下文基线下界（历史长度近似T，实际含本token）

| T | 36层+lm_head理想时间 | 理想上限tok/s | KV逻辑容量 |
|---|---:|---:|---:|
| 1024 | 43.493ms | 22.99 | 150,994,944B |
| 4096 | 47.916ms | 20.87 | 603,979,776B |
| 8192 | 53.815ms | 18.58 | 1,207,959,552B |
| 16384 | 65.611ms | 15.24 | 2,415,919,104B |
| 32768 | 89.204ms | 11.21 | 4,831,838,208B |

这张表是**条件化理论下界**，不是吞吐预测，更不是20tok/s达标证据。4K的50ms预算仅剩2.084ms余量。32tile/层×36层，每tile额外128个exp发射拍就贡献约0.737ms（未计alpha/reduce/rescale/drain），向量与bank/AXI overhead会进一步占用余量。

每token 4K KV读≈604MB；attention理想5.898ms合计得到约102.4GB/s。8通道理想102.4GB/s意味着100%利用率、**没有协议/转置余量**。14读口统一分段仲裁，在attention阶段至少10通道目标且验证共享物理带宽。W4阶段约3.581GB含scale /33.915ms≈105.6GB/s。有效效率η必须实测：例如10通道η=0.8仅102.4GB/s，已小于W4需求。

### 10.3 HBM容量门槛

36层W4 Linear含FP16 scale、无zero约3,581,411,328B；独立BF16 embedding+lm_head共2,489,319,424B。4K KV≈603,979,776B，三项合计≈6.675GB十进制，未计其他张量/对齐/staging。32K三项合计≈10.903GB，8GB与8GiB均不够。若目标真实板卡容量不同，应重新计算。

导出器需给出实际 padded byte total；启动前计算 `sum(regionBytes)+reserved<=availableHBM`。共享embedding/lm_head需要兼容两种布局或选择额外重排策略，不能直接按“权重共享”减去一份却保留两种独立布局。泛化最大参数不作为同一bitstream的容量保证。

## 11. SpinalHDL 实现组织

建议在独立新项目实现，不把当前open_tc改成Scala工程。此处仅定义计划，没有创建或声称已配置Spinal工具链。

```
npu-v1/
  build.sbt / project/build.properties
  src/main/scala/npu/
    NpuConfig.scala                 // widths, latency, capacity
    protocol/{Commands,Streams,Descriptors}.scala
    control/{RegisterFile,Dcu,Scoreboard,SlotManager}.scala
    memory/{VectorStore,OperandStore,KvTileStore,AddressRouter}.scala
    mcu/{ReadCtrl,WriteCtrl,WeightUnpacker,KvCacheManager}.scala
    compute/{FpWrapper,MpuTile,MpuGroupReduce,MpuScale,Mpu}.scala
    vector/{Uvc,SegmentReduce,Sfu,Rope,OnlineAttention}.scala
    top/{NpuTop,BoardShell}.scala
  src/test/scala/npu/...
  reference/                       // bit-accurate + algorithm model
  tools/                           // exporter/assembler/cycle model
```

SpinalHDL/Scala/sbt/JDK版本必须锁定且相互兼容，待选择真实工程模板后冻结，不凭空宣称某版本可用。采用Stream、StreamFifo、Mem、ClockDomain、BlackBox抽象；board专属AXI/HBM、DSP/IP在wrapper隔离。可移植RTL实现和vendor IP实现使用同一算术合同及tag接口。

FP wrapper暴露format、latency、II、rounding、FTZ、hasReady、exception flags；simulation用位精确函数或经过验证的算术RTL，不能将Scala Double结果直接转Float当作所有中间舍入参考。BlackBox仿真须有模型/RTL，不允许空模块“通过”功能测试。

层次切分：8或16tile一个cluster，局部W bank与activation寄存复制，cluster间只分发小广播与tag；256lane UVC同样分cluster。禁止Scala组合循环构建1024-to-256通用全连接crossbar；固定映射串化输出。elaboration assertions检查正值、容量、bank数量、C≥Lacc、数据宽度、burst不跨页、descriptor ABI bit总和。

## 12. 验证、覆盖与验收

### 12.1 四种reference明确分开

1. 数学模型：高精度算子语义，用于理解算法误差。
2. checkpoint框架模型：实际量化checkpoint、实际BF16 dequant/累加实现，用于端到端差异评估。
3. bit-accurate硬件模型：本文所有舍入、部分和树、FTZ/饱和、tile边界和p_bf16规则；RTL必须与其匹配。
4. cycle模型：有限FIFO、HBM延迟/效率、bank端口、latency/II、stall；不能用MAC÷1024替代。

当前open_tc standalone CModel具有FP22计算及FP8等转换，本次实测默认FP8_E5M2用例通过8个pipelined jobs、19周期，但它**不能验证本NPU BF16×BF16→FP32语义**。FP22 reference仅能复用测试组织思想。

### 12.2 模块验收清单

| 目标 | 必测项 | 通过条件 |
|---|---|---|
| W4 exporter | 16种q、FP16 scale、实际checkpoint qzeros/g_idx | decode后权重与文件解释一致；错误配置明确拒绝 |
| AddressRouter | layer/token/head/dim边界、最大地址 | 所有有效地址唯一、不越界、不跨页burst |
| MCU | 跨ID乱序、随机RREADY/ARREADY、RRESP错误 | 无丢失/重复、tag排序正确、可排空 |
| WPU | q=-8..7、mask、group切换stall | BF16逐bit正确、scale与group完全对齐 |
| MPU | N=1/127/128/129、K=8/127/128/129/4096 | 输出与bit model逐bit相等，无RAW hazard |
| UVC norm | D=128/256、多head、零向量、eps | 分段无串扰、与bit model一致 |
| RoPE | position0/4095/32767、pair、theta/scaling | 符号/索引/绝对position正确 |
| SFU | 分段边界、极端范围、近0 | 全输入范围误差报告通过冻结门限 |
| online attention | tile1/127/128/129、不断抬升max、GQA | m/l/o正确重缩放，尾mask正确，无head串扰 |
| KV | append后read、T上限、abort/restart | B响应与可见性一致，epoch隔离 |
| DCU | 事件复用、队列满、alias、fault | 无deadlock、未完成数据不被消费 |
| argmax | vocab尾部、相同值、负Inf/NaN | 最小id tie-break、尾部不参与、NaN错误 |

全链路随机反压覆盖每个Stream；记录seed和失败cmdTag。断言：接受数=完成数+在飞数；释放slot前consumer为0；valid在stall时稳定；所有写在range内；完成只一次；masked lane不更新acc。超时是失败，零测试执行不能作为通过。

### 12.3 模型级验证

先小shape、单算子、单layer，再完整36layer固定token decode；对hidden、Q/K/V、attention、FFN、logits逐阶段采样。比较指标：maxAbs、RMS error、cosine、logit排序/greedy一致率、长序列漂移、perplexity。延迟scale、BF16 p、SFU近似分别做消融，不能把所有误差归因于量化。

端到端门限由baseline checkpoint及任务确定；本稿不捏造“top1必须100%”或统一ulp门限。必须有固定数据集、tokenizer版本、prompt/token序列、baseline框架版本和精度配置。精度门限未冻结并通过前，延迟scale只叫候选方案。

### 12.4 实现阶段与放行门槛

- P0：manifest/exporter/assembler/数值模型/地址与容量检查；不依赖FPGA实现即可完成。
- P1：1个MpuTile + 小VectorStore + AXI仿真；确认真实FP IP latency/II/subnormal与tag同步。
- P2：16tile cluster、group scale与在线attention小规模功能；随机反压与bank访问覆盖。
- P3：128tile完整阵列、256 UVC位置、32 SFU；综合后测Fmax、RAM、DSP、LUT及routing，更新cycle模型。
- P4：板级HBM单读/多读/读写压力测试，测可持续带宽、返回延迟、并发效率，再跑单layer。
- P5：完整decode并评估数值/性能；20tok/s在4K真实上下文、包含lm_head/控制/append后才算达标。
- P6（v1必选、最终放行前）：prefill多query tile、causal mask、KV生成、权重复用与容量预算；完整prompt→decode联调，不能只切MPU mode。P5通过仅表示decode阶段完成。

每阶段输出证据：版本hash、seed、test count、失败数、有效参数、仿真/综合/板测分别标注。性能计数拆为issue、drain、HBM wait、scale wait、bank stall、SFU wait、DMA transpose、queue wait，避免重复累计并发段。

## 13. 尚需冻结的信息

| 问题 | 当前建议/影响 | 何时必须解决 |
|---|---|---|
| checkpoint配置已核验，tensor解释待完成 | compressed-tensors pack-quantized，static act-order，scale dtype待核验 | exporter与端到端reference之前 |
| v1范围已确认 | prefill与decode均为必选，见§14 | 不再作为待用户选择项 |
| V80料号/HBM实际容量/AXI shell | 决定32K容量、通道、DSP与跨域实现 | board shell与资源预算前 |
| vendor浮点IP的L/II/FTZ/资源 | 决定DOT/acc结构、真实性能 | MpuTile冻结前 |
| 允许的数值误差门限 | 决定是否接受delayed scale、BF16 p、SFU | 模型级放行前 |
| host接口与部署 | PCIe/DMA、coherency、模型载入 | 集成测试前 |
| RoPE特殊scaling、attention bias/滑窗 | 基础v1不隐式支持 | checkpoint兼容声明前 |

建议先冻结软件可见ABI、W4 layout、KV layout、Stream握手与bit reference，再写Spinal模块；性能与资源通过实际IP及HBM测量逐步收敛。若精度验证不接受延迟scale，应新增BF16 dequant权重路径并重新做带宽/资源预算，而非修改reference掩盖差异。

## 14. Prefill 必选功能的详细合同（按用户确认补充）

本章覆盖§1中早先decode优先的默认规划：**v1最终验收同时包含prefill与decode**。prefill不是把tokenNum改32即可，它增加二维激活、持久hidden存储、多query softmax状态、因果掩码和权重复用。支持首次prompt与已有cache的增量prefill；并发序列仍为1。

### 14.1 维度与程序顺序

历史已提交长度T0，新增prompt长度S，要求T0+S≤maxContext；query绝对position为T0+q。Linear定义 `Y[M,N]=X[M,K]·W[N,K]^T`，M为当前query block有效token数，基础tile目标Mq=32、输出tileNr=128、Kgroup=128。M尾部允许1..31；物理填充不生成KV，不影响计数。

采用**layer-major**程序（先完成该layer所有prompt token，再进入下一layer）：

1. prompt token embedding写HiddenA HBM region。
2. 对layer l的所有query blocks执行attention RMSNorm→GEMM Q/K/V→QK-Norm→RoPE→KV_APPEND_BLOCK。Q保存到当前layer Q scratch，K/V存本layer KV。
3. 等待本layer全部新增K/V写确认。再对每query block执行ATTN_PREFILL→O GEMM→residual ADD→FFN RMSNorm→gate/up GEMM→SwiGLU→down GEMM→ADD，结果写HiddenB。
4. 完成本layer后交换HiddenA/B，Q scratch可复用。已完成layer KV保留。最后layer只对最后一个有效prompt token执行final norm/lm_head/argmax；完整prompt logits为可选调试功能，不能默认151936×S缓存。
5. 所有layer写入与计算成功后，原子提交 committedLen=T0+S，再进入decode。正常后续decode输入是prefill输出的next token，position=T0+S。

此基线先写全layer KV再注意力，便于明确可见性；未来token虽然已在cache中，仍必须由causal mask禁止读取贡献。layer内部可用visibleLen优化流水，但须按query可见性而不是全局len判定。

若prefill失败，global committedLen保持T0，临时hidden/Q scratch失效，整个新增区间重新执行。不得将某些layer已append视作完整模型已提交。已写的未提交KV可覆盖，不要求清零。

### 14.2 GEMM映射与weight复用

128 MpuTile组织为 `tile=qt*32+nt`，qt=0..3、nt=0..31；每tile的8乘法位置沿K。每拍4query×32输出×8K=1024乘法。其activation供给为4×8 BF16=64B，独立于decode单广播8元素；权重32×8 BF16=512B经4query广播复用，仅需256个独立q值。

一个M32×N128×K128工作块：query microbatch有8个（每次4token），output microtile有4个（每次32输出），每个组合16个DOT8 step，总32×16=512个issue拍。计算量32×128×128=524,288 MAC，符合÷1024下界。

每个W4 G128 block（8448B，见§9）载入一次并在全部32个micro组合期间保持；优先按group g外层循环、query/row micro组合内层循环。下一weight group双缓冲预取。WPU从128-row块中选择当前32行，四query共享相同32行scale。

PrefillTotalStore必须保存整个M32×N128 tile的4096个FP32总和=16KiB；不是decode的128行总和。scale单元处理每micro组合128个group sum，8lane×16拍，吞吐与16issue拍匹配。其tag扩为queryIdx、outputIdx、groupIdx；相同query/output总和在下一group更新前已完成。微组合的局部group16部分和可沿用§6，但寄存器状态不得在上一组合drain前复用。

TileStore输入一次保存M32×K的BF16 X；H4096为256KiB，K19200最大约1.172MiB。有限buffer时按Kgroup分片读取，保留当前Mq×128 BF16=8KiB；weight也按group驻留。每个output tile重新读X group可能增加HBM流量，必须由cycle模型计入。不能在规格中同时声称小片上容量与完整输入无限重用。

理想issue `ceil(M/4)*ceil(N/32)*ceil(K/8)`；M与N尾tile利用率损失明确计入。实际group调度复用Nr128，输出tile计算完写Y并推进；Mq32的W4 GEMM最低weight读带宽约decode的1/32（不计尾部和spill）。复用收益来自物理驻留，不来自HBM自动缓存。

### 14.3 Prefill Attention 映射

query block Bq=32、key tile Bk=128。每head的QK子块用MPU 4query×32key×8dim映射：8个query microbatch×4个key microtile×ceil(D/8) issue拍。D128时每head512拍，32head共16,384拍/key tile/query block（未计mask利用率与输出stage）。head间顺序遍历，KV head=floor(qh/4)；可按共享kvh的4个qh连排，复用相同TileStore数据。

PV对一个qh采用4query×32dim×8key的同阵列映射；D128对应4dim microtile，Bq32对应8query microbatch，每micro组合128/8=16步，同样每head512拍。QK与PV共享MPU，key tile循环必须执行各query独立的online state更新。

概率与scores不要求32query×32head同时驻留：首版按一个qh处理完整Bq×Bk矩阵（4096score=16KiB FP32、8KiB BF16 p），保持该head全部query的online o为32×D×4B（D128=16KiB）。然后处理下一个head并把结果写AttentionOut；可复用decode score/p/online RAM，但需要配置不同维度与索引。

外层建议 `queryBlock → kvHead → qHeadWithinGroup → keyTile`，一个qHead处理全context后释放online state。KV tile在四个Q头间可重用的条件是所需keyTile仍驻留；若按每qh完整key循环，会重复从HBM读取。要实现GQA读复用，可换为 `queryBlock→kvHead→keyTile→4 qHeads`，此时至少保存4个qh的online状态（4×32×128×4B=64KiB）及4套m/l，推荐采用后者；scores/p仍顺序复用，不必四倍复制。其QK/PV算术总量不变。

### 14.4 因果掩码与online状态

每query绝对位置pq=T0+queryStart+localQ，key绝对位置pk=keyTileStart+localK；仅 `pk≤pq`、`pk<T0+S`、queryMask有效时参与。整tile都未来时跳过DMA/计算；混合tile必须逐元素mask。跳过tile不修改m/l/o，不能把max=-Inf更新为NaN。

每(qh,localQ)独立维护FP32 m/l/o，按§7相同alpha与p_bf16合同；query之间不共享m或l。SFU输入携带query/head tag，归约只能沿key维。单个qh的Bq32×Bk128 exp=4096元素，32SFU发射下界128拍；GQA4qh共512拍，不能沿用decode每tile128拍总开销。

D=128时32query o有4096元素，UVC每遍至少16issue拍；4qh至少64拍。最终o/l沿每query独立normalization，写[q,head,dim] contiguous AttentionOut。所有padded query禁止写输出和KV。

增量prefill的注意力计算有效总token-pair数为 `S*T0 + S*(S+1)/2`，不是S*(T0+S)满矩阵。不能用整个prompt非因果attention的MAC数宣称causal性能；也不能在mask仅屏蔽而未跳过计算时按半矩阵计算cycles。

### 14.5 Prefill外部与片上存储

新增HBM区域：HiddenA、HiddenB（每份maxPromptBlock/globalS×H×2B）、当前layer Q scratch（S×H×2B）、阶段AttentionOut/residual/FFN scratch（由生命周期allocator复用）。hidden整prompt驻HBM而**当前query block驻片上**：4K prompt的一份hidden=32MiB，不能沿用decode“hidden全部片上零往返”的声明。

M32 block建议片上预算：hidden input/output/residual各256KiB；Norm input256KiB可复用；Q256KiB、K/V各64KiB；AttentionOut256KiB可与Q生命周期复用；gate/up/activated各768KiB（F12288），峰值约2.25MiB；PrefillTotal16KiB、4qh online64KiB，加共享KV1MiB与FIFO。物理峰值通过lifetime图求，不直接累加全部buffer；粗略至少数MiB，远大于decode约1.4MiB。最大F19200时单FFN block1.172MiB，三份3.516MiB；最大D256在线状态翻倍。

两个可实现选项：
- `PREFILL_LOCAL32`：上述中间块驻片上，对目标URAM/BRAM资源进行综合；运行时不存在全S向量onchip假设。
- `PREFILL_SPILL`：将gate/up/activated写HBM scratch，保留小tile缓存，支持更大F或资源不足。宏指令保持相同语义，额外DMA和带宽明示计数。

这是显式能力配置；不能无声切换导致性能与资源报告不可重复。v1至少实现一种并通过最大宣称shape测试。HBM总容量要加入prefill scratch；maxContext和maxPrompt长度由region allocator实际剩余容量决定。

### 14.6 Prefill描述符与接口补充

GEMM_W4字段 M=query有效数、N=output、K=input、T=0；src0=二维X descriptor、src1=W、dst=二维Y；stride0Bytes必须表达token行stride。ATTN_PREFILL的M=Bq valid、N=Nq、K=D、T=当前可见key总长；aux0=queryStart、aux1=layerId；extension descriptor包含T0、queryPositionBase、kvBaseRef、head counts、KV region stride、scratch refs与mask policy。extension格式需assembler/RTL统一固定64B版本，未支持版本拒绝。

新增ActivationOperand：4×8 BF16及queryMask4；权重操作数仍逻辑128×8，但tile之间广播共享。MpuResult增加queryIdx及rowIdx，VectorStore写地址为base+queryIdx*stride0+rowIdx*elementBytes，不是扁平decode row编号。

KV_APPEND_BLOCK接收[M,Nkv,D] K/V，写token=T0+queryStart+localQ；每token K/V不交错错误，block completion必须收齐全部写B。EMBED_BLOCK输入prompt tokenId数组（INT32），每个token检查id<V；所有读取及输出地址越界检查。

### 14.7 性能模型与必选验收

Linear prefill每层MAC=`S*192,937,984`（当前目标尺寸）；attention总MAC=`2*Nq*D*(S*T0+S*(S+1)/2)`；final lm_head通常只执行一次。实际MPU循环按tile补齐与未来tile跳过计算，HBM traffic按实际weight复用和scratch读写计算。

prefill性能指标独立报告：TTFT（包含embedding、所有layer、final norm/lm_head）、prompt tokens/s、peak HBM/片上用量。decode20tok/s目标不能推出TTFT目标；用户未给出TTFT门限，本稿先以功能完整与报告数据为验收，不捏造数值保证。

必测S=1/31/32/33/127/128/129、query block和key tile边界、已有T0非0、最后query只见合法prefix、prefill后第一decode。比较相同token序列的“一次完整prefill”与“分块/增量prefill”，按同一数值模型解释tile顺序可能导致的FP差异；若硬件保持相同key tile分界应逐bit一致，否则固定容差并记录分界。重排prompt future token只允许影响其自身及之后query，不能影响之前query输出。

v1放行要求：prompt输入→prefill KV→greedy next token→至少多步decode全链路通过，真实checkpoint数值验证通过，prefill和decode各自的cycle/resource证据存在。主机预生成KV仅为调试手段，不能代替prefill验收。

## 15. 交付与环境状态

本spec保存于仓库外，不修改open_tc源代码、Makefile或测试。独立C++17 TensorCore构建与现有FP8_E5M2测试已通过（8jobs、19cycles）；SpinalHDL工程、工具链安装与本NPU RTL验证尚未执行。环境草稿已保存C++ install_script/start_skill，并新增huggingface.co允许域名以读取checkpoint metadata；保存不代表当前实例已应用或环境已发布。当前config读取失败是代理403网络访问限制，不是已确认checkpoint私有或需要token。无需在聊天提供凭据；可在环境设置保存网络变更后重试。

### 附录A：延迟scale不能保证输出<1 ulp的反例

固定seed=7生成的8项（可补零至group128），scale为FP16可表示值 `0.09783935546875`，q=`[-4, 4, -7, -6, -5, 3, -7, -2]`，BF16 x=`[-1.8515625, -0.265625, -1.71875, -1.640625, -0.302734375, 1.3046875, -1.5078125, -1.109375]`。采用逐项FP32非FMA顺序累加：先BF16(q·s)再乘x得到 `4.533195495605469`；先sum(q·x)再乘s得到 `4.5416951179504395`；输出FP32 ulp约 `4.76837158203125e-07`，差值为 `17825` ulp。此例只反驳通用等效保证；不同归约树的实际误差需按本文bit reference重新计算。

## 16. 最新结构复核（优先于早期规模估计）

用户已接受W4共享激活解码、多路小整数产品路径，保持1024产品/拍作为首版基线，BF16通用路径用于QK/PV/lm_head。VPU当前选择32物理lane，256元素为逻辑块而非256完整ALU；旧§7/§8/§14按256物理位置计算的吞吐数字不适用于此基线。详细资源/阶段算术、瓶颈与候选优化见同目录npu_v1_bottleneck_review.md。32lane规模仍需有限buffer cycle模型验证。

配置读取限制已解除，本次已保存checkpoint_config.json；先前网络403说明是历史状态。已确认的模型算子不需要扩展到未给出的25项通用清单。实际权重导出、static act-order处理、scale格式与端到端精度仍未验证。建议优先融合QK出口max、PV出口state合并、独立attention bank并比较2/3 KV slot；scale 8lane满负荷，16lane为综合候选。所有优化尚为架构建议，不是已验证硬件。

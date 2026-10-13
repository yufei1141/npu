# 07 宏指令、接口与裸金属软件

## 1. 顶层与能力

NpuTop=DCU+MPU+RPU+VPU/ConvUnit+DMA+bankedStores+boardShell。A53裸金属负责tokenizer、模型/manifest加载、tensor地址分配、position系数、任务提交、输出token解码、错误/状态snapshot。主要计算在PL；不将A53算GDN当已实现NPU兼容。

能力寄存器明确numMode/actFormat/groupSizes、W4/A16product数量、RPU lane/headDim上限、VPU/SFU能力、FP32mode、maxDescriptorCounts、clockprofilehash。支持其他约1B模型依赖其算子与shape，不能按参数量宣称兼容任意1B。普通Qwen3可关闭RPU并用全部层KV；LFM卷积/状态规则不同，不在当前默认支持名单。

## 2. CSR建议ABI

offset0x000 VERSION，0x004 CAPABILITIES，0x008 CONTROL START/ABORT，0x00C STATUS，0x010 IRQ W1C；0x020/024 PROGRAM_BASE64，0x028 PROGRAM_COUNT；0x030/034 TENSOR_TABLE_BASE64，0x038 TENSOR_COUNT；0x040/044 MANIFEST_HASH_REF；0x050 CONTEXT_LEN，0x054 POSITION，0x058 TOKEN_IN，0x05C TOKEN_OUT；0x060 EPOCH，0x064 FIRST_FAULT，0x068 FAULT_TAG，0x06C FAULT_ACTOR；0x080/084 CYCLES64，0x090 COUNTER_SELECTOR/DATA。所有shadow配置在START原子锁存，busy期间不可修改active表。

board address base不在spec硬编码，由Vivado地址编辑器+BSP header导出。DMA采用64bit physical地址，baremetal uintptr_t须AArch64或显式64bit，不截断高DDR地址。START前range与spacepermissionvalidator必须完成。

## 3. 64B宏指令格式

| byte | 字段 |
|---|---|
| 0 /1 /2..3 | opcode8 /flags8 /cmdTag16 |
| 4..11 | src0/src1/src2/dst，各TensorId16 |
| 12..19 | M/N/K/segment，各16 |
| 20..23 /24..27 | aux0/aux1，layer/head/position或扩展索引 |
| 28..31 /32..35 | scalar0/scalar1，FP32位模式或op整数 |
| 36..39 /40..43 | wait0/wait1，event slot8+generation24 |
| 44..47 | signalEvent |
| 48..51 | extensionId |
| 52..55 | repeatCount（仅loop opcode） |
| 56..63 | reserved0 |

M/N/K字段为op定义的tile或普通layer尺寸，不承载248094词表总长。HEAD_ARGMAX从TensorDesc.dim0读取32bit vocab，N表示输出microtile（如32）；词id及rowIndex32bit，禁止将vocab截断进N16。其他op按真实shape校验，0/越界不得隐式变成65536。

该ABI与旧Qwen3-8B字节布局不同，VERSION/manifest必须区分；不同时解析。little-endian固定。opcode：EMBED=01、LINEAR_W4=02、LINEAR_A16=03、NORM=04、CONV4_SILU=05、GDN_COEFF=06、GDN_STATE=07、NORM_GATE=08、ROPE_PARTIAL=09、KV_APPEND=0A、ATTN_FULL_GATE=0B、SWIGLU=0C、ADD=0D、HEAD_ARGMAX=0E、DMA=0F、FENCE=10、TOKEN_COMMIT=11、LOOP_BEGIN/END=12/13、PREFILL_BLOCK=14。

op只定义必要算子，不任意通用微码；mode/shape通过tensor及ext解析。GDN_STATE内部展开headloop，两遍RPU并DMA；PREFILL_BLOCK是编译器宏或可选硬件controller，首版compiler可展开有限任务，不声称软件只写一条就有实现。

## 4. TensorDesc与GDN扩展

TensorDesc64B：byte0..7 base64；8..15 bytes64；16..27 dims[3]32；28..39 strides[3]32；40 format8；41 space8；42 layout8；43 flags8；44..51 auxBase64；52..55 auxStride32；56..63 reserved64=0。dtype枚举INT4_SIGNED、FP16、BF16、FP32、UINT32；space=PS_DDR/PL_DDR/LOCAL_*。bytes64允许描述大region，但addrvalidator检查实际区域，descriptor不自动证明内存存在。

GdnExt64B：schema16,layer16,heads16,Dk16,Dv16,queryCount16,stateOldId16,stateNewId16,convHistoryId16,qId16,kId16,vId16,zId16,betaId16,decayId16,outId16,tokenVersion32,flags32,eps32,invSqrtD32,snapshotLayout32,reserved[12]byte（共64，pack检查）；ext含全部权重与data引用时不够则用TensorTable连锁引用，不挤占reserved复用。

## 5. 调度与事件

DCU顺序dispatch当前PC，等待显式events、unit queue、banklock、slot及outputcredit；actor内部有限pipeline。事件64slot，generation24，与epoch16一起避免复用误满足。Completion必须destinationACK、statewriteB、slotrelease符合op定义；错误事件不使依赖success。

普通VPU/MPU独立queue4；RPU2head；DMA各class8；completion16初值须按最大inflight证明。head-of-line等待会阻挡后续独立prefetch，compiler先放read任务。仅Acu可有限lookahead QK，不全面乱序。

## 6. 裸金属cache与提交

权重/model加载后flush cache到DDR再DMA；CPU写descriptor/prompt也flush。CPU读DMA写结果前invalidate且memorybarrier。Noncoherent HP通路不能因为A53裸金属就忽略cache；HPC coherent配置需真实验证snoop/shareability，不自动启用。

cacheline按BSP定义（常见64B，待平台确认），buffer对齐并避免CPU/DMA同时修改同一line。doorbell前DSB类屏障由BSP封装；结果IRQ/poll后再invalidate。软件不能在START到DONE间释放或移动descriptor/tensor。

模型可SD/FatFS或预载DDR，纯文本canonical文件约0.77GB，启动加载时间单列不计steadydecode。Tokenizer可host离线产生tokenIDs做第一阶段测试，后续A53实现完整tokenizer；不将hosttokenize模式当已实现自主裸金属聊天。

## 7. 错误与恢复

fault类别BAD_ABI/BAD_SHAPE/RANGE/FORMAT/GPTQ_UNSUPPORTED/BUS_R/BUS_B/NUMERIC/DEPENDENCY/TIMEOUT。ABORT停止新issue但排空不可撤销AXI/IP，过期epoch不会commit。

journal模式token写新state区、KV在未提交token位置写入，所有layer成功后TOKEN_COMMIT换state表并推进validLen；失败弃new区保持old表，重发同token从old重新执行。conv history也必须old/new版本或可从inputjournal恢复，不能仅state双区而conv已被覆盖。prefill大block提交按block/layer策略记录checkpoint，host明确恢复点。

top1greedy按A16logitsround后比较，tie最小id；thinking模式与chattemplate由software生成，PL不理解自然语言。测试先enable_thinking=false减少额外生成，但不是硬件限定。

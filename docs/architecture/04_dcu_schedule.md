# 04 DCU、macro展开与全局调度

## 1. 控制层次

DCU是全局宏指令调度器；MPU/VPU command controller展开本单元循环；Acu只展开attention。禁止DCU和Acu同时拥有同一online state或给MPU发相冲突job。

DCU部件：ProgramFetcher→DescriptorValidator→EventScoreboard→ResourceChecker→Dispatcher；独立CompletionCollector、FaultController、PerformanceMonitor。共享指令FIFO8、MPU4、VPU4、DMA每口8、KV4、completion16为初始参数；全部有限并计入模型。

宏指令512bit格式沿用总spec。微任务不使用宏指令ABI，通过内部typed Bundle；避免把20条宏指令膨胀为每lane软件控制。

## 2. 事件与scoreboard

eventRef=(slot8,generation24)，slot64；无事件0xFFFFFFFF。只允许程序中的已提交producer作为等待目标，不得等待之后取指的producer。等待事件匹配generation且state=DONE才ready；旧generation无效不能误满足。

producer在dispatch前分配事件及完成FIFO credit，后续UnitDone更新事件。完成需要该命令所有出口写入ack、所有子任务retired；有DMA write则等待B。ERROR依赖不是DONE-success，下游取消并返回DEPENDENCY_FAULT。

资源checker检查physical bankGroup读/写lock、slot generation、unit queue credit、result/return credit。普通命令锁粒度为buffer/tensor chunk，attention由Acu内部生命周期管理。读同buffer可共享但bank带宽仍仲裁；写独占直到writeAck，不以最后一个发射拍释放。

## 3. 顺序dispatch的作用与限制

首版只按PC顺序检查当前命令，可dispatch后推进，不等所有前序完成；不同单元独立执行且显式依赖。若队首等待，会阻塞后面的独立命令（head-of-line），这是可测性能成本。compiler将prefetch提早放到需要数据之前。

不增加任意乱序窗口，先保证无死锁。Acu可对不同KV tile QK进行有限lookahead，必须保留输出槽与旧o更新依赖；本单元性能优化不改变全局指令顺序语义。

## 4. decode层内事件图

```
Hidden[l] ──┬── RMSNorm_attn ── Q/K/V Linear
            │                       │
         residual                 Q/K Norm → RoPE Q/K
            │                       │
            │                KV_APPEND ← V
            │                       │ B响应
            │                ATTN (QK/P/scale/PV/merge)
            │                       │
            └────────────────── O Linear → ADD
                                          │
                                  RMSNorm_ffn
                                          │
                                  gate/up Linear
                                          │
                                        SwiGLU
                                          │
                                     down Linear
                                          │
                                  ADD → Hidden[l+1]
```

Q/K/V共MPU因此排队执行；VPU Q/K Norm可在其它projection占MPU时运行，但buffer需独立且producer完成该Q/K输出。首版各projection完整结果ready后发对应VPU，逐head更细的ready作为后续优化。

gate/up先形成成对tile，VPU可与后续output tiles overlap。down仍等完整activated ready。layer输出事件确认才允许下一layer残差buffer复用。finalNorm→BF16 lm_head→stream argmax→TOKEN_DONE；不存全vocab logits。

## 5. Acu首版与优化调度

首版模式SERIAL_TILE：有2KV slot，prefetch i+1并行于compute i；当前i完整state COMMIT后才发i+1 QK，支持max融合与PV merge但不跨tile state乱序。足够用于功能验证，不保证20tok/s。

候选LOOKAHEAD1：同时至多2个已发QK tile，MPU在oldest PV ready时优先PV，否则提前QK next。example QK0→QK1→PV0→QK2→PV1。KV slot释放在PV最后读完成，state slot完成在NewO ack后，两种完成不能混淆。下一tileexp必须等上一state版本commit。

2slot下tile2可能无法提前DMA，3slot候选容许tile0等PV、tile1等计算、tile2正在fill。全部slot占用则prefetch停，不能覆盖。score/p buffers深度随lookahead cap设为2tile，state保持一个committed+一个pending版本，不建立无限版本链。

模型若证明2slot仍足够可保持2；若选3，容量增加512KiB、对应物理bank与DMA目的slot编码必须更新。不是只给slotId多一位。

## 6. prefill程序与调度

layer-major：全部queryblocks生成该layerQ/K/V、Norm/RoPE、append并等待写可见；随后queryblock attention/O/FFN，写新hidden region。hidden HBM ping-pong；片上M32 working set。最终只最后prompt token执行lm_head。失败不推进global committedLen。

GEMM_W4 group-major内部保留M×Nr TotalStore，weight复用M；Acu prefill按queryBlock→kvHead→keyTile→4Qheads，保存4qh在线o64KiB（D128）及m/l；K/V使用统一bank视图。query/key因果mask逐元素，整未来tile允许跳过但记录省略计算计数。

普通vector算子不能跨不同token混合归约。norm lengthH对应每tokensegment；QK norm lengthD对应每token/head。RoPEposition=T0+queryIdx。appendBlock仅validquery写HBM。profile包括prefillspill与transpose，不沿用decode“hidden片上无HBM往返”。

## 7. 前向进展与故障

关键等待边：input credit→合法slot；output credit→已分配独立目的buffer；state dependency→前tile merge；DMA fill→已空闲KVslot。禁止writer持有唯一输出资源同时等待依赖该writer释放的下游。

ABORT：停止fetch/dispatch/newAR，停止新compute发射；已发AXI与无readyIP返回必须继续drain；内部出口切discard但不误写新epoch。记录第一个fault，汇总其余fault计数。drain完成后发送FAULT_DONE，软件才可START新epoch。硬件reset只在shell保证接口没有在飞事务时用，不用reset掩盖未完成AXI。

watchdog超时报告cmdTag、actor state、等待资源，不当作自动成功。性能归因至少包括WAIT_MPU、WAIT_VPU、WAIT_STATE、WAIT_KV_FILL、WAIT_BANK、WAIT_SCALE_RAW、WAIT_RESULT_CREDIT、WAIT_EVENT、HEAD_OF_LINE。每周期可多原因但主stall分类互斥，避免总和超过idle周期。

## 8. 实现门槛

静态compiler验证event DAG无环、tensor range/format/alias；仿真assert旧epoch不commit、completion一次、slot释放前reader0、bank每拍限额、结果buffer never overflow。检查最长保留output credit周期及公平仲裁，压力测试异步backpressure，区分可恢复waiting与deadlock。

## 9. ATTN扩展描述符64B（版本1）

byte0 schemaVersion16=1；2 flags16（bit0 prefill，bit1 causal，bit2 lookahead-capability，其他0）；4 layerId16；6 qHeadCount16；8 kvHeadCount16；10 headDim16；12 queryValid16；14 keyTileTokens16=128；16 queryAbsBase32；20 historyCommitted32；24 keyVisibleLength32；28 contextCapacity32；32 kvTensorId16；34 qTensorId16；36 outTensorId16；38 scoreTensorId16；40 probabilityTensorId16；42 onlineStateTensorId16；44 sinCosTensorId16；46 reserved16=0；48 scaleInvSqrtD FP32；52 maxQueryBlock16=32；54 reserved16=0；56 reserved64=0。

各tensorId引用统一64B tensor table，KV stride使用KV tensor layoutParameters+manifest的已验证映射，不在命令中重复造不同地址。flags不能用来跳过物理slot容量检查。historyCommitted及queryAbsBase为绝对token索引，keyVisibleLength含本次有效token；causal仍按pk≤pq逐元素mask。version不支持、queryValid>32、Qhead/KVhead不整除、headDim不支持时在dispatch前拒绝。硬件内部最大dim/query/head宽度由配置检查，不默许字段高bit被截断。

# 09 验证、SpinalHDL规划与放行条件

## 1. 四类reference分离

- 官方checkpoint runtime：固定modelrevision、Transformers/GPTQ加载版本、tokenizer/chattemplate、ACTdtype、纯文本输入。验证真实headalias、missing/unexpectedkeys、outputlogits和statecache。
- 高精度数学模型：明确GDN、conv、mask等公式，用于独立算法验证。
- bit-accurate硬件reference：记录Cpartial树、非FMA、FTZ、ACTfence、dequant/延迟scale、SFUapprox、RPUbank循环；用于RTL精确检查。
- finite-resource cycle model：队列、IP实际L/II、RAM端口、AXI混合traffic、状态版本和反压；用于优化，不替代上三者。

本次已有design_space分析和独立GDN/布局/ABI验证脚本，不是完整checkpoint运行或RTL。不要把模型脚本成功叫“模型部署成功”。

## 2. 本次必须有的架构检查

models/validate_recurrence.py：分head两遍state与直接公式compare，初始非零state、r/beta边界、tokenstream与block16拆分、lane分片、banklayoutroundtrip。NumPy算法测试容差不代表神经网络精度验收。

models/validate_layout_and_buffers.py：KV/state/W4地址唯一、每microstepbank无超限、64BABI尺寸、group slotoccupancy、feedbackC限制、tokenepoch新旧表。证明logicalmapping，不证明BRAM inference。

## 3. SpinalHDL模块组织

```
spinal-npu/
  NpuConfig / Capability / LatencyContract
  protocol/{MacroCmd,TensorDesc,TxnTag,Streams}
  dma/{Read,Write,Reorder,LocalMover,TransferTracker}
  memory/{WeightStore,PartialBank,VectorStore,KvStore,HeadStore}
  mpu/{W4Product,A16Product,AddFabric,GroupReduce,Total,DotHead}
  rpu/{SlotManager,StateScan,Decay,Delta,RankUpdate,ReadReduce,Commit}
  vpu/{MulAdd,Reducer,Sfu,Conv4,Norm,Gate,Rope}
  control/{Dcu,Events,Resources,Acu,Fault,Perf}
  board/{Zcu102Shell,PsAxi,ClockReset}
  sim/ + reference/ + tools/{exporter,packer,assembler}
```

工具版本未在本次安装或编造。创建工程时锁Scala/Spinal/sbt/JDK兼容版本；board synthesis用真实Vivado/BSP版本。FpWrapper声明ACT_FMT/FP32、L/II、rounding/FTZ/exception，BlackBox仿真必须有模型，不能空模块pass。

## 4. 分阶段交付和依赖

| 阶段 | 具体交付 | 放行证据 |
|---|---|---|
| P0 模型/格式 | 真checkpoint导出器、puretextmanifest、headalias、qzeros/g_idx、ACTdtype | 张量抽样decode与原loader一致，puretext一tokenreference成立 |
| P1 平台/带宽 | ZCU102裸金属CSR、DDRread/write/mixed DMA benchmark、cache维护 | 4/6/8GB/s等预算用实测替换，page/tail/response错误正确 |
| P2 算术/MPU小核 | 产品、FP32wrapper、interleavedpartial、group、DOThead | 非零测试数、special/rounding/随机ready，C随L变化，无overflow |
| P3 VPU/Conv | Norm1+w、L2、SiLUfence、softplus、convhistory、partialRoPE | peropreference、SFU全范围误差及corner，无head串扰 |
| P4 RPU | 一head→16head，两遍FP32state、三slotDMA、journalcommit | streaming/block同语义、longrunstate漂移、故障不污染old |
| P5 单layer | 一linear＋一full完整链及FFN | stage dumpcompare、stall分类、tailcausalmask |
| P6 全textdecode | 24layer、nativehead、greedy、tokenizer/runtime | 固定数据集精度验收，200MHz资源/时序、实际tokens/s |
| P7 fullprefill | M16projection、headmajor、scratch生命周期与stateKV导入导出 | prompt→prefill→多步decode，TTFT和RAM峰值实测 |
| P8 可选优化 | headW4、W4A8、更多lane、多cluster | 独立checkpoint/NUM_MODE与校准结果，不替换旧baseline |

P0/P1可并行进行，不要求等全部板卡编译才能做数学reference。当前没有外部板卡执行工具，因此P1/P6板级测量仍由实际环境/板卡完成；spec给出具体操作和计数器，不假装已测。

## 5. 数值验收

逐opmaxAbs/RMS/cosine、逐layer hidden/conv/qk/beta/r/o/state、最终logits与greedy一致率、长生成漂移及任务score。先冻结基准数据集/tokenizer/runtime/mode再定门限，不编造“所有top1必须100%”或输出永远<1ulp。

消融分别比较：ACTdtype、dequant/延迟scale、Cpartial树、SFUsoftplus误差、RPUFP32规约、onlineA16prob、headlogitA16fence；误差不能全归给W4。若delayedscale不通过，切exactdequant参考路径重新核算资源，不修改断言或dataset让通过。

## 6. 硬件/系统验收

所有Stream随机backpressure；任何不可撤销返回先预留credit。assert接受数=退休数+在飞；slot释放reader0；COMMIT仅一次；epoch过期不写；contextoverflow拒绝；A16/FP32bank写不可半字误覆盖。

综合报告资源分FP32 ADD/MUL、INT4decode、RAM actual blocks/register、LUT/DSP、criticalpaths；200MHz须post-route通过而非只综合estimated。多AXI端口共享DDR时reportpayloadBW和R/W混合效率。

性能分别报告cold model load、tokenizer、prefillTTFT、steadydecode@1K/4K/8K、outputhead、stateDMA、KV、VPU与queuewait。6.1tok/s等本次数字是预算，不是达标证据。长contextlinearstate固定、fullKV增长，按真实context重新建模。

## 7. 尚未完成的必要核验

真实tensorheader/loader；nativeheadalias；scale/nativeACT实际dtype；delayedscaleaccuracy；vendorFP L/II/FTZ；board PS/PL memorymap和payloadBW；RAMphysicalbankmapping；SFUcoeff误差与ROMports；完整finitequeue仿真；prefill完整数值/TTFT。公开RKLLM支持Qwen3.5不代表已经验证MiniCPM4.7模型或知道其内部RTL。

这套规划可直接用于细分实现任务，但不把设计文档的完整性当作硬件已可运行。

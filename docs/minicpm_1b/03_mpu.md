# 03 MPU：不同模式的数据流

## 1. 内部模块

```
MpuCmdQueue -> JobSequencer -> OperandFetcher
  -> W4Cluster(shared activation) / A16Product32
  -> Configurable FP32 Add Fabric
  -> PartialBank / GroupSnapshot
  -> GroupReduce -> ScaleApply -> TotalBank
  -> ResultCaster / Store / Argmax stream
```

模块Bundle使用01/07的共用TxnTag、mask、query/output/Kgroup/epoch。命令先分配destination/partial及所有不可撤销返回credit，不等待compute结束才找空buffer。MODE_W4_OS、MODE_A16_DOT、MODE_A16_ATTN不可混合在同一active epoch microjob；模式切换须drain。

## 2. W4输出并行

decode每拍x[k]广播给64输出行，读取64signed4 q=32B。FP16/BF16 x解码成尾数、指数、符号；局部cluster生成整数倍m/3m/5m/7m，再选择移位形成FP32产品。FP16有效尾数11bit、BF16为8bit，因此normalize宽度由ACT_FMT决定，不能使用BF16专用8bit逻辑处理FP16。

64产品各送FP32 ADD partial[row,k mod C]，C初始16，要求相同partial写回间隔≥Ladd；ADD为非FMA RNE。每group128产生C个部分和，每partial约128/C项。K尾部固定补0，validMask屏蔽实际输出行。

部分和配置R64×C16×4B=4KiB/slot，双slot8KiB。group结束最后一笔ADD退休后才Snapshot READY；GroupReader每拍读一row的16partial，15个ADD位置组成8/4/2/1树。64拍读完64row；规模不是原128tile独立DOT8树。

ScaleApply2lane、TotalBank2bank使group后处理有余量；reader输出1row/拍可至少1scale/拍。总和按group递增，首group隐式0但不读取未初始化RAM，RAW scoreboard阻止过期版本。每row group间隔128issue拍通常远大于Ltotal，但不替代实际proof。

## 3. group slot数量由占用时间计算

以Lproduct+Ladd=Lret、G128、R64读出拍为例，占用从first issue到最后ADD退休再读出：G+Lret+R。要每G拍发新group，slot≥ceil((G+Lret+R)/G)。示例Lret14，ceil206/128=2，双slot初步够用；若有stall或另一模式G缩小重新计算，不能硬编码2slot能满速。特别是PV的Ktile32、32outputRows、示例Lret14时，ceil((32+14+32)/32)=3，所以A16_ATTN首版至少3个microgroup snapshot credit（或明确stall）；不是W4的2slot配置。

input.ready除partial space还检查GroupReduce output credit、scale/total及ResultStore空间；各阶段stage FIFO都必须覆盖固定latency返回。Ladd>16时C提高到nextPow2(Ladd)或stall，不能读后write同一partial无forwarding地每拍累加。

## 4. weight布局

canonical W4_OS64_G128：输出rowBatch64、Kgroup128为一个block；128B FP16 scale（实际format若FP32则256B）、可选zero metadata、随后每k64nibble=32B，共4096B q。little-endian nibble low先，signed4二补码；有效q范围[-8,7]必须经GPTQ importer验证。

W4Store64个4bit slice，按k供给；scale cache保留64项FP32供group结果读取。prefetch当前/下一block双缓冲，HBM/DDR源读返回完成及pack完成后READY。不是从原GPTQ row-major随机读64个远地址。

GPTQ的packed qzeros可能有版本偏移，sym=True不代表serializedzero0；desc_act=false仍须检查g_idx映射。mapper生成canonical q=(packedQ-effectiveZ)，若超范围拒绝。不要静默裁剪权重或把AWQ非对称数据当对称。

## 5. native16输出head：DOT32

保持embedding/lm_head row-major，顺序读每row1024个16bit数=2048B。每拍32weight×32对应x产品，5层平衡ADD tree形成DOT32，再交错C个partial累加32step；最后Cpartial树归约FP32 logit。

DOT32所需31ADD和partial ADD从可配置Add Fabric复用，而不是在64lane W4 MAC旁额外复制另一套大树。实现时支持两套静态routing；timing/resource是否合理必须综合。同一cycle只一个mode，不抢占未退休ADD。head row的发射32拍，不能等其全部tree/partial drain后才发下一row；用多row context及返回credit保持连续64B/拍读取。

rowContext数量≥ceil((rowIssue+drain)/rowIssue)，按真实latency与partial readout计算；预留多row PartialBank。示例DOT32产品/树/partial返回54拍、末端C16归约32拍、rowIssue32，rowContext≥ceil((32+54+32)/32)=4；这是示例而非已测IP。默认4个row context，实际配置按latency自动验证。完成结果stream给argmax，不写248K完整logits，除debug显式开启。未找到lm_head独立tensor时由exporter真实loader证明alias，不能生成随机权重。

head若被额外量化：用独立W4 packed head，保留original embedding lookup不改变token语义；容量需包含必要副本。该能力NUM_MODE不同，必须校准后声明，不能认为当前GPTQ包含headW4。

## 6. 完整attention模式

A16_ATTN32采用输出并行32lane，K维每拍一项，partial interleaved处理feedback。QK：8qh×4token位置，D256依序；K由2kvh广播给4qh，避免四份重复cache。每32token tile有8个position microbatch，各256issue拍=2048。PV：4qh×8dim一个kvh组，两个kvh×32dimBatch，共64microbatch，每个32key step=2048issue拍。

softmax保留FP32 m/l/o，p格式A16，mask无效key0；原生框架softmax/p舍入次序需验收。VPU处理score scale1/sqrt256、max、exp、sum与state update；仅发P_READY后才能PV。sigmoid(qGate)乘attention结果再o_proj，不能省略。

## 7. prefill映射

W4_OS64改为4query×16output row，x4广播各16权重行；同group block64row在M16内复用。64输出行×16token总和4096B；inputX可缓存M×H=32KiB，FFN输入M×F=112KiB。query microbatch、output subrow按固定顺序遍历，tag持有(query,row)。常规MPU64产品总吞吐不因weight复用自动增加。

如增加mesh模式，另能力位和测试，不使用同一cmd silently改变累加顺序。每NUM_MODE应具有明确bit-accurate reference。

## 8. 数值门槛

NUM_W4_DELAYED为候选：先sum(q*x)再scale，与先A16(q*scale)再乘x不位等价。生产支持必须通过09端到端验收；无通过证据时不得宣称checkpoint精度兼容。

fallback探索：group setup生成16可能q的A16反量化表，每lane按q选择，再走A16 product；或offline显式table。它保留dequant rounding但增加乘法/表端口/metadata流量，性能需要重新建模。不能通过调整reference掩盖数值变化。

ACT_FMT为FP16/BF16，scale_fp32内部统一；dtype以真实tensor/header/loader为准。FTZ、special值、rounding fence、非FMA规则进入manifest，scalar/matrix同时匹配。

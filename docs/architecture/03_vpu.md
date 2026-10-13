# 03 VPU、SFU与Attention控制合同

## 1. 层次和资源

```
VpuTop
  CmdQueue(4) / VpuOpController
  VectorReadEngine / OperandAligner
  Mul32 / AddCompare32
  SegmentReducer32
  Sfu32(op=EXP/RECIP/RSQRT)
  ScalarUnit / ScalarRegisterFile
  VectorWriteEngine / CompletionTracker
Acu (Attention Control Unit)
  ScoreMaxCollector
  AttentionTileController
  OnlineStateManager(m,l,o版本)
  ProbabilityBuilder
  OldOScaler / PvResultMerger
```

Acu是ATTN宏操作内部控制器，使用MPU QK/PV与VPU资源，不额外创造第二计算阵列。不同actor可分别请求MUL/ADD/SFU/reducer，资源独立；一个actor不得占着某资源等待另一资源无限ready。请求队列至少2项，发射前同时预留所需输出/旁路buffer。

默认一个普通VpuCmd active；Acu内部tasks可在不同资源上重叠。attention active时普通Norm/SwiGLU不与其混行，避免首版过度仲裁；DMA与MPU仍可在物理bank独立时并发。

## 2. VectorRead与OperandAligner

每beat32FP32，BF16读转换为FP32后参与。双源从不同组同时读，跨组tag必须匹配，读延迟独立时用join FIFO，不按当前地址猜配对。SwiGLU的gate/up保留在chunk scratch，不能经多条SFU路径后丢失原gate。

初版chunk=256元素，每命令拆成chunk；scratch各256FP32 gate/up/exp/tmp/dst（按生命期复用），双chunk允许读/写与计算交叠。SFU请求tag含chunkId、beat0..7，返回不得跨命令配错；elementIdx始终指原向量位置。

## 3. SFU实现接口与范围合同

每SFUReq含32value、mask、op、tag；32Resp同字段加exception。内部统一入口，但不能宣称EXP/RECIP/RSQRT互不占用：首版shared request capacity32值/拍，op切换支持每beat或必须drain由实现配置明确。baseline可先每阶段drain后切op，模型计入额外L。

EXP采用base2范围规约与分段多项式：k=round(x/log(2))，r=x-k*log(2)，从r segment取coeff，Horner求值，exponent adjust 2^k。RECIP规范化m∈[1,2)、查分段初值/多项式、指数取负，可用一次Newton修正但明确是否FMA；RSQRT处理指数奇偶，归约m区间并查表/修正。每函数有独立coeff ROM，系数版本hash及误差报告。

尚未选择系数分段数、阶数与定点/FP实现，不伪造确定误差或DSP数量。实施前给出bit reference，覆盖所有区间端点及常用值。单lane prototype先完成，再32lane复制/共享ROM；ROM读端口也可能瓶颈，不能单ROM1read供32个不同index。

domain：EXP attention x≤0，x<-80显式输出0；sigmoid exp输入=-abs(x)，同范围。RECIP接受有限正数，softmax l>0；RSQRT接受有限正数，norm mean+eps>0。正常Qwen路径不需要tanh/GELU，首版不实现。非法输入sticky fault；NaN/Inf遵循manifest运行策略。全mask head不送非法SFU输入。

## 4. 稳定sigmoid与SwiGLU

按原gate符号执行：a=exp(-abs(gate))，r=reciprocal(1+a)，sigmoid=gate≥0 ? r : a*r。gate<0时需要额外MUL，初稿“固定2MUL/元素”低估稳定形式在负值处的工作量；可用sigmoid负支1-r减少MUL但大负值抵消不稳定，不作为默认。

统一可固定算 `negValue=a*r`（32lane MUL），再select得到sigmoid；随后gate*sigmoid、再*up。这样3MUL/元素（正值也可mask第一乘但不改变固定beat数），2SFU操作、1ADD/元素，输出最后才BF16。参考与GPU具体BF16中间边界差异仍需评估。

OpController每chunk：READ_GATE_UP→ABS_NEG→EXP→ADD_ONE→RECIP→MUL_NEG_BRANCH→SELECT→MUL_GATE→MUL_UP→WRITE。所有阶段每8beat，latency drain显式；可后续改成多chunk resource pipeline，不改变数值顺序。

资源下界M1F12288：SFU768请求拍、MUL1152请求拍、ADD384拍，max≥1152拍；阶段发射和2304拍=11.52µs（不含读写、select、latency）。依然比down49152issue小得多，但不再沿用之前768拍总下界。prefillM32各项×32。

## 5. SegmentReducer32具体实现

32输入平衡树5层，SUM每节点FP32 ADD，MAX/ARGMAX为compare/select；tag/index随之传递，argmax相同value选最小originalIndex。各op pipeline不能用不同延迟却只共一条固定tag delay。

segmentLength128（QK norm）有4个32输入blockSum；首版缓存4partial并用2层树归约，不用每拍反馈到一个sum寄存。D256为8partial+3层树。H4096全局norm有128partial：分组保存、树归约，或交错context分批归约且顺序固定。首版按显式partial RAM和静态树/时分遍历做，mean计算前所有partial must retire。

浮点SUM次序：每32块树归约，再各块partial平衡树（不足2次幂补0）。RMSNorm x² MUL逐项FP32。接近0及eps由ScalarUnit添加；square_sum/M的FP32 reciprocal constant由软件manifest提供并校验，避免SFU动态除法占用。

max可在score退休时串行比较每head provisionalMax，compare1cycle旁路处理同head相邻beat；未知FP compare latency则scoreboard stall或先存blockMax后树归约。流式max与score写入fork必须两个出口都拥有credit。

## 6. Norm/RoPE/ADD流程

RMSNorm FSM：READ_SQUARE_REDUCE→WAIT_SEGMENTS→SCALAR_MEAN_EPS→RSQRT→READ_X_GAMMA→MUL_R→MUL_GAMMA→CAST_WRITE→DONE。每segment独立r；Q/K Norm gamma index按模型权重（head_dim128）复用，hidden norm gamma长度H。初版32lane同一MUL array两次缩放意味着第二遍2ceil(elements/32)MUL请求拍，不能按一遍128拍算完全部norm。

RoPE：每beat16pair（32scalar input）可用32MUL array分两拍计算4product/pair，ADD array做two sums；目标每拍最多8pair等价32product，5120Q/K scalar=2560pair需至少320MUL请求拍。NeoX pair地址i与i+D/2，sin/cos按position和i；Q/K用相同position不共享x。pair layout需要read gather逻辑或软件/局部RAM排列，首版每head缓存D元素再顺序pair读。

ADD residual采用FP32(x)+FP32(residual)后BF16，保证host reference相同。支持同索引dst=src0时，read后write顺序不会覆盖未读索引；跨索引alias拒绝。linear outFP32如果要直接融合residual，改变BF16中间边界，初版不默认融合。

## 7. 在线attention事务

每stateKey=(epoch,layer,query,head)，m/l/o和version。一个tile事务tid：

1. ScoreMaxCollector保存所有有效scores与tileMax。
2. 确认该stateKey的prevVersion已commit；计算mNew=max(mOld,tileMax)，alpha=lOld0?0:exp(mOld-mNew)。
3. 概率SFU EXP(score-mNew)并fork：FP32 p送SUM得到tileL；BF16 p存ProbabilityStore。subtract使用ADD array，不是零成本。tileL与lOld*alpha+tileL形成lNew。
4. MUL缩放全部old o到ScaledO；alpha0可直接标ScaledOzero，但同样保证读有效标志。
5. p写可见、scaled-o ready后发PV。PV可以在scale还未完全完成时按chunk消费，但初版只在全部ready后发，减少credit环。
6. PV FP32 result按head/query/dim与ScaledO对应ADD，写NewO；最后writeAck后提交mNew/lNew及swap o version。
7. 所有tile结束，RECIP(l)再MUL(o)→BF16 AttentionOut。

mNew/lNew在o完成前属于pending版本，不能提供给下一tile state更新。下一tileQK/score/max可提前，概率及alpha使用哪版m/l必须由dependency决定。

pFP32用于l、pBF16用于PV是当前候选数值模式，输出与纯FP32softmax并非位等价。mask probability=0，全部masked tile直接pass stateVersion，不做 -Inf减法。所有head完成与版本提交是tile完成事件，不用一个head先完成代表整tile可复用。

## 8. 状态机及资源请求

AttentionTileController：WAIT_KV→QK_QUEUED→QK_RUNNING→WAIT_SCORE_MAX→WAIT_PREV_STATE→BUILD_P_ALPHA→WAIT_P_SCALE→PV_QUEUED→PV_RUNNING→MERGE_DRAIN→COMMIT→RELEASE。

普通VPU command使用resource arbiter；Acu task拆为SUB_SCORE、EXP_P、EXP_ALPHA、MUL_OLD_O、ADD_L、MERGE_PV、FINAL_RECIP、FINAL_MUL。优先完成已发PV merge与标量依赖，再新tile probability，再普通任务；带age上限，不能永久饿死已预留buffer事务。

QK输出scale(1/sqrtD)占MUL，概率subtract占ADD，EXP/sum占SFU/reducer；必须纳入真实调度。前次128拍exp估算不包括subtract/MUL/tag drain，模型中明确作为简化，不拿来直接证明吞吐。

## 9. 验收

正常/极值SFU及coeff边界；sigmoid大负值不能因1-r抵消为错误0；SwiGLU最终舍入边界；Norm segment无串扰、归约顺序匹配；RoPE pairing/position；attention m持续抬升需缩放o，每state version按顺序commit，causal尾mask；随机ready及不同SFU latency无错位；ScalarUnit在vector满负荷下有界完成。

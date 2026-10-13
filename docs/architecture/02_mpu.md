# 02 MPU 计算模块实现合同

## 1. 层次与调度范围

```
MpuTop
  MpuCommandQueue(depth4)
  MpuJobSequencer                 // macro→有限microjob
  OperandFetcher / Broadcast      // RAM读、mask、Kgroup
  W4ProductCluster ×16            // 每cluster8tile×8产品
  Bf16ProductCluster ×16
  Dot8Tree ×128
  GroupCollector / GroupStore(two slots)
  GroupReducePipeline             // 8行/拍，64/32/16/8 ADD
  ScaleApply(8lane)
  TotalAccumulator(8bank)
  ResultEmitter / ResultStore
  CompletionTracker
```

W4与BF16产品资源是否共享物理乘法器待综合。首版接口两模式显式，未激活模式clock enable关闭；不要仅用一个FP BlackBox宣称支持多路INT4。

只允许一macro active，但内部最多2group收集、16个groupBatch归约/scale在飞；容量由实际credit限制。prefill工作集不同，tag带query/row，状态不会因mode切换残留。新命令必须等上一命令所有返回/写确认drained，除显式内部lookahead执行的独立attention job。

## 2. W4产品模块

输入BF16 x与signed4 q；输出FP32精确产品（在合同浮点mode范围内）。分类x为zero/normal/subnormal/Inf/NaN。normal有效尾数m为8bit，符号xor；共享算出m、3m、5m、7m，按q magnitude选择，2/4/6/8使用shift或指数修正，q0输出规定signed zero。有效系数product最多11bit（m×7），×8等效shift；保留所有bits再normalize，不BF16中途舍入。

decode每K lane x共享给128输出行，prefill每query/K x共享给32输出行；局部cluster复制m系数生成，避免一条高fanout网络跨128tile。共享系数只对应同一个x，不能在4query间错误共享。系数选择结果+exponent+sign产生FP32，exception跟随tag。mask lane直接+0并不引异常。

对subnormal做前导零归一化；FTZ仅在配置FP_MODE_FTZ且reference同模式时使用。special q0×Inf按IEEE invalid→canonicalNaN而非无条件0，除该lane被mask。默认有限输入权重，NaN/Inf sticky fault由CompletionTracker收集。

## 3. BF16产品与DOT8

BF16×BF16有效尾数8×8→16bit精确产品，指数加与归一化；FP32可容纳正常产品，但溢出/下溢依IEEE舍入合同。vendor乘法器可以实现此逻辑，但其latency/II须测量。

DOT8固定次序：a0=p0+p1,a1=p2+p3,a2=p4+p5,a3=p6+p7；b0=a0+a1,b1=a2+a3；dot=b0+b1。每次FP32 RNE、非FMA。逻辑3层，延迟3Ladd，II目标1；IP输出不能被任意停顿，input credit覆盖全部3层在飞返回。

ProductModeWrapper两模式latency可不同，用tag-valid delay参数化；串行mode切换等待drain。DotBeat包含128行mask，无效行不写，部分K填零但固定树次序。

## 4. group16归约的具体硬件

GroupStore READY后，reader顺序读取cluster0..15，每拍输出8行、每行16个DOT结果。归约次序固定：第一层相邻DOT0+1、2+3...，第二层相邻第一层...，最后形成16项平衡树。

每阶段吞吐8行/拍，adders总数：8row×8=64；8×4=32；8×2=16；8×1=8，共120 FP32 ADD位置，延迟4Ladd。stage间向量valid与tag延迟匹配。reader发16拍即可覆盖128行，在下一group间隔16拍时该pipeline平均吞吐足够。

这比每tile独立15ADD树（1920ADD位置）面积小，但GroupStore宽读与120ADD仍需综合，不能隐藏在“fabric累加约450LUT”数字里。

flow控制以最终GroupBatch FIFO credit约束reader，未预留足够返回位置不读。每slot reader开始前预留16batch输出容量，简单首版GroupBatch FIFO≥32batch；slot读完可释放。若缩小FIFO改逐batch credit，需保证最大在飞4Ladd及scale backpressure下不溢出。

## 5. scale与TotalStore事务

8row GroupBatch请求读取8scale，scale由row/group映射到FP16或manifest声明格式，转换为FP32。FP16 scale语义尚待checkpoint tensor验证；内部采用FP32 scale cache避免转换成为每拍瓶颈。对应W4 block128 FP16 scale=256B，prefill复用相同row的scale但query不同。

顺序：READ_SCALE→SCALE_MUL→TOTAL_READ→TOTAL_ADD→TOTAL_WRITE。可pipeline不同地址，每stage独立FIFO。第一次group不是读未初始化RAM：totalValid按tile/epoch管理，直接sum（或规定+0）；后续必须读先前group完成的值。所有group按同logicalOutputId严格递增。

8bankTotalStore每拍8row地址必须覆盖8distinct banks；prefill microjob row顺序选择连续8outputRows，同query，同样无冲突。对于散射attention结果scaleMode=NONE，走result旁路，不在这里累加group延迟scale。

Ltotal>同行相邻更新间隔16时，scoreboard阻止下一更新，产生STALL_TOTAL_RAW。参数断言不能把未知Ltotal假设成1。16lane scale是可选后续配置，需要groupReader改16行/拍与树128/64/32/16、TotalStore16bank同时配套，不能只改乘法数量。

## 6. 模式映射

### W4 decode

输出tile128行，Kgroup128，16step；每step8x广播、128×8q。完成group后的scale与下一group compute重叠。命令总和最后group写ack后，ResultEmitter按32行/beat转BF16；输出tile写完成后推进nBase。

### W4 prefill

当前M≤32、Nr128；group循环外层，query microbatch4与row microtile32内层。tile=qt*32+nt。一个group有8query微批×4row微块=32microjobs，每microjob16step，GroupStore绑定microjob独立tag，不能上一microjob尚未退休就清写slot。

TotalStore至少M×Nr FP32，128个product输出对应4query×32row；GroupReader每拍8行不跨query边界，确保TotalStore bank无冲突。block weight驻留到32microjob全部accepted，下一group另slot预取。K输入可按group缓存M×128 BF16，X布局见prefill manifest。

### QK decode

tile=qh*4+positionLane，positionLane0..3；每head独立Q8，K由kvHead=floor(qh/4)广播。positionBatch0..31，dimStep0..15，最后形成128个score。输出tag含每元素head/position，跨qh有效长度尾mask独立。

QK通常D128，一个group不乘W4 scale，但最终乘1/sqrt(D)。用32laneVPU MUL或独立result32lane multiplier，硬件资源必须明确。首版使用VPU MUL并受其仲裁；当进行alpha/o scale时会争用，cycle模型需计入，不能称完全独立QK输出。

### PV decode（修正旧连续row映射）

tile=qhWithinGroup*32+dimLane，qhWithinGroup0..3。外层kvHead0..7、dimBatch0..3；8token一step，16step形成128输出。V同kvHead/32dim/8token共享给4Qheads，p各qh独立。输出scatter至head=kvHead*4+qhWithinGroup，dim=dimBatch*32+dimLane。结果不是连续128flat行。

### QK/PV prefill

QK4query×32key×8dim，PV4query×32dim×8key；一个qh各需8query微批×4key/dim微块×16step，输出mask含query与causal key。QK score维度[q,key]，PV result维度[q,dim]。每个job是与onlineState事务关联的微命令，按Acu控制依赖提交。

## 7. MpuJobSequencer FSM

IDLE→VALIDATE→ALLOC_TILE_STATE→WAIT_OPERANDS→RESERVE_GROUP→ISSUE16→WAIT_DOT_RETIRE→NEXT_MICRO/GROUP→WAIT_TOTAL_ACK→EMIT_RESULT→WAIT_DST_ACK→DONE。

允许pipeline overlap的条件由两个独立actor实现：issueActor拥有当前compute slot，reduceActor拥有READY slot；外层FSM只检查credit和最后totalACK。首个实现可采用串行模式作为功能baseline；其bubble必须由计数器真实报告，不以issue16替代completion。

counter只在fire增加。发射前确保每条不可撤销的product/dot返回能进入已保留slot。所有counter最终值、lastStep与退休计数对齐。异常中止stopIssue→drain→faultDone；不可撤销返回落到discard sink，不向下游可见的tensor写新结果。

## 8. 验收

单产品16种q×BF16边界/random与reference匹配；DOT8/group16固定次序逐bit匹配；groupStore所有slot/step/cluster覆盖。N127/128/129、K127/128/129固定补齐，prefillM1/4/5/31/32，总和groupIdx恰好一次、RAW scoreboard不读旧值。随机ready、不同IP latency、Ltotal8/16/20（20必须stall）测试无丢数。资源报告区分1024产品、896DOT ADD、120group ADD、8scale MUL、8total ADD及memory。

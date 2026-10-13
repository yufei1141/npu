# 01 协议、存储与生命周期合同

## 1. 共用Bundle

Spinal方向由master/slave确定，字段类型为Bits/UInt；浮点数据用Bits，不用SInt做数值运算。宽度随NpuConfig生成，禁止重复硬编码。

| Bundle | 必需字段 |
|---|---|
| TxnTag | epoch16,cmdTag16,jobId16,queryIdx16,headIdx8,tileIdx16,groupIdx16 |
| MpuCmd | tag,kind[W4_LINEAR/BF16_LINEAR/QK/PV],M16,N16,K16,srcX16,srcW16,dst16,rowBase16,queryBase16,keyBase16,scaleMode,outFormat |
| VpuCmd | tag,op,src0/1/2/dst16,elements32,segmentLength16,eps32,scalarRef16,outFormat |
| DmaCmd | tag,addr64,bytes32,targetSpace,targetSlot8,layout,writeFlag |
| OperandBeat | tag,step4,validTiles128,validProducts[128][8],x/w operands,first,last |
| DotBeat | tag,step4,values[128]FP32,mask128,lastStep |
| GroupBatch | tag,rowIds[8],queryIds[8],values[8]FP32,mask8,lastBatch |
| VecBeat | tag,baseIndex32,values[32]FP32,mask32,segmentLast,commandLast |
| SegmentResult | tag,segmentId16,value32,argIndex32,valid |
| UnitDone | epoch16,cmdTag16,status,errorCode,completedElements32 |
| SlotRef | slotId8,generation24 |

OperandBeat逻辑宽度不意味着顶层把所有字段做成长距离总线：操作数分16个8tile cluster本地布线，tag按cluster复制。DotBeat也按cluster传输，GroupCollector聚合只对退休索引做一致性检查。

mask每元素有效；无效乘积取+0且不引入异常，invalid output不写。lastStep用于Kgroup内部，commandLast用于整个命令；两者不能混用。尾组仍以16step标准格式补零，首版可保持固定16发射拍避免不明确的归约树次序。

## 2. 握手及无ready IP

Stream.valid&&ready唯一接受事件。stall期间payload与valid稳定。Flow仅允许接收端已经预留空间的固定延迟返回；禁止直接把BlackBox无ready输出当普通可停顿Stream。

CreditFpAdapter状态：outputCapacity=C，reserved=accepted-notConsumed；input.ready需reserved<C且IP允许新输入。输入fire增加reserved，输出消费者fire减少。latency流水只移动valid/token，在IP无clock-enable时不能整体freeze。返回先入不会满的FIFO；同时入/出时计数按两者差更新。多路径join每个branch都预留，fork不能只接受一支。

RAM read采用请求预留response FIFO slot，固定Lram后返回；请求tag与read enable同步。初版in-flight max有限，不允许无界Mem读后全局阻塞。

## 3. MPU GroupStore：16DOT/行的真实存储

128行×16step×4B=8192B/slot，2slot=16KiB；prefill存当前128 tile microjob的group结果，query/output ID与slot metadata一起保存。早期仅保存128个groupSum的buffer不能替代该空间。

逻辑布局：16个cluster，每cluster8行；每slot包含[step0..15][cluster0..15]，每word=8×FP32=256bit。收到DotBeat时，所有cluster把8个结果写当前step；读group时每拍选择一个cluster，读取其全部16step，输出8行×16DOT=4096bit。

首版用寄存器/分布式RAM实现逻辑矩阵，允许写正在收集的slot、读已经READY的另一slot。若映射BRAM，必须逐项证明16step并行读与128行同拍写，不能把16KiB容量直接折算为一个单端口RAM。拆成大量窄bank导致的器件碎片须在综合报告中列出。

slot状态FREE→COLLECT→READY→READING→FREE。allocator在group第一step前保留slot；最后一step退休后才READY。collector检查16个step各恰好一次、epoch/microjob一致，不接收另一group覆盖。reader发完16个8行batch即可释放GroupStore；后续树结果另有credit，不能等到total更新才释放读存储。

## 4. TotalStore与ResultStore

TotalStore按logicalOutputId mod8选择bank，bank各1R1W/拍。logicalOutputId decode=row；prefill=query*Nr+row。decode一outputTile128 FP32=512B，prefill M32×Nr128=4096 FP32=16KiB。

read response必须携带版本groupIdx；writeback前RAW scoreboard占用该地址。早期读旧值可由已完成的write bypass，但尚未产生的新sum不能“forward”。有hazard时stall scale input，credit传回group树。先采用显式scoreboard，不承诺时序无bubble。

ResultStore decode至少两个128FP32 capture slot=1KiB，向32元素口串化4拍。prefill支持从TotalStore分32元素读取，不要求4096元素同拍copy。QK结果capture存入ScoreStore并同时更新head max；PV capture交给state merge。slot保护到最后消费者fire，错误/abort也必须drain不可撤销返回。

## 5. VPU bank与物理端口

每向量组32bank，每bank一个FP32元素/word，bank=e mod32,row=floor(e/32)。BF16占低16bit；物理紧凑16bit模式可优化，但不与FP32相互别名。首版独立物理组：Hidden/Residual、QKV、FFN gate、FFN up、Activated、Score、Probability、OldO、ScaledO、NewO、Gamma/Scalars。

ADD src0/src1不同组时各32读+dst32写可满速；同组双源用READ_A→READ_B，实际II≥2。allocator必须把算子需要的并行组写入metadata，而不只给base。

Attention用OldO/ScaledO/NewO三个逻辑角色，Old/New双版本，Scaled独立。PV merge从scaled bank和incoming vector各取32值，用ADD后写NewO；一次tile的所有writes ack后换Old/New。任何head o版本不得半新半旧可见。

## 6. KV两种视图：给出可检查bank公式

HBM物理地址沿用token-major正确公式，每token4096B。每slot K/V各256KiB。

K logical bank按 `(kvHead,dimGroup)`，dimGroup=dim//8；word为32position×8dim×BF16=4096bit，row=position//32，4row。128个logical bank合计256KiB。QK decode选8head×4position时，每kvHead读取一个当前dimGroup的word，选其中4position并广播；prefill一个kvHead×32position×8dim可从一个word直接提供256 BF16，4query复用。同一word内position索引随microtile推进，禁止跨32position边界的一次读取请求。

该宽word是计算视图，实际分割成32个128bit slice布线；需要32key并行时所有slice读同row，decode仅选择4slice。物理Mem组合或拆分必须证明读端口与宽布线。fill用32token K pack buffer=64KiB，填满后按kvHead/dimGroup形成4096bit word写入；或者采用明确支持16bit write mask的分片bank，保持read/write不重叠。未证明write mask硬件实现前采用pack buffer合同。
V bank按 `(kvHead,dim mod32)`，word为8token×BF16=128bit，row=(dim//32)*16+(position//8)。bank数8×32=256，每bank64row×128bit。PV当前32dim microtile同时读取对应kvHead的32bank，形成32dim×8token；4query复用同V，decode多个Qhead共享同kvHead时同样复用。

decode如果映射128输出行为同一qh完整D128，V需要同bank的4个不同row同拍读，当前公式不满足。**首版PV映射明确改为4个qh×32dim，与prefill一致；四qh按组共享一个kvHead，dimBatch=0..3，kvHead=0..7，总32输出microtile。** 因此每拍32bank单读可给4qh，128tile产品全利用。输出散射到[head,dim]，不假定扁平连续128row。

V填充必须组装8token对应的同dim word。直接每收到token就write同word会发生read-modify-write：采用8token pack buffer，64B beat拆出32dim BF16，逐token写本地寄存pack，8token齐后写128bit word。尾tile不足8时补零并mask。V pack buffer为16KiB；加K的32token pack64KiB，合计80KiB staging，可在各KV slot共享单DMA fill engine时共享。不能在V尚未完成pack时标记整个slot READY。

bank数多，物理块碎片严重；小规模先用可综合Mem/寄存验证端口trace，后续允许把bank组合到宽RAM。组合后保持同周期读写合同，不能只保留地址公式却减掉端口。

## 7. DMA、slot和完成事件

KV Fetch完成需全部R responses处理、V pack flush、bank writes可见。append完成需B响应，prefill global committedLen只在所有layer成功后推进。AXI 4KiB边界拆burst，错误响应sticky，旧epoch必须drain但不得提交。

读FIFO与DMA目的slot独立credit；prefetch不能占用随后state/PV完成所必需的唯一输出buffer。保留控制/错误完成FIFO空间；默认禁止请求队列把所有credit用尽后再等待自身产生的释放事件。

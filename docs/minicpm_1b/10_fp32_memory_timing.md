# 10 FP32多周期与物理存储时序（修订合同）

本章细化并修正03/04/06中仅有逻辑bank的部分。多周期latency不代表每次都要停，但II不为1时不能靠更多buffer恢复每拍吞吐。所有示例L是待测参数，未宣称vendor IP已符合。

## 1. Latency与II分开

Ladd：一笔ADD被接收到结果产生的周期数；IIadd：同一个ADD pipeline相邻输入的最小间隔。Ladd8、II1可以同时有8笔在飞；Ladd8、II8是每8拍才能接一笔，64lane每拍产品的性能前提不成立。

循环读取Partial RAM、ADD、写回的保守依赖间隔：

```
Dloop = LramRead + Ladd + LwriteCommit + LsameCycleGuard
C * IIaccept >= Dloop
```

C为交错上下文数，IIaccept包括算术II和实际RAM/调度限制。初始化C=16只是起点；配置按nextPow2(ceil(Dloop/IIaccept))自动检查，未知值不强制当1。FIFO不足、RAW未解决时stall，不读旧值。

若Partial RAM读请求在产品ready时发起，产品latency不进入此反馈loop；product必须存于alignment FIFO直到RAM返回。若选择输入发射时提前读Partial，则需把product对齐延迟等计入Dloop，不能用同一公式略去。首版使用product-ready后的RAM读，容易验证。

## 2. MPU W4 partial物理组织

64row bank，每bank32word×FP32：2slot×16ctx。每bank1R1W，compute当前slot每拍读一个ctx、若干拍后写同ctx。64bank同时操作形成2048bit内部向量，分8cluster×8row本地布线，避免单中心长总线。

地址 `slot*C+ctx`，ctx=k modC。scoreboard保留每entry pendingWriteCycle/valid；read须在前一writeCommit之后，默认不依赖BRAM同拍read/write的旧/新值选择。first update该ctx使用+0，而不读上一group的旧内容。tail未初始化ctx由mask返回0。

例RAM2、ADD8、write1、guard1：Dloop12，C16足够。产品p0就绪后读ctx0；2拍后ADD接收；8拍后结果；再1拍write；下一次ctx0读距离16拍，满足安全间隔。Ladd20时Dloop24，C16不够，选C32或stall。

small PartialBank可用LUTRAM/寄存阵列实现，不能64bank各用一个大BRAM而不报告碎片浪费。每bank32×32=1024bit，总8KiB逻辑容量，综合报告实际LUT/FF/RAM及端口。

## 3. 为什么需要转置Snapshot

PartialBank一bank对应一row，其单read口无法同拍读取ctx0..15；若16读/row再64row，共1024拍，会压住每128拍来的group。

修订结构：

```
PartialBank[64row][16ctx][2slot]
       │ group最后写退休后，ctx0..15扫描16拍
       │ 每拍64个row各读1值（2048bit，局部cluster）
       ▼
SnapshotCapture / transpose
       ▼
SnapshotRegisters[8cluster][8localRow][16ctx]
       │ 选一个row，同时读16个FP32
       ▼
Group16 tree (8+4+2+1 ADD) -> scale / total
```

Snapshot必须实现多值读合同：首版register/分布式结构，每context写一个cluster的8values，每row读取其16context。分cluster把局部row mux限制8:1，再注册顶层cluster选择，避免不明确64:1跨芯片mux。不能用一块普通1R1W BRAM直接替代。

每snapshot slot4KiB，建议2slot8KiB；PartialBank另8KiB。原W4 cache/partial/total约32KiB预算须包含新增snapshot，若原分配未包括它应提高profile预算，不把两者计成同一空间。

capture需C16发射拍+RAM返回/写ack；在捕获结束前Partial slot保持FROZEN，另一slot继续compute。snapshot读完64row并预留tree返回credit后释放，不等整个下游所有计算完成才释放，但outputcredit必须覆盖在飞tree。

示例Partial占用G128+Lret14+captureRAM2+C16+write1=161拍，2slot初步可维持每128拍group。若capture/reducerstall，占用增加，依credit停compute或增加slot；不是永远满速保证。capture reader与compute reader访问不同slot，也必须仲裁物理端口：首版每slot分离Partial物理bank组（64bank/slot），或使用明确的双读资源；若把两个slot放同一个1R1W row bank，则compute read与capture read冲突，需要capture期间stall，不能宣称完全重叠。

**首版端口选择：每slot独立64个1R1W小bank（各16word），总逻辑仍8KiB；在capture旧slot时compute新slot，不争read口。** 只有证明2R1W或安全time multiplex后才将两slot合并。

## 4. 权重cache避免64个窄BRAM

一个W4 compute beat是64×4bit=256bit，cache组织为256bit word×128k/slot，2slot payload8KiB；scale cache独立。DMA128bit每两个beat拼成一个word后写下一slot，compute读当前slot每拍一个word。

物理候选为4个64bit RAM slice，slot0/1地址分区；simple-dual-port一个read一个write，读写同地址由slotlock禁止。举例配置成64bit×512需要4个RAMB36，物理比8KiB逻辑容量更大，实际mapping由综合决定。scale metadata从独立cache读，不能与weight payload抢唯一读口。

单128bit200MHz通路仅16B/拍，不能独自满供32B/拍W4。有效聚合带宽不足时weightready停顿；此前6GB/s预算约30B/拍，已不等于满64product issue。Native head每compute拍64B更需prefetch/宽度转换和实际bandwidth限制。

## 5. RPU state物理组织和共享算术调度

每head16bank×1024wordFP32=64KiB，1R1W；3slot各自一套bank，总48bank，逻辑192KiB。compute独占当前slot，DMA写下一slot、读上一slot，不直接争current端口。每bank可对应一个RAMB36的32bit×1024配置，例约48RAMB36，实际推断/宽度/输出register需核验，不当已综合结果。

state read每beat16value=512bit，但每pass两个MUL任务复用同一16laneMUL，平均一个state beat需2发射拍。第二次MUL不能在第一次结果未产生时就用值：

pass1任务A：r*S，结果写local Sd并入bypass FIFO；任务B：k*Sd，结果进入u ADD。
pass2任务A：k*delta，ADD(Sd)形成Snew并写RAM；任务B：q*Snew，结果进入o ADD。

MUL arbiter交错不同state beat的A/B，至少预存多个i/jBatch使依赖链有独立工作。任务B优先级与任务A注入需有界公平；每任务发射前同时预留tag/operand/return FIFO。共享ADD同样区分pass2 state ADD与o ADD，前者结果是后者MUL依赖。

理想4MUL/元素给出的时间只是工作量下界，不能解释成“两拍完成一个state值”。last返回、partial归约和RAMwriteack属于drain。模型中的state_eff包含假设效率，真实效率仍要micro-scheduler测。

u/o是128元素FP32，分16lane×8jBatch的register/小LUTRAM，不存18MiB外部state；若用同步RAM，Dloop同样包括read/write。每j在下一i更新前至少8jBatch×2发射拍=16拍（无更激进乱序）。若Dloop>16，设i modC partial其中C>=ceil(Dloop/16)，或stall。标量reduce后delta和nextpass barrier显式。

state RAM读写都按TxnTag/i/jBase携带mask；Sd/Snew完成值可bypass，未完成的值不能forward。DMA只有全部Snewwrites ack后才读dirtyslot写DDR。

## 6. 其他存储端口

VPU16bank32bit，每组支持16read+16write；双源必须来自不同物理组或两phase读取。KV K8bank/V16bank满足指定microstep，填下一slot与compute当前slot也需要物理隔离。即使地址不同，同一个1R1W组的两个reader仍冲突。

DDR存权重/完整KV/18MiB state；片上存工作集，不构建“所有单元随意访问”的单一多端口大SRAM。bankarbiter根据每cycle真实read/write请求发grant；不grant则请求保持，counter不推进。

## 7. 本次时序验证

`python models/validate_fp32_feedback.py`模拟RAMread/ADD/write/guard及RAW scoreboard。整数payload仅验证依赖，不能当FP32数值测试。

| C | RAM读 | ADD L | ADD II | 128输入发射周期 | RAW stall |
|---:|---:|---:|---:|---:|---:|
| 8 | 2 | 8 | 1 | 188 | 60 |
| 16 | 2 | 8 | 1 | 128 | 0 |
| 16 | 2 | 20 | 1 | 184 | 56 |
| 32 | 2 | 20 | 1 | 128 | 0 |
| 4 | 2 | 8 | 4 | 509 | 0 |
| 16 | 2 | 8 | 8 | 1017 | 0 |

说明上下文消除反馈RAW，不能消除II限制；最后两行虽然无RAW，吞吐仍低。新增转置检查验证16次context读取能得到64row完整snapshot。尚未覆盖物理RAM inference、RPU双资源完整仲裁或全NPU周期仿真。

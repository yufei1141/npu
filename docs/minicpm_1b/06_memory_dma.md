> 物理端口和Partial→Snapshot捕获细节见 [10](10_fp32_memory_timing.md)，不能仅按本章逻辑容量推定多端口RAM。

# 06 DDR、片上存储与DMA

## 1. 两级存储不是统一无代价内存

推荐：PS DDR存canonical W4、native16 embedding/head、KV、FP32 state及prefill scratch；片上BRAM/banked scratchpad存当前计算工作集。PL DDR可选用于state/KV，与PS独立搬运通路是否更快必须实测，不预先把全部weight复制过去。

板卡profile记录实际memoryRegion(base64,bytes64)、PS↔PL AXI port/width/clock、PL DDR容量/频率、可持续R/W混合带宽、BSP address map。不能假定4GB PS DDR物理上单连续region、所有HP/HPC端口独立提供DDR峰值。

FP32 state18MiB全部不能当片上常驻。KV4K48MiB、8K96MiB、16K192MiB；另statejournal双region36MiB、conv history648KiB。text-only权重按导出后manifest确切size，约0.77GB级别，不含vision，不将完整1.866GB文件全加载。

nativehead row-major可共用embedding存储的条件是loader确认同一weight；若实际不同tensor则增加一份508MB容量并按实际权重输出。body W4单独canonicallayout，不把head强制重排造成embedding随机lookup失效。

## 2. 片上工作集（逻辑容量目标）

| 区域 | 初始预算 | 用法 |
|---|---:|---|
| RPU head slots | 3×64KiB=192KiB | fetch/compute/writeback互不覆盖 |
| RPU q/k/v/delta/u/o等 | 16KiB | FP32scalar/vector、tag/scoreboard |
| KV tile A/B | 2×64KiB=128KiB | 2kvh×32token×256dim×K/V×2B |
| hidden/residual/normalized | 96KiB | prefillM16各32KiB，decode小部分 |
| QKV/Z/Gate/attention | 256KiB | 分phase生命周期复用，M16 |
| FFN gate/up/activated | 3×112KiB=336KiB | M16×F3584A16 |
| Conv weights/history | 约84KiB | currentlinear layer的48+36KiB |
| W4 cache/partial/total | 约32KiB | block双buf、Cpartial、M16 sums |
| native head/X cache | 16KiB | row contexts、input x、resultcredit |
| instruction/descriptor | 64KiB | task loops、tensor table窗口 |
| DMA FIFO/layout staging | 初始128KiB | 返回重排/pack，按在飞证明调整 |
| LUT/scalar/其他 | 参数化 | 不预先把32SFU等比乘到16lane |

表中约1.32MiB逻辑容量，尚未计全LUT、registerfiles、FIFO尾部和BRAM粒度；真实RAM使用须综合。没有ZCU102 URAM假设。布局allocator以peak生命周期分配，不允许所有工作集同时用同bank1R1W。通道数量和每bank宽度独立于容量。

## 3. 关键bank公式

RPU：S[i,j] bank=j mod16,row=i*8+j//16；每slot16bank1024wordFP32。

Vector：16bank，bank=e mod16,row=e//16；FP32与A16空间可独立或低半字存一元素，禁止未经证明RMW打包。双源算子srcA/srcB分物理组；未分组则READ_A/READ_B增加phase。

W4：64row分别4bit lane，addr=kGroup*128+kInGroup；scale block独立FP32cache。partial bank按row/ctx，64ADD每拍读写64不同row bank，ctx切换保证同entry间隔C。

KV tile：K逻辑[2kvh][32pos][256dim]，QK每dim读取4pos/kvh，共8A16值广播到8qh×4pos。bank=(kvh,pos mod4)，row=(pos//4)*256+dim，8bank每cycle一row；当前posBatch不跨4pos边界。

V逻辑[2kvh][dim][pos]：bank=(kvh,dim mod8)，row=(dim//8)*32+pos，16bank，每个PV microstep固定kvh、8dim，一pos读取8bank单row，复用给4qh。prefill多query使用相同V广播，QK读keypos4每lane匹配querymicro映射。没有4096bit Kword或32query必需的原V80大crossbar。

DDR KVtoken-major：fullLayerBase + token*2048 + isV*1024 + kvHead*512 + dim*2；layerStride=maxContext*2048，按4KiB对齐。fullLayerId0..5与原layer3/7/11/15/19/23有显式映射。CPU和DMA不能各自用不同stride。

## 4. DMA引擎和公平仲裁

DmaReadEngine按可用PS AXI端口实例化，目标逻辑128bit beat；同流类别独立queue。WriteEngine处理KVappend/state write/scratch，LocalMover处理片上stride/transpose受限布局。AXI-Lite32控制口。有效bytes/cycle记录实际目标写入，不能只统计DDR响应。

DmaCmd：src/dst regionId与baseOffset64、rowBytes32,rowCount16,srcStride32,dstStride32,kind,format,tag,slotRef。burst建议16beat，128bit总256B，4KiB边界拆分；其他宽度由adapter支持，tailWSTRB/keep精确。AR前预留ID、全burst返回与目的pack空间；多ID乱序按logicalOffset重排。

Read类别weight/head/param/KV/state/activation；Write类别state/KV/scratch。先排空在飞出口，再完成依赖的state/KV写，再新大weight预取；按age限制饥饿。port partition只是phase目标，不保证独立物理DDR。writeback占据read共用链路时纳入带宽模型。

读DONE：全部R接收、layoutflush、destinationwrites ACK；写DONE：全部W+B OKAY，不能以AW或WLAST标记完成。B错误使对应state/sessionfault，不silentcommit。

## 5. double/triple buffer和credit

W4GroupSlot在compute之前reserved，stateHeadSlot在DMA之前reserved。每slot FREE→FILLING→READY→COMPUTE→DIRTY→WRITING→FREE；readonlyweight用读完即FREE。slotRef generation防旧完成写到新角色。

3headslot使上一headwrite、当前compute、下一headfetch各有空间，但有效memory吞吐仍决定是否stall。2KVslot SERIAL_TILE基线，nextfetch与currentattention独立bank；lookahead第三KVslot需要模型证明再加。

FIFO容量≥不可撤销返回＋最坏burst重排＋消费者blocked区间；不能只设置depth2后期望IP输出可停。每发compute先预留resultslot，RPU/VPU与DMA都必须有discard sink用于epoch过期返回。allocation graph禁止持有唯一resultbuffer再等待使用该buffer的下游。

## 6. 预处理和导出

host导出时把text参数与vision分离；记录original tensor name、dtype/shape、qzeros/g_idx解释、canonicalbytecount/hash、headalias及NUM_MODE。只在实际解码证明signedrange后保留signed4，不能改expectedchecksum或静默重定量化。

state与KV初始清零逻辑使用valid/version，firstuse不读未初始化DDR。statejournal区域old/new同layout不同base；交换表在TOKEN_COMMIT原子发布，KV可见长保持旧值直到完整token结束。

## 7. 验证

PS→PL纯读、纯写、R/W混合、多个AXI端口、各种burst/stride/256B与4KiB边界；pack尾部、dtypeconversion、返回乱序、samebank争用；statewriteB延迟和journal版本；裸金属cacheflush/invalidate前后数据。必须测可持续payloadBW、延迟分位和resource counters，不能用DDR标称速度填模型。

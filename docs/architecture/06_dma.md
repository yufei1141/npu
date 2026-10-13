# 06 DMA 子系统实现合同

DMA是顶层必要子系统，不是由MPU/VPU直接访问HBM，也不是仅用于host模型加载。前版把其分散描述在MCU/ReadCtrl/KvCacheManager中，本章统一显式组织；MCU是访存控制层，可包含DMA子系统，不另造重复HBM端口。

## 1. 顶层数据路径

```
DCU/Acu ── DmaCmd ── DmaScheduler / TransferTracker
                         │
       ┌─────────────────┼─────────────────┐
  ReadEngine[j]       WriteEngine      LocalCopyEngine
       │                  │                 │
  HBM AXI AR/R       HBM AXI AW/W/B     on-chip bank ports
       │                  ▲                 │
  ReturnRouter        WritePacker            │
       │                  │                 │
  Weight/KV/Activation LayoutAdapter ────────┘
       │
  WeightStore / KvTileStore / VectorStore
       │
     MPU / VPU
```

14逻辑512bit读口为上限目标，实际board映射待确认；写引擎按可写AXI接口绑定，可与某读接口共用物理AXI但读写仲裁需要board带宽证明。Host→HBM的PCIe DMA属于shell，不能默认它由本DmaScheduler实现；NPU DMA负责HBM↔片上及可选片上搬运。

## 2. 工作类型

| kind | source→destination | adapter | completion |
|---|---|---|---|
| WEIGHT_LOAD | HBM→WeightStore | W4/BF16 block拆分、scale缓存 | 全R处理+目标writes可见 |
| KV_FETCH | HBM→KvTileStore | K32position pack、V8token pack、尾mask | pack全部flush+slot READY |
| KV_APPEND/BLOCK | currentK/V→HBM | token-major打包、WSTRB | 全B响应成功 |
| EMBED_LOAD | HBM embedding row→VectorStore | BF16向量、token range | 目标writes可见 |
| ACT_LOAD/STORE | prefill hidden/scratch↔VectorStore | 二维row stride、BF16/FP32格式 | read侧写ack或store侧B |
| PARAM_LOAD | norm gamma/RoPE/程序/descriptor→对应store | 固定格式、不静默转换dtype | 目标可见 |
| LOCAL_COPY | 片上space→片上space | contiguous/stride，支持受限格式转换 | 全destination writeAck |

首版不实现任意HBM→HBM copy或通用scatter/gather链表；未知layout拒绝。LOCAL_COPY可用已有bank engine时分实现，不承诺独立物理端口或与VPU满速并行。

## 3. 命令和控制模块

DmaCmd沿用共用tag、addr64/bytes32、targetSpace/slot/layout；二维搬运扩展srcRef/dstRef、rowBytes32、rowCount16、srcStride32/dstStride32。宿主tensor引用经validator解析，检查所有row范围和64bit乘加溢出。kind决定位于HBM的一侧；片上目标不能把HBM地址直接解释为bank地址。

子模块：DmaScheduler（按流类别分队列）；AddressGenerator（逐row/burst）；ReadEngine×实际口数；WriteEngine；ReadReorder；LayoutAdapter；LocalBankPortAdapter；TransferTracker；DmaCompletionQueue。

各类队列独立避免KV append被大型weight命令阻塞。初版优先排空已经预留slot的返回/写响应，append与控制参数有age保证；weight/KV/activation读按DCU phase分配带宽，phase切换先drain对应在飞tag，不回收尚未完成的ID。

## 4. 发射及credit

AR前同时预留AXI ID、完整burst返回容量及adapter目的buffer credit；目标RAM写不立即ready时，可在已保留FIFO中停留。不能只保留ReadFIFO而未保留layout pack上下文，造成多个命令互相覆盖pack。

每ReadEngine在飞burst初始8，具体按实测latency/burst/BW收敛。ID内顺序，跨ID按tag/byteOffset重排，布局adapter要求原序时进入reorderFIFO。AXI burst不跨4KiB，最大目标16beat，尾部合法keep。RRESP错误停发该transfer并继续接收/drain。

WriteEngine先得到可发送完整burst数据，再AW；或逐beat预留不能缺的source数据，首版采用burst staging简化。W last精确对应AWLEN；WSTRB屏蔽尾部，不覆盖邻接token。B响应关联ID和transfer，收到全部OKAY才成功。AW或W已发不能ABORT后忘记B。

## 5. 状态机与完成

TransferTracker：ALLOCATE→RESERVE_DST→GENERATE_BURSTS→IN_FLIGHT→ADAPTER_FLUSH→WAIT_DEST_ACK/WAIT_B→DONE。generatedBytes、receivedBytes、adapterWrittenBytes、inflightRead/Write、packPartial均单独计数。所有必要计数归零/完成才UnitDone，不能以最后AR/W/R接收替代完成。

KV FETCH slot状态FILLING，直到K/V全部pack完成和writes可见才READY；DMA只发SlotReady，消费者计数由Acu管理。KV APPEND完成可推进layer visibleLen，global committedLen仍由TOKEN_DONE控制。

ABORT停新AR/AW，既有数据/响应drain到合法暂存或discard sink；不提交过期epoch slot/tensor。保留首fault及源地址、cmdTag、ID，不打印credential信息。不能通过全局reset中断未知在飞AXI。

## 6. pack及物理端口成本

K32token pack64KiB、V8token pack16KiB，总80KiB staging，单active KV fill时可共享；若并发多个slot填充要复制pack或显式限制active fill，不能让两transfer共享未区分tag的partial word。

K full word4096bit由bank slices写，V128bit word写相应dim bank；若write port宽度更小需多拍flush。模型必须增加pack收尾、bank write争用和phase切换周期。attention_schedule_model.py当前把KV fill视为固定带宽整块就绪，尚未覆盖这些实际阶段。

MPU读weight、DMA写下一weight slot；Acu读当前KV、DMA填另一KV slot，必须物理bank隔离。VectorStore ACT搬运与VPU对同组同时访问会仲裁，不能把“DMA独立引擎”误解为“存储端口无限”。

## 7. 验证

burst页边界/尾strb、二维stride、跨ID乱序、随机ARREADY/RREADY/AWREADY/WREADY/B延迟、R/B错误、adapter pack尾部、abort全阶段、same bank conflict。检查搬运字节内容与顺序、完成条件、slotREADY时刻及不越界；超时失败。最终DMA性能按有效目标写入bytes/cycle报告，包含pack/flush，非只统计HBM线上beat。

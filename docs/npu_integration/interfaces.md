# 接口与流控合同草案

状态：逻辑协议草案，尚未冻结为 RTL 端口或二进制指令编码。更新：2026-10-29。

本文件是背景合同摘要；当前已补充[02建议指令编码](02_instruction_set.md)及[14顶层连接](14_npu_top.md)，完整模块索引见[README](README.md)。如粒度不同，按编号详细文档实现。

## 1. 通用握手

- 仅在时钟沿 `valid && ready` 时接受一个传输。
- `valid && !ready` 时，生产者保持 valid 和全部 payload 稳定，包括 tag、mask、last、status。
- 消费者 ready 可以在没有 valid 时拉高；生产者不得要求先见 ready 才开始提供 valid。
- 每个请求的配置和元数据在接受时锁存，不继续读取可能已变化的上游输入。
- SRAM 读、外存请求、计算任务和完成事件分别计数，不能共用一个 done 信号代表全部状态。

这是新外围接口的目标合同，不是对现有 SFU 占位 RTL 已正确实现该合同的声明。

## 2. MPU 任务适配器

### 外层逻辑接口

| 接口 | 必要信息 | 接受条件 |
|---|---|---|
| 任务输入 | job_id、mode、m_len、n_grp、tiles、k_steps、源/目的 buffer 信息 | 当前任务已完成交接，配置合法，结果容量预约成功 |
| A 输入 | 1024 bit 数据，一个 K-step 的激活 | 适配器允许且 core a_ready |
| B 输入 | 2304 bit 数据，权重与 scale 一起提交 | 适配器允许且 core b_ready |
| 结果输出 | 4096 bit tile、job_id、tile_index、有效行列、last | 适配器结果 FIFO 非空；payload 在背压时保持 |
| 完成输出 | job_id、状态、预期/实际结果数 | 结果已交给下游，且下游写回确认已纳入顶层完成管理 |

源/目的地址字段可以由外层搬运和 pack 引擎持有，不要求将地址直接送入 `gemv_stream`。

### 合法性与计数

- m_len ∈ [1,4]，n_grp ∈ [1,8]，tiles ∈ [1,65535]，k_steps ∈ [1,65535]。
- 实际可接收的 tiles 还受结果预约容量约束；超容量任务在外层拆分或返回明确错误，不能静默截断。
- 每个任务恰好接受 `tiles × k_steps` 个 A beat 和同样数量的 B beat。
- 乘积计数至少 32 bit；不能用 16 bit 计数器保存两个 16 bit 数的乘积。
- A/B 可以独立到达，但各自严格保持 tile-major、K-step-minor 顺序；适配器禁止写入额外 beat 污染下一任务。
- first/last/配置在整个任务中由锁存值和计数推导；下一个命令到来不能改变仍在流水线中的任务配置。
- 结果捕获数、结果下游接受数和写回确认数分别维护；命令完成时预期数量必须一致。

### 状态生命周期

```text
IDLE
  ├─ 非法任务 → 报错完成（不 start）→ 完成记录被接受 → IDLE
  └─ 合法任务 + 预约成功
         → 锁存配置 → core start 一拍
         → FEED（精确计数 A/B）
         → DRAIN（输入结束，等待 core done 与全部结果）
         → WRITEBACK（等待已产生结果提交并确认）
         → COMPLETE（完成记录保持到接受）
         → IDLE
```

这是控制状态的逻辑划分，结果捕获与写回可在 FEED 时就开始，不要求串行等待。
初版只有一个计算任务在途，避免仅凭 `busy=0` 就重用计数器或配置。

### 结果容量不变量

设总结果槽数为 C，已占用结果槽数为 O，已预约但尚未到达的结果数为 R：

```text
0 <= O + R <= C
接受一个 tiles=T 的任务：要求 C - O - R >= T；R += T
每捕获一个 acc_valid：R -= 1；O += 1
每个结果槽安全释放：O -= 1
```

同拍事件必须合并更新，不能让多个非阻塞赋值互相覆盖计数。
首版按[09](09_result_writeback.md)保守实现：该tile全部目的写确认后才安全释放槽，所以O包含serializer当前占用的槽。不能仅在FIFO头出队就释放却继续从原槽读取。未来如另设holding容量可更早转移所有权，但仍须独立跟踪目的写确认。

## 3. MPU 数据布局

以下位序来自当前 `gemv_cluster.v`，下层 PE 权重内部位序还应以 PE / unit RTL 和参考向量核对。

```text
A row m                  = a_din[256*m +: 256]       m=0..3
B weights, group n       = b_din[256*n +: 256]       n=0..7
B scale, group n         = b_din[2048+32*n +: 32]    n=0..7
Result element (m, col)  = acc_out[32*(32*m+col) +: 32]
```

W16 时每行 A 的低 64 bit 放 4 个输入元素，cluster 内部复制给 PE；其余 A bit 不携带新增 K 元素。
W4 时一行 A 对应 16 个元素。两种模式的 B 都使用完整的 256 bit / group。
scale 以原始 E8M0 位型传送，不在 DMA 中误转成普通有符号 int8 或 BF16。

内存地址统一以字节计，元素排列以 bit slice 明确。逻辑 tensor layout 与 MPU pack layout 分开描述，
不要求片上所有 buffer 都存成 4096 bit 或 2304 bit 宽。

## 4. SRAM 与外部内存逻辑接口

### SRAM

请求包含 read/write、buffer_id、byte_offset、byte_enable、data、request_tag。
响应包含 request_tag、read_data 或 write_ack、error。

必须明确逻辑读延迟和每 bank 的并发端口数；接入实际 SRAM 后仍保持相同的可见事务语义。
首版建议每个逻辑客户端按序返回，避免在没有需要时引入乱序 tag 重排逻辑。
byte_enable 负责尾部写保护，不能依赖写零覆盖邻接有效数据。

### 外部内存

已确定的输入为 `clk_io` 域 128 bit 数据流，采用 `in_valid/in_ready/in_data[127:0]`；
tile 元数据由独立描述符提供，不要求每个数据 beat 携带宽地址。
v0.1帧已在[06](06_dma_engine.md)建议使用keep[15:0]、last、tag和error；最后一个部分beat必须有精确有效字节数，物理接口仍由平台适配。

用户提出的约 18 GB/s 暂记为系统预算目标；单路 128 bit 每拍最多 16 B，达到该吞吐理想情况下也需要 1.125 GHz。
多路 128 bit 或更宽聚合口是后续选项，尚未冻结；不能让窄口继续串行接收却按聚合带宽安排计算。

计算域为独立的 `clk_core`。权重经双时钟 RAM 从 IO 域传入；整个 tile 的描述符通过握手或异步 FIFO 跨域，
普通 valid/ready 不能直接跨时钟域连接。各接口的通用握手规则仅对各自所在时钟域成立。

DMA 发出地址、方向、长度和 tag，适配层转成具体平台事务。后续读数据和写数据均支持背压；
写完成以适配层返回的确认事件为准，而不是最后一拍写数据被接受。

结果输出端的物理总线宽度和是否与输入共享尚未确定，不因输入为 128 bit 就自动视作输出也是 128 bit。

首版可以限制每个通道的在途事务数，并由仿真内存按序应答。后续接 AXI 时才补充 burst、边界、ID、
多在途和错误码映射；不能用这个逻辑合同声称已经实现 AXI。

## 5. buffer 所有权与依赖

```text
FREE → DMA_FILLING → READY → ENGINE_READING → FREE
FREE → ENGINE_WRITING → READY → DMA_DRAINING → FREE
```

这里表示一个 buffer 的某一代数据；多消费者时，需要保留到最后一次读取确认。
首版按整个逻辑 buffer 保守锁定，先不实现任意地址范围的细粒度别名检测。

- DMA 预取下一块只使用另一个空闲 bank。
- 双缓冲切换条件是数据就绪和前一代访问完成，不能单纯按奇偶拍切换。
- 任务依赖在写入可见后解除；“命令已接受”不是“buffer 已就绪”。
- SFU/VPU 读改写的原地操作必须显式许可，不能让搬运引擎同时覆盖源数据。

### 双时钟权重 bank 所有权

- 每个 bank 的装填状态只在 `clk_io` 域更新；消费状态只在 `clk_core` 域更新，不用两个 always 块修改同一状态寄存器。
- IO 端完成全部权重和 scale 的 RAM 写入后，发布包含 bank_id、job_id、mode、n_grp、k_steps、长度与偏移的描述符。
- 描述符经完整 req/ack 握手或异步描述符 FIFO 交给计算域；多 bit payload 保持稳定直至协议允许更新。
- 计算域仅读取已提交 bank；IO 端在收到最终 release 之前不得覆盖该 bank。描述符已收到的 ack 不等于 release。
- release 在最后一个缓存 RAM 读响应及其消费完成后发布；不直接等同于 MPU busy 清零或结果写回完成。
- 初版对整个 bank 串行转移所有权，不依赖异步同址读写的 RAM 行为。
- 双域复位会使未消费 tile 作废；单域独立复位必须经协调恢复或 generation/epoch 协议，不能让另一域继续使用旧所有权。

详细状态和带宽合同见 [128 bit 双时钟权重路径](weight_path.md)。

## 6. VPU / SFU 的任务约定

- 每个向量任务描述有效元素数与尾部 mask；padding lane 不参与归约。
- SUM 归约的无效 lane 单位元为 0；MAX 使用显式有效位处理空 lane，避免依赖 MPU 不支持的 Inf 数值。
- RSQRT/RCP 等标量结果通过显式广播用于后续向量任务；不占满向量带宽反复从外存读取同一标量。
- SFU 适配器锁存输入 op、format、mode 和数据；真实接入时验证在握手后修改上游输入不影响结果。
- SINCOS 是两次输出还是双端口输出，以最终 SFU 实现为准；若按现有架构文档的两 beat 方案，则同一输入必须收到 sin、cos 两个结果后才算完成。
- mask、status、job_id、输出 beat 序号在背压时与数据一起保持。
- 当前 SFU 有 flush，但 MPU stream 无独立 flush；整机取消任务不能只 flush SFU 就发布所有资源可复用。

## 7. 验证时必须覆盖的集成情形

1. 随机且相互独立的 A/B 供数暂停，结果仍保持顺序和数量正确。
2. 输出 ready 长时间为 0，MPU 的 acc_valid 仍无丢失，后续任务被正确限流。
3. k_steps=1、tiles=1、最大可预约 tiles，以及 FIFO 同拍进出。
4. 最后一个输入提交后 busy 已低但结果未到，此时新任务不能修改旧任务配置。
5. 16 bit tiles/k_steps 的乘积不溢出；零维度、越界和不支持格式明确报错。
6. m_len/n_grp 部分使能及 K/N 尾部补零，不覆盖相邻有效内存。
7. 结果已被写回接口接受但写确认延后，后继读任务仍不得提前启动。
8. 连续不复位的多个 job，没有残余操作数或完成事件串入下一任务。
9. SFU 输入接受后改变上游数据、输出背压、SINCOS 两输出和异常位。
10. DMA 错误、复位和取消任务期间的已接受事务，有清晰的排空与报告行为。
11. IO/core 两时钟频比、相位和周期均变化时，bank 提交、消费、释放不丢不重。
12. 128 bit 输入的最后一个部分 beat、权重与 scale 分别装填、未写满 bank 禁止发布。
13. 计算域慢于 IO 域时正确背压，快于 IO 域时正确等待完整 tile；禁止覆盖仍在使用的 bank。
14. n_grp 部分启用时，只装填和读取有效列组；不能仍从外存传完整 8 列组再丢弃。

以上是待实现的验收条件，不是本轮已经执行通过的测试。

# 05 Scratchpad、bank 仲裁与 SRAM wrapper

状态：v0.1 建议设计。此处只管 clk_core 域普通 scratchpad；双时钟权重 RAM 见 [07](07_weight_buffer.md)。

## 1. 逻辑地址空间

| buf_id | 名称 | 典型用途 |
|---:|---|---|
| 0 | ACT0 | 当前激活 / 输入 |
| 1 | ACT1 | 下一激活 / 输出 |
| 2 | TMP0 | 向量中间量 / residual |
| 3 | TMP1 | softmax / SFU 中间量 |
| 4 | KV_TILE0 | 外部 K/V 的当前分块 |
| 5 | KV_TILE1 | 下一 K/V 分块 |
| 6 | ATTN_STATE | online attention 的 m/l/o，未实现时禁用 |
| 7 | PARAM | Norm gamma / 小常量 |

每个 buffer 在构建时配置 `enabled, bank_id, base_word, size_bytes`，只允许在全局空闲时经平台配置更新。首版 buffer 不跨物理 bank，物理区间互不重叠；关闭的 buffer 一律拒绝。容量必须为 B=MEM_DATA_W/8 的整数倍，物理 bank 数和每 bank 深度不以模型参数数量直接推定。

逻辑 `byte_offset` 相对 buffer，RAM 字地址为 `base_word+byte_offset/B`。计算端必须按元素范围验证真实 tensor，memory_subsystem 再验证字地址、写字节和配置范围。读越界不会通过“补零”默默成功。

## 2. memory_subsystem 端口

客户端编号固定：0 MPU_A、1 VPU、2 SFU、3 RESULT_WB、4 DMA_CORE。每客户端一个 req/rsp 通道，使用 [01](01_common_protocol.md) 的类型。

| 信号 | 方向 | 类型 | 说明 |
|---|---|---|---|
| clk_core / rst_core_n | I | 1 / 1 | 时钟、复位 |
| buf_enable / buf_bank / buf_base_word / buf_size_bytes | I | 8×(1/8/32/32) | 来自顶层配置寄存器；分别为启用、物理bank、基字地址、字节容量 |
| c_req_valid[4:0] / c_req_ready[4:0] | I / O | 各 5 | 客户端请求 |
| c_req[0..4] | I | 5×mem_req_t | 客户端 metadata / 数据 |
| c_rsp_valid[4:0] / c_rsp_ready[4:0] | O / I | 各 5 | 响应及写确认 |
| c_rsp[0..4] | O | 5×mem_rsp_t | 响应保序 |
| outstanding_count | O | 16 | 所有已接受未完成响应数 |
| access_fault_valid / info | O | 1 / 64 | 错误诊断，不能代替 error response |
| bank_busy | O | BANKS | 物理 bank 已预约未交付响应 |

首版每客户端只允许一个在途请求，包含等待响应被接受的时间。`req_ready` 除仲裁获胜外还要求该客户端无在途、目标 bank 有请求容量、已预约响应寄存器。客户端不能发新请求后再覆盖旧 tag。

buffer配置与scheduler的范围检查共用同一寄存器输出；仅在全局IDLE且无活跃任务／在途请求时更新。旧完成事件可保留，但不能被解释为映射更新后新内容的生产者，软件须重新建立相应数据依赖。

## 3. bank_arbiter

每个物理 bank 使用独立轮询仲裁，同 bank 每拍最多接受一个请求；不同 bank 可同拍各接受一个。轮询指针在真正 req_fire 时更新，不能因 valid 而更新。资源公平性以消费者最终释放背压为前提。

| 子模块信号 | 方向 | 类型 | 含义 |
|---|---|---|---|
| req_valid / req_ready / req | I / O / I | 每客户端 | 目标为本 bank 的请求 |
| ram_en / ram_we | O | 1 / 1 | 固定延迟物理请求，无 ready |
| ram_addr | O | RAM_ADDR_W | bank 内字地址 |
| ram_wdata / ram_wstrb | O | MEM_DATA_W / B | 写数据与字节 enable |
| ram_rvalid / ram_rdata | I | 1 / MEM_DATA_W | wrapper 固定延迟返回 |
| ram_error | I | 1 | 仿真错误注入 / 平台错误 |
| selected_client / selected_tag | 内部 | 3 / 8 | 请求时锁存，返回时路由 |

建议最小实现每 bank 一个请求在途和一个响应 holding register，追求明确正确性而非每拍接受新访问。要增加流水吞吐需扩充对应 metadata/response 队列，不能仅去掉 busy。

## 4. bank 状态机与固定延迟

```text
EMPTY --合法 req_fire--> RAM_WAIT --L_RAM 拍--> RESP_HOLD
EMPTY --非法 req_fire--> RESP_HOLD(error)
RESP_HOLD --对应客户端 rsp_fire--> EMPTY
```

非法请求不产生 ram_en。写也经过 RAM_WAIT，在 wrapper 确认写沿完成后返回；写时不能将同拍 ram_rdata 当成读结果。L_RAM 是 wrapper 参数，仿真默认 1，改变后必须一起调整 metadata 延迟。

响应 holding register 的 valid 在无 fire 时保持，不能因为下一拍 RAM rvalid 降低就丢响应。复位清 valid 和 owner，不要求清 SRAM 内容；有效数据来自 buffer 生命周期，不来自 RAM 默认值。

## 5. sram_wrapper 端口与读写语义

| 信号 | 方向 | 位宽 | 规则 |
|---|---|---|---|
| clk / rst_n | I | 1 / 1 | core 域；rst 仅清响应状态 |
| en / we | I | 1 / 1 | 接受沿发起操作 |
| addr | I | RAM_ADDR_W | 字地址 |
| wdata / wstrb | I | MEM_DATA_W / B | 按字节更新 |
| rvalid / rdata / error | O | 1 / MEM_DATA_W / 1 | 请求后 L_RAM 拍响应 |

仿真 wrapper 支持读/写应答，不能用组合读数组去假装同步 SRAM。首版单访问端口避免读写同址语义差异。若物理宏无字节写使能，则 wrapper 用内部 read-modify-write，增加延迟并阻塞后续访问；不能无声覆盖无效字节。

## 6. 多字访问和冲突

MPU_A 收集一个 A beat 可能多次读取；VPU 二元算子可先读 src0 再读 src1；SFU 和 DMA 也按单字事务推进。单客户端一在途不保证高性能，但不会阻止正确计算。后续性能优化依据计数器增加 bank / 队列。

同 buffer 的生产者消费者冲突由 scheduler 锁阻止。memory_subsystem 不猜测任务依赖；来自旁路调试主机的访问也必须经过相同锁或只在 idle 允许。

## 7. 错误、完成与验收

每个接受请求必有恰好一个响应，包括 ALIGN/RANGE。外部 task 若收到 MEM_ACCESS，停止发新请求并消费已有响应，是否可恢复由是否已有副作用及计算核是否可停止决定。

验收：字节 strobes 保留邻接数据、同 bank 五客户端轮询、不同 bank 并发、响应背压、写确认延迟、tag 路由、关闭 buffer、最后合法字/第一个非法字、改变 L_RAM 后流水对齐，以及复位不把旧响应路由给新请求。

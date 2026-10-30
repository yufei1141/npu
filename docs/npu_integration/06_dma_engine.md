# 06 平台无关 DMA 与普通数据 CDC

状态：v0.1 建议设计；不是 AXI master。具体 AXI burst、DDR 控制器和带宽由平台适配完成。

## 1. 模块边界

dma_engine 包含 core 任务桥、io 地址/帧控制、普通数据双向 CDC FIFO，以及 core scratchpad 访问桥。LOAD_W 直接送 io 域 weight_loader，不经过普通 LOAD 数据 FIFO。首版一个 DMA 命令在途，外存每次一个行事务在途。

| 信号 | 方向 | 域 / 位宽 | 说明 |
|---|---|---|---|
| clk_core/rst_core_n, clk_io/rst_io_n | I | 两域 | 协调复位 |
| task_valid / task_ready / task_cmd | I / O / I | core，1/1/512 | 调度输入 |
| event_valid / event_ready / event_data | O / I / O | core，1/1/128 | 完成管理器 |
| mem_req_valid/ready/req | O/I/O | core，mem_req_t | 客户端 DMA_CORE |
| mem_rsp_valid/ready/rsp | I/O/I | core，mem_rsp_t | 普通 LOAD/STORE |
| rd_req_valid / rd_req_ready | O / I | io，1 | 外存读请求 |
| rd_req_addr / bytes / tag | O | io，64/32/16 | 每行一个读事务 |
| rd_valid / rd_ready / rd_data | I / O / I | io，1/1/128 | 128 bit 输入 |
| rd_keep / rd_last / rd_tag / rd_error | I | io，16/1/16/1 | 读帧信息 |
| wr_req_valid / wr_req_ready | O / I | io，1 | 外存写事务起始 |
| wr_req_addr / bytes / tag | O | io，64/32/16 | 每行一个写事务 |
| wr_valid / wr_ready / wr_data | O / I / O | io，1/1/EXT_WR_W | 外部写数据，宽度参数 |
| wr_keep / wr_last / wr_tag | O | io，EXT_WR_W/8、1、16 | 尾部字节和事务身份 |
| wr_rsp_valid / wr_rsp_ready / tag / error | I/O/I/I | io，1/1/16/1 | 最终外存写确认 |
| wload_valid/ready/cmd | O/I/O | io，1/1/512 | 申请权重 bank，见 [07](07_weight_buffer.md) |
| wdata_valid/ready/data/last/error | O/I/O/O/O | io，1/1/128/1/1 | LOAD_W 数据，keep 必须全 1 |
| wload_done_valid/ready/status | I/O/I | io，1/1/16 | core 已提交 tile 或失败 |
| fatal | O | core，1 | 同步后的致命故障 |

EXT_WR_W 在平台配置中独立定义，仿真适配器可选 128；它不是用户已确定的物理输出宽度。rd 输入固定 128。

## 2. 帧协议

每行 byte 数由 rd/wr_req_bytes 指定，tag 为该 DMA 内部分配的事务序号，不直接把 job_id 当作所有行的唯一 tag。平台必须按当前单在途协议返回对应 tag，不能将不同事务数据混入一帧。

- 正常输入 `rd_keep` 为低位连续 1；非末拍必须全 1，末拍的有效字节数精确等于剩余长度，rd_last 只在末拍为 1。
- `rd_ready=0` 时，平台保持 data/keep/last/tag/error。每帧累计 popcount(keep) 必须等于 req_bytes。
- LOAD_W 的长度已 16 B 对齐，所有拍 keep=FFFF；错误帧不发布 bank。
- wr_last 随最后有效数据保持到被接受；最后 wr_fire 不是写完成，必须等 wr_rsp_fire。
- 平台读错误若能终止当前帧，应返回带 rd_error 的终止拍并保持至接受；中途错误且无法提供可靠帧边界则进入 fatal，等待协调恢复，不能把下一事务数据当作剩余数据。

## 3. CDC 通道

| 跨域路径 | 首版机制 | 所有权 |
|---|---|---|
| task_cmd core→io | 一个完整 req/ack mailbox | core payload 保持到完整握手结束 |
| 普通 LOAD 数据 io→core | 深度 8、128 bit+keep+row_last+error 的 async FIFO | 入队/出队指针各自在本域 |
| 普通 STORE 数据 core→io | 同上，内部仍为 128 bit 小块 | io 按 EXT_WR_W 聚合/拆分 |
| core 行写确认 / 行读数据已排空通知 | 控制 mailbox，带 row_index 与 status | 普通 LOAD 下一行等待当前行写确认 |
| io 命令终结→core | 完成 mailbox，携带 engine_event_t | core 事件被接受前不能复用任务 |

async FIFO 采用标准独立时钟实现：深度为 2 的幂，二进制本地指针、Gray 跨域指针、写域 full 和读域 empty。不要用跨域组合 ready/valid；复位和 busy 由 FIFO wrapper 暴露为本地 ready 条件。若平台采用 XPM_FIFO_ASYNC，按对应器件手册接 reset/busy；平台无关行为模型仍需双时钟验证。

FIFO 不能提高 128 bit 窄口的持续带宽。Gray 总线的实现约束和同步器属性属于平台 CDC 交付，不由仿真通过替代。

## 4. IO 命令状态机

| 状态 | 动作 | 下一步 |
|---|---|---|
| IDLE | 接收跨域任务并验证 | LOAD_W→W_ALLOC；其他→ROW_REQ |
| W_ALLOC | 等 wload_fire，确保空 bank 已预约 | 成功→ROW_REQ；参数失败→FINISH |
| ROW_REQ | 对 LOAD 发 rd_req；STORE 发 wr_req | 请求 fire→ROW_DATA |
| ROW_DATA | LOAD 推 FIFO / weight_loader；STORE 拉 FIFO 发外存 | 末拍 fire→ROW_WAIT |
| ROW_WAIT | LOAD 等 core 本行写确认；LOAD_W 等 wload_done；STORE 等 wr_rsp | 正常且还有行→ROW_ADV，否则→FINISH |
| ROW_ADV | 扩位更新 src/dst 地址，row_index++ | ROW_REQ |
| FINISH | 把终结结果交回 core | 完成 mailbox 被接管→IDLE |
| ERROR_DRAIN | 停止新行，消费能可靠终止的当前帧/响应 | 可终止→FINISH；边界不可靠→FATAL |
| FATAL | 停止请求和 bank 发布，保留诊断 | 仅协调 reset |

LOAD_W 视作一行，其实际读总长度 P4；weight_loader 只有在 core 提交确认后给 done。普通 LOAD 的 io 末拍到达后还要等待 core 所有字写确认。

## 5. core 数据桥状态机

LOAD：`EMPTY → COLLECT_128 → WRITE_WORD → WAIT_ACK → COLLECT_128/ROW_DONE`。MEM_DATA_W=256 时通常拼两个 128 bit；末尾不足一字产生 wstrb。每行开始清聚合有效字节，不能把上一行的 padding 写到下一行。

STORE：`IDLE → READ_WORD → WAIT_DATA → EMIT_128_CHUNKS → NEXT_WORD/ROW_DONE`。读尾部只发送有效字节。数据桥必须锁存响应，不能在消费者停顿时再次改其切片索引。

core 一行的数据桥归零、FIFO 当前行已排空且写响应完成后，才能确认行完成。rows>1 的 stride 跳转不能只按实际接收 beat 数推算。

## 6. 指令、计数与异常

使用 [02](02_instruction_set.md) P0..P6 的 2D 行参数；LOAD_W 使用特殊布局。总字节、当前行字节、剩余 beat、row_index、memory outstanding、成功写确认字节分开计数。

普通 LOAD 完成单位是确认写入的有效字节；STORE 为外部确认的有效字节；LOAD_W 为完整发布的 W_bus。发生部分写入错误，event.completed_units 只报告已确认部分，不得把目的数据发布为成功可用。若事务边界可信且错误可恢复，FAILED提交后可按正常retire释放锁，但软件必须重新完整覆盖所需数据并建立新的成功依赖后再读；若进入fatal，物理资源与锁保持隔离，直至协调恢复。释放锁本身不表示部分数据有效。

只允许一个 DMA 命令在途不意味着 bank 在 DMA 完成后即可覆盖：LOAD_W 完成时 bank 仍属于 core，释放由 weight_reader 决定。

## 7. 验收

长度 1/15/16/17/31/32/33 B，单行/跨 stride，多频比、FIFO 满空、读/写背压、last 提前/延后、keep 非连续、tag 错、外部写确认延后、普通 LOAD 与 MPU 并发、LOAD_W 不走激活 FIFO、错误时不再发下一行以及复位后的旧帧隔离。

物理适配参考：AMD UG953/UG974 的 XPM_FIFO_ASYNC reset/busy 和独立时钟约定；它们约束对应宏的接法，不把本逻辑接口自动变成 AXI。

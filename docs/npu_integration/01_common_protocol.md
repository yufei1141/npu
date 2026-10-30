# 01 公共协议、类型与设计参数

状态：外围 RTL 的 v0.1 建议合同，2026-10-29。除明确标为“现有接口”的 MPU / SFU 端口外，本目录新增端口、编码和参数均为本次设计提案，尚未实现、综合或冻结为软件 ABI。

## 1. 阅读与实现边界

模块以本文件统一时钟、握手、tag、错误和存储事务，以 [02 指令](02_instruction_set.md) 统一控制。模块文档的信号方向均相对该模块；`I/O` 表示输入/输出，不表示时钟域。类型名代表一组明确字段，可在后续 RTL 中用 SystemVerilog packed struct 或展开端口实现。

当前确定：128 bit 输入、双时钟权重搬运、W4A16 / W16A16、GEMV 部分列组、0.5B–3B 模型范围。18 GB/s 为评估目标；频率、实际存储容量及物理输出接口仍未冻结。所有新参数的默认值仅为可仿真的首版建议。

## 2. 参数与时钟

| 参数 | 建议默认值 / 约束 | 含义 |
|---|---|---|
| IO_DATA_W | 128，首版仅支持 128 | 输入字宽；未来聚合口另定义适配器 |
| MEM_DATA_W | 256，可取 128/256/512 | core 域逻辑 scratchpad 字宽，不能当作外部总线宽 |
| ADDR_W / EXT_ADDR_W | 32 / 64 | buffer 内字节偏移 / 外部字节地址 |
| JOB_W / SLOT_W / TAG_W | 16 / 8 / 8 | job 标识 / 事件槽 / 客户端本地事务 tag |
| CMD_FIFO_DEPTH | 8，至少 2 | 已接受但未发射的指令数 |
| EVENT_SLOTS | 16，2..254 | 显式释放的依赖事件槽；0xFF 保留为无槽 |
| CPL_FIFO_DEPTH | 8，至少 2 | 主机完成记录容量 |
| RESULT_SLOTS | 2，1..65535 | MPU 4096 bit 结果槽；限制每个 MPU job 的 tiles，须可用16bit编码 |
| G_CAP / M_CAP | 8 / 4，匹配现有核 | 最大列组和行数，不随外围默认参数修改 |
| WBUF_K_STEPS | 256，1..65535 | 每 ping/pong bank 的 K-step 容量；须可用16bit编码，不代表目标模型尺寸 |
| V_LANES | 4，正整数 | 新 VPU 的逻辑 FP32 lane 数 |
| SFU_LANES / SFU_DATA_W | 当前接入为 8 / 32 | 当前 sfu_top 固定两个四元素 slice，不能随意参数化 |
| CDC_STAGES | 2，至少 2 | 控制同步器级数；不代表 CDC 已验收 |

`clk_core/rst_core_n` 管命令、调度、scratchpad、MPU/VPU/SFU、完成；`clk_io/rst_io_n` 管外存平台适配、DMA 数据端和 weight_loader。复位默认协调断言，各域同步释放；系统通过双方 `domain_ready` 交换确认后才开放指令。任何单域意外复位均进入全局恢复，不能继续沿用旧 bank 所有权。

## 3. 通用握手与保序

- `fire = valid && ready`，仅在接口所在域的上升沿接受。valid 不得依赖先见 ready。
- `valid && !ready` 时，valid、数据、tag、mask、last、status 保持稳定。
- `start / acc_valid / core_done` 是现有计算核的脉冲，不属于 ready/valid；必须由外围适配成可保持的事务。
- 每客户端的 memory response 按请求接受顺序返回。首版每客户端最多一个在途事务，tag 仍保留用于错误定位和后续扩展。
- 内部任务一经接受必须终结为一条成功或错误事件；输出完成记录被背压不能导致重复执行任务。
- 首版不支持正常运行中的取消指令，不以强制清计数器实现取消。

## 4. 公共任务与完成类型

`cmd_t` 是 [02](02_instruction_set.md) 定义的 512 bit 编码；模块任务口保留整个 cmd，未使用字段必须验证为零。

`engine_event_t` 和主机完成记录都为 128 bit：

| bit | 字段 | 位宽 | 含义 |
|---|---|---:|---|
| 15:0 | job_id | 16 | 来自指令 |
| 23:16 | event_slot | 8 | 0..EVENT_SLOTS-1，行政指令用 0xFF |
| 31:24 | opcode | 8 | 原指令类别 |
| 47:32 | status | 16 | 0 成功，非零见下表 |
| 63:48 | detail | 16 | 模块内部原因 / 子阶段；0 表示无补充 |
| 95:64 | completed_units | 32 | DMA 为成功搬运字节；MPU 为写确认 tile；向量/SFU 为已写确认元素 |
| 127:96 | aux | 32 | 首个错误偏移 / tile 序号等，具体模块定义；正常为 0 |

错误编码：`0000 OK`、`0001 BAD_VERSION`、`0002 BAD_OPCODE`、`0003 BAD_FIELD`、`0004 RANGE`、`0005 ALIGN`、`0006 UNSUPPORTED`、`0007 DEP_FAILED`、`0008 DEP_MISSING`、`0009 DMA_BUS`、`000A FRAME_LENGTH`、`000B MEM_ACCESS`、`000C NUMERIC`、`000D INTERNAL`、`000E RESET_REQUIRED`、`000F EVENT_BUSY`。全部为十六进制 16 bit。

命令拒绝且无副作用时属于可恢复错误；任务运行后出现无法补齐的操作数、不可恢复总线错误或内部数量不一致时锁存 `fatal`。fatal 时停止新发射，保留完成出口和诊断；资源进入隔离状态，不能发布依赖可用或自动重新分配。

## 5. scratchpad 请求/响应类型

所有地址以字节计。`B=MEM_DATA_W/8`，每次请求覆盖一个 B 字节对齐的逻辑字。

| mem_req_t 字段 | 位宽 | 语义 |
|---|---:|---|
| write | 1 | 0 读，1 写 |
| buf_id | 8 | 逻辑 buffer，见 [05](05_memory_subsystem.md) |
| byte_offset | 32 | 相对 buffer，必须 B 字节对齐 |
| wdata | MEM_DATA_W | 写数据；读时为零 |
| wstrb | B | 写有效字节；读时为零，不能越过 buffer 末尾 |
| tag | 8 | 客户端本地 tag |

| mem_rsp_t 字段 | 位宽 | 语义 |
|---|---:|---|
| rdata | MEM_DATA_W | 读结果；写响应为零 |
| tag | 8 | 原请求 tag |
| error | 1 | 地址 / 实现错误 |
| reason | 8 | 具体错误，0 无错 |

写请求也恰好有一个响应。`write_ack` 是该写请求响应成功握手且 error=0；不是请求被接受。引擎若已经收到 error，后续排空响应仍计数，但不能再将其报告为成功工作量。

`sram_wrapper` 的底层 RAM 端口是固定延迟的 enable/address/data 接口；仲裁器负责在发物理请求前预约响应寄存器，不能将可背压的逻辑响应直接映射为可背压 RAM。

## 6. 全局资源与完成定义

首版支持一个计算任务（MPU / VPU / SFU三选一，LAYOUT复用VPU）和一个DMA任务并行。结果写回属于该计算任务，未写确认前保留计算任务和目的buffer锁。

completed_units补充：REDUCE成功为1个标量，SINCOS成功为2*count个输出元素，LAYOUT为源有效元素数且不含padding。所有计数先扩位检查32bit上限。

所有指令按序发射，但完成允许按 job_id 区分而非按指令顺序返回。后继任务只依据可见完成事件：MPU / VPU / SFU 的最后目的写确认，LOAD 的最后目的写确认，STORE 的外部写确认，LOAD_W 的 core 域完整 tile 提交确认。

事件槽不是无限历史表。普通任务发射时占槽，完成后保留结果；软件在所有消费者已发射后用 `EVENT_RELEASE` 显式释放。不能仅因主机接收完成记录就清除依赖状态，也不能未清槽就复用 job_id。

## 7. 通用验证断言

1. 每条已接受任务恰好一条完成，无 tag 串入下一任务。
2. 所有 ready/valid 接口在背压时稳定；任何脉冲结果有保证容量的接收者。
3. 请求、响应、写确认分开计数，不存在读响应无归属或写已发即完成。
4. 无请求跨出 buffer；所有计数乘法先扩位，再校验地址和长度。
5. 错误和复位不使旧 buffer、旧事件或旧 bank 重新被视为有效。

后续逐模块端口同样遵守本文件；若修改共享类型，必须同时更新所有生产者和消费者。

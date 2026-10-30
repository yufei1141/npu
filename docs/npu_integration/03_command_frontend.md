# 03 命令前端 command_frontend

状态：v0.1 建议接口。仅 clk_core 域；主机异步接口由平台适配，不直接跨域接入本模块。

## 1. 职责与端口

接收完整 512 bit 指令，做静态解码和字段检查，保存原始 job 身份，按顺序提供给调度器。非法指令也进入同一顺序流，避免旁路错误在前面正常指令之前改变依赖表。

| 信号 | 方向 | 位宽/类型 | 含义 |
|---|---|---|---|
| clk_core / rst_core_n | I | 1 / 1 | 时钟和复位 |
| accept_enable | I | 1 | 顶层RUN且双域就绪、无fatal时允许接新指令 |
| cmd_valid / cmd_ready | I / O | 1 / 1 | 主机命令握手 |
| cmd_data | I | 512 | [02](02_instruction_set.md) 编码 |
| issue_valid / issue_ready | O / I | 1 / 1 | 向调度器提供 FIFO 队头 |
| issue_cmd | O | 512 | 原始指令，不在出队前修改 |
| issue_error | O | 16 | 静态校验错误，正常 0 |
| issue_identity_error | O | 1 | 接收时已与更早排队／调度持有命令重名；强制无槽拒绝 |
| held_valid/held_job | I | 1/16 | scheduler当前持有但尚未发射／拒绝提交的命令 |
| fifo_level | O | ceil(log2(CMD_FIFO_DEPTH+1)) | 已接受未出队数量 |
| accepted_count / rejected_count | O | 各 64 | 诊断计数，饱和 |

`cmd_ready = accept_enable && fifo_has_space`。首版可以不利用满 FIFO 同拍出队腾出的空间，允许一拍保守停顿。接受即存下 cmd 和静态检查结果；host 后续改 cmd_data 不影响已接受指令。

accept_enable只控制入队。issue_valid由FIFO非空产生，队头及valid在背压时保持；RUN/DRAIN由scheduler继续消费，FAULT时scheduler不再给出issue_ready。不得将issue_valid也与accept_enable相与，否则停机后无法排空已接受队列。

## 2. 指令检查分层

静态检查：version/opcode/subop、保留位、字段枚举、零维度、格式组合、tiles/k_steps 参数上限、已知对齐、指令自身地址范围扩位。硬件能力关闭的 SFU op 返回 UNSUPPORTED。

调度检查：job_id 是否与存活任务或未消费完成记录冲突、事件槽状态、依赖身份、buffer 实际容量、资源锁。运行时检查：外存响应、真实数据长度、权重 tile 身份和引擎数量一致性。

前端入队时将job_id与FIFO中所有更早有效项（包括本拍即将弹出的旧队头）及held_job比较。重名仍按序入队，但锁存issue_identity_error=1，issue_error=BAD_FIELD；它不会预约别人的事件槽。调度在CHECK另查事件表／未消费CPL，不将当前命令自己或FIFO中更年轻命令当作冲突。这样两条同ID命令按先后归属判定，不会因“扫描了整个队列含自己”把每条命令误判为重复。

边界沿以旧状态判断，允许保守拒绝恰在同拍被消费／释放的ID。job_id重用时软件须确认EVENT_RELEASE已完成且旧CPL已消费，再提交新命令；不要依赖同拍旁路。

前端不能仅做静态校验就认为所有请求已安全；buffer 配置只读但容量检查仍由调度统一执行。

## 3. FIFO 与状态机

```text
RESET → ACCEPT_DISABLED（等待 accept_enable） → ACCEPT_ENABLED
ACCEPT_ENABLED --accept_enable=0--> ACCEPT_DISABLED
```

两种接受状态都可处理dequeue，只有ACCEPT_ENABLED可enqueue：`level_next=level+push-pop`，同拍进出不能用两个互相覆盖的赋值。读写指针只按相应fire更新。ACCEPT_DISABLED保留队列并允许正常排空；协调reset才丢弃全部在途命令，系统软件需重建任务。

没有“等执行完成”的前端状态；issue_fire 仅表示调度器接管命令，前端便可弹出。

## 4. 错误与交互

前端对非法指令不触发引擎，调度将它转成完成事件。完成通道堵塞时，该错误命令保留在队头。若事件槽或 job 身份本身非法，错误完成使用 `event_slot=0xFF`，不修改其他槽。

正常指令 issue_error=0。调度锁存完整队头后才能拉 issue_ready，不能在未接管 metadata 时先弹出。

## 5. 验收

连续满速入队、满/空边界、同拍进出、accept_enable抖动／关闭后排空、队头长背压、非法opcode和保留位、接收后立即改变host payload；检查前端从不重复或跳过指令。

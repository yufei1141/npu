# 13 完成事件、依赖表、故障与性能计数

状态：v0.1 设计。可见完成在本模块提交，host接收可以更晚。

## 1. 端口

所有端口属 core 域。五个事件源：DMA、MPU、VPU、SFU、ADMIN；MPU源已经包含result_writeback汇总，不将core_done另接成第六个成功事件。

| 信号 | 方向 | 类型 | 含义 |
|---|---|---|---|
| clk_core/rst_core_n | I | 1/1 | 时钟 / 复位 |
| src_valid/src_ready/src_event | I/O/I | 5×(1/1/128) | 可背压事件源 |
| alloc_valid/ready/slot/job/opcode/irq | I/O/I | 1/1/8/16/8/1 | 调度预约普通任务事件槽 |
| lookup_slot/job | I | 8/16 | 依赖查询 |
| lookup_state | O | 3 | FREE/RUNNING/OK/FAILED/IDENTITY_MISMATCH |
| slot_probe_slot/slot_probe_free | I/O | 8/1 | 空槽探测，不分配、不改变状态 |
| id_query_job/id_query_busy | I/O | 16/1 | 有效事件表或任一未消费CPL项存在相同ID则busy |
| admin_action/admin_target_slot/job/admin_irq | I | 2/8/16/1 | ADMIN源sideband；NONE=0、RELEASE=1 |
| cpl_valid/cpl_ready/cpl_data | O/I/O | 1/1/128 | 主机完成记录 |
| retire_valid/retire_event | O | 1/128 | 无背压的单拍提交广播 |
| irq | O | 1 | 队列中仍有IRQ标记的完成，或fatal |
| fatal_in / fatal_latched | I/O | 1/1 | 不可恢复故障锁存 |
| cpl_level / active_slots | O | 各参数计数宽 | 诊断 |

lookup 可组合查看已注册表项；若与本拍retire/alloc同槽，定义为读取旧状态、下一拍可见，调度允许一拍等待。错误时不能组合绕过registered done提前解依赖。

## 2. 事件槽状态机

每槽保存 valid、job_id、opcode、irq_on_complete、state、status、completion_enqueued：

```text
FREE --alloc_fire--> RUNNING
RUNNING --成功事件入CPL--> OK
RUNNING --失败事件入CPL--> FAILED
OK/FAILED --合法EVENT_RELEASE--> FREE
```

RUNNING槽只接收匹配job/opcode的终结事件。FREE槽收到普通引擎事件，或已完成槽再次收到相同完成，均为INTERNAL并fatal，不能重复产生成功通知。

行政拒绝可使用0xFF无槽事件；它们进CPL但不写依赖表。普通拒绝若已合法预约槽，则置FAILED。0xFF不是可查询依赖槽。

## 3. 仲裁、提交与背压

事件仲裁为轮询，只有CPL FIFO可接受且身份验证通过时，才给一个源ready。源必须一直保持event。仲裁选择变化不得影响已经发给CPL的holding记录。

同一提交沿原子执行：源event_fire、CPL enqueue、槽置OK/FAILED、retire广播、调度释放正常资源。CPL host ready可以为0，队列只要有空位仍可提交；满时事件保留在源，资源也暂不释放。

成功事件代表目的数据已经可见。任何引擎若尚有写确认未到，不得先提交事件来腾资源。host的cpl_fire仅弹记录和更新pending_irq，不再次retire或清槽。

## 4. FENCE 与 EVENT_RELEASE 行政控制

ADMIN在scheduler内执行等待，完成管理器执行原子表更新和CPL提交。合法FENCE只在旧任务终结后发事件，它的槽与其他普通任务相同；成功／DEP_FAILED由04的barrier_error区间汇总决定，不能把更早已完成的失败丢掉。

EVENT_RELEASE的target检查与清槽必须和其行政完成入CPL原子完成；CPL满时ADMIN源ready=0，不能先清槽后丢掉行政完成。释放target通过admin_action[1:0]和admin_target_slot/job随ADMIN事件保持到fire。若内部实现命名release_valid/ready/slot/job，它们只能是此原子事务的内部派生信号，不再暴露可单独fire的外部释放口。

软件应在所有目标事件消费者已发射后释放。目标完成记录即使已入CPL但尚未被host读取，也可释放依赖槽；job_id仍不能与该CPL记录冲突。实现维护一个未消费CPL job_id集合，加上有效事件表及调度在途记录，供动态重复ID检查。

id_query_busy对事件表及CPL有效项做逐项比较；行政拒绝可能因重名而携带相同job_id，所以CPL身份集合实际须按每个队列项／引用计数维护，不能弹出一条就删掉同ID其他项。队列内更早身份由03的入队比较处理，当前held命令只在分配前查询一次；不把自己刚alloc的槽误判为新冲突。

alloc与ADMIN释放同拍命中同槽时不旁路复用，下一拍才可分配。ADMIN无槽记录的IRQ来自admin_irq，普通记录来自预约槽；CPL内部另存irq bit，不改变128bit主机记录。所有有效事件槽RUNNING/OK/FAILED都计入active_slots诊断，不能将active_slots直接当FENCE的运行中任务数。

## 5. 故障与恢复

可恢复：静态拒绝、缺失/失败依赖、能力不支持、VPU/SFU可排空的数值错误。FAILED依赖不能使后继读到部分目的数据；软件通过新的写任务和新事件建立下一代有效数据。

不可恢复：MPU启动后无法补齐A/B、结果槽溢出、CDC所有权矛盾、丢失帧边界、未知memory响应或内部tag不一致。fatal必须在相应错误事件对完成管理器可见前锁存，调度在该事件retire时保留/隔离锁，而非将其当正常失败释放。

io故障随终结mailbox的额外fatal位一起跨域；core先锁存fatal再允许该错误event_valid，避免一个独立同步位晚于完成到达。若链路失效无法终结任务，仍输出sticky fatal诊断，不编造成功completed_units；需要协调reset清空未完成任务。

协调恢复流程：停止接新指令→平台停止/隔离旧外存事务并确认→两域复位→各域同步释放→双方ready确认→重新配置buffer/重新装载。旧job不能在reset后复活；仅清SFU flush、仅清CPL或仅清一个bank均不构成恢复。

## 6. 性能计数与读取

64bit饱和计数建议：core_cycles、io_cycles、cmd_accepted、cmd_dispatched、cpl_committed、dma_payload_bytes、dma_padding_bytes、io_backpressure_cycles、wbank_wait_cycles、mpu_a_stall、mpu_b_stall、mpu_compute_beats、result_queue_full、mem_bank_conflicts、vpu_active_cycles、sfu_wait_cycles、fault_count。

io计数不能跨域直接组合读64bit；状态读请求触发snapshot mailbox，io锁存快照后经握手返回core。多个计数不承诺同一物理时刻，但同一次io快照内一致。清计数只在idle或明确snapshot版本边界执行，不影响任务控制。

irq为电平：未消费IRQ完成数>0或fatal则1；host读走对应记录后数量递减。不能为每条完成发一个可能丢失的窄异步脉冲。后续若host异步，在平台层同步电平。

## 7. 验收

五源同拍、CPL满、同拍入出、完成与依赖查询邻拍、job错误/重复完成、释放已完成槽、释放仍运行槽、FENCE不等自己、host长时间不读、irq直到记录消费、fatal先于事件可见、io快照一致性、协调reset清掉旧所有权。

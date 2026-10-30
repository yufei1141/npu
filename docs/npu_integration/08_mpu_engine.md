# 08 MPU 任务适配、A 操作数打包与现有核接入

状态：v0.1 外围设计；gemv_stream/gemv_cluster 现有端口不改。

## 1. 层级与职责

```text
mpu_engine
  task_adapter：锁存 job、预留结果槽、启动和完成计数
  activation_packer：scratchpad → A beat
  B tile controller：依序选择 weight_reader 的 tile
  gemv_stream：现有 A/B FIFO + sequencer + cluster
  result_capture：脉冲结果 → 可背压 tile 流（见 09）
```

MPU 不直接发外存请求。每个 job 的 M 只包含 1..4 行，N 分 tiles 个相同最大宽度，K 为完整归约维度。

## 2. mpu_engine 外层端口（core 域）

| 信号 | 方向 | 位宽/类型 | 含义 |
|---|---|---|---|
| clk_core / rst_core_n | I | 1/1 | 时钟 / 复位 |
| task_valid / task_ready / task_cmd | I/O/I | 1/1/512 | 调度输入 |
| event_valid / event_ready / event_data | O/I/O | 1/1/128 | 全 job 写回确认后的事件 |
| mem_req_valid/ready/req | O/I/O | mem_req_t | A 读取，客户端 MPU_A |
| mem_rsp_valid/ready/rsp | I/O/I | mem_rsp_t | A 响应 |
| w_select_valid/ready/job/tile/mode/g/s | O/I/O | 1/1/16/16/2/4/16 | 接 weight_reader |
| w_b_valid / w_b_ready / w_b_data | I/O/I | 1/1/2304 | B 输入 |
| w_tile_done_valid/ready/status | I/O/I | 1/1/16 | bank 已安全释放 |
| wb_job_valid/ready/cmd | O/I/O | 1/1/512 | 为写回配置 job 及预留结果槽 |
| wb_done_valid/ready/status/tiles | I/O/I/I | 1/1/16/32 | 整 job 的写确认汇总 |
| fatal | O | 1 | 不能安全排空的运行错误 |

task_ready 仅在 IDLE、结果预约条件可满足、写回上下文空闲时拉高；真正接受后仍须通过 wb_job_fire 完成预约，再允许 core start。

## 3. 现有 gemv_stream 逐信号映射

| 现有端口 | 相对 gemv_stream | 外围驱动/采样规则 |
|---|---|---|
| clk/rst_n | I，1/1 | clk_core/rst_core_n |
| mode | I，2 | 从任务锁存，直到任务写回终结保持 |
| m_len/n_grp | I，3/4 | 锁存有效行与前缀列组 |
| tiles/k_steps | I，各16 | 锁存；禁止 0 |
| start | I，1 | 预约成功后一拍脉冲，任务只发一次 |
| a_wr/a_din | I，1/1024 | `a_wr=a_valid&&a_ready&&allow_a` |
| b_wr/b_din | I，1/2304 | `b_wr=w_b_valid&&b_ready&&allow_b` |
| a_ready/b_ready | O，各1 | 仅表示操作数 FIFO 空间，不表示 job 可启动 |
| acc_out/acc_valid | O，4096/1 | 每个 acc_valid 当拍无条件捕获到预留槽 |
| busy | O，1 | 输入 sequencer 在运行；只用于诊断 |
| done | O，1 | 结果计数完成脉冲，外围锁存为 done_seen |

现有 start 保护只有 `!running`，但最后输入后 running 会先于结果清零，外围必须比该条件更严格。绝不以 !busy 重新 start 或改配置。

## 4. activation_packer 端口与位序

| 信号 | 方向 | 类型 | 说明 |
|---|---|---|---|
| cfg_valid/ready/cmd | I/O/I | 1/1/512 | 每 job 一次 |
| mem_req/rsp 通道 | O/I | 公共 memory 类型 | 一在途读取 |
| a_valid/a_ready/a_data | O/I/O | 1/1/1024 | tile-major / K-step-minor |
| a_tile/a_step/a_last | O | 16/16/1 | 诊断和记分板 |
| pack_done / error | O | 1/16 | 全部 A beat 接受 / 错误 |

```text
W4:  A row m, element i → a_data[256*m+16*i +:16], i=0..15
W16: A row m, element i → a_data[256*m+16*i +:16], i=0..3
```

W16 其余 row bit 清零，由 cluster 复制低64 bit。无效行和 K 尾部填零，不读越界元素。A_base、row_stride 是字节，输入元素固定 16 bit，地址为 `A_base+m*A_row_stride+2*(s*K_step+i)`。

FSM：`IDLE → ROW_SETUP → READ_REQ → READ_WAIT → GATHER → ROW_NEXT → HOLD_A → NEXT_STEP/TILE → DONE`。
对每行待收集的有效字节范围，读取其覆盖的 B 字节对齐 memory 字；一个响应可贡献多个相邻元素。跨字最多再发读，收集齐所有有效行才 a_valid=1。HOLD_A 背压时不改 row/step/tile。每个 N tile 首版从 scratchpad 重读 A，外部激活不必重传。

## 5. task_adapter FSM

| 状态 | 行为 / 离开条件 |
|---|---|
| IDLE | 接受 task，锁存所有 cmd / 计数初值 |
| RESERVE | wb_job_fire，预约 tiles 个结果槽，并给 A packer 配置 |
| START | 一拍 core_start；计数器清零、allow_a/b 置有效 |
| FEED | 独立推动 A/B；按 tile 顺序选择并释放权重 bank |
| DRAIN | 输入计数达标后，等待 done_seen 和全部结果被捕获 |
| WRITEBACK | 等 wb_done，确认 tiles 与结果计数一致 |
| COMPLETE | 保持 event_valid，event_fire 后回 IDLE |
| FATAL | 禁止新 A/B/start，保留诊断与可排空响应；协调 reset |

START 前至少保证第一权重 tile 已提交；不要求全部 K-step 已进浅 FIFO。A/B 可以独立被 FIFO 接受，但每个 job 两流各恰好 `T*S` 个；比较和乘积用 32 bit，额外一拍不得进入 FIFO。

B 控制每 tile 发一次 select，等 tile_done 后才请求下一 tile；w_b_ready 除现有 b_ready 外还要求 allow_b、正确 tile 和未达到预期数量。A packer 的 tile_index 独立推进，但绝不能乱序。现有 sequencer 只在 A/B 均非空时取一对。

## 6. 结果与完成计数

独立维护 `a_accepted, b_accepted`（32 bit）、`results_captured, tiles_written`（32 bit）、`done_seen`。结果按 acc_valid 顺序分配 tile_index，不使用此刻已超前的 A/B tile 指针。

结果捕获可在 FEED 时发生，写回也可同时进行；表中状态只是主控制，不把输出服务禁止到 DRAIN 才开始。

成功事件条件：`a_accepted=b_accepted=T*S`、`results_captured=tiles_written=T`、done_seen=1、A/B 原有 FIFO 应已排空、最后 bank 释放已完成。completed_units=T，aux=0。

## 7. 异常

非法参数在 start 前拒绝。start 后缺权重、内存错误或数量错会使现有 core 无法正常完成；不支持靠 flush 取消，进入 fatal/RESET_REQUIRED，目的 buffer 不视为有效。
算术核自身的近似、有限数范围不由外围重新定义；[12](12_format_and_numeric.md) 记录输入限制与比较参考。

## 8. 验收

四 mode、m=1/4、g=1/2/8、T=1/RESULT_SLOTS、S=1/最大、A/B 独立随机停顿、K/N 尾部、连续 job、busy 已降但 done 未到、core_done 与最后捕获同拍、结果长期背压和配置始终稳定。对 first/last 在 cluster 内的实际行为可层级观测，不新增其外部端口。

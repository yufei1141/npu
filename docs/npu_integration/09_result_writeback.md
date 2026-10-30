# 09 MPU 结果捕获、分拍与写回

状态：v0.1 建议设计；首要约束是现有 acc_valid 无 ready。

## 1. 端口与所有权

| 信号 | 方向 | 位宽/类型 | 含义 |
|---|---|---|---|
| clk_core/rst_core_n | I | 1/1 | 时钟 / 复位 |
| job_valid/job_ready/job_cmd | I/O/I | 1/1/512 | MPU 预约结果和设置目的上下文 |
| acc_valid / acc_data | I | 1/4096 | 现有核脉冲输出，无反压 |
| mem_req_valid/ready/req | O/I/O | mem_req_t | RESULT_WB 客户端 |
| mem_rsp_valid/ready/rsp | I/O/I | mem_rsp_t | 写确认 |
| done_valid/done_ready/status/tiles | O/I/O/O | 1/1/16/32 | 全部目的写确认 |
| reserved_count/occupied_count | O | 各 ceil(log2(RESULT_SLOTS+1)) | 槽状态 |
| fatal | O | 1 | 无预约捕获、溢出或无法恢复写错误 |

一个 job 上下文持有 D_base/D_row_stride/m_len/g/N_valid/tiles/dst_fmt。只有 job_fire 才从空闲空间中一次预约 T 槽；job_ready 要求无旧上下文且空槽≥T。

## 2. 捕获与 FIFO 不变量

设 C 总槽、R 已预约未来结果、O 已捕获未释放槽：

```text
0 <= R+O <= C
reserve T: R += T
capture:   R -= 1; O += 1
release:   O -= 1
```

同拍事件统一算 next 值。每个槽保存 4096 bit、job_id、tile_index、有效行/列数。acc_valid 时必须有 R>0 和对应空槽，无条件当拍捕获，不能等 memory ready。

本首版选择保守释放：该槽所有有效元素写确认后才释放，避免额外 4096 bit 隐藏 holding register；因此 O 包含当前正被 serializer 使用的槽。不能“出 FIFO 就释放”又在原槽里继续读取数据。

捕获与写回可同拍访问不同槽。RESULT_SLOTS=1 时按顺序串行，预留本身仍保证不溢出。

## 3. tile 到 tensor 的地址映射

当前结果始终是物理 4×32 FP32，位序：

```text
value(m,c)=acc_data[32*(32*m+c)+:32]
valid_cols(t)=min(4*g, N_valid-4*g*t)
dst_addr(m,c,t)=D_base + m*D_row_stride + (4*g*t+c)*E_dst
```

仅 m<m_len、c<valid_cols 写出。失活组和组内 N 尾部不写目的。转换 FP16/BF16 后 E_dst=2，FP32 为4，不能沿用物理 512 B 总长度来写所有尾部。

目的地址只要求元素对齐；serializer 生成 B 字节对齐的 mem_req.byte_offset，并把元素放到 `dst_addr mod B` 位置的 wdata，设置对应字节 wstrb。行与 tile 可能共享一个 memory 字，必须保持其他字节原值，不能为方便写整字零填充。

## 4. 写回 FSM

| 状态 | 行为 |
|---|---|
| WAIT_TILE | 等 FIFO 有已捕获槽，锁住该槽 |
| BUILD_WORD | 按 row/col 取 FP32，调用转换器，累积当前目的对齐字及 strobes |
| CONVERT_WAIT | 等转换返回，不假定单拍；只允许一元素转换在途 |
| WRITE_REQ | 保持完整 wdata/wstrb/addr，等 req_fire |
| WRITE_ACK | 等 rsp_fire；成功累计有效字节/元素，否则锁存错误 |
| ADVANCE | 同字继续收集或推进字/行；tile 全部确认 → 释放槽、tiles_written++ |
| JOB_DONE | 已捕获 T、已写 T、R=O=0，保持 done_valid |
| ERROR | 停止新写；已有响应排空；fatal 通知，不把槽重新借给新 job |

高吞吐可改为多个转换 lane 或多写在途，但必须增加对应顺序、tag 和错误恢复结构。首版 FSM 不保证输出每拍一个元素，也不把 MEM_DATA_W=256 当成已验收写回吞吐。

## 5. 转换器合同

使用 [12](12_format_and_numeric.md) 的 RNE/FTZ 与数值状态。FP32 目的可绕过舍入，但仍保留允许数值范围检查。发生 overflow/invalid 时不写该字，已写的部分也不能宣布整个 tensor 成功；MPU 后续不可背压输出仍需捕获或协调 reset，故此任务进入 fatal。

写回 done 与 core done 不互相替代。mpu_engine 汇合 core 数量、结果捕获和此处写确认后才提交完成。

## 6. 验收

每拍 acc_valid 的突发极限、同拍捕获/槽释放、输出长停顿、FP16/BF16/FP32、非字对齐但元素对齐的 D_base、行 stride、N=1/3/4/5/31/32、多 tile 连续列、周边哨兵字节不改变、写确认延后、无预约 acc_valid、错误后仍不得覆盖在途槽。

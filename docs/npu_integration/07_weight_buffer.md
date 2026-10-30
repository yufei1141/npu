# 07 权重装填器、双时钟 bank 与读取器

状态：v0.1 可据此展开 RTL 的建议合同。带宽与容量背景见 [weight_path](weight_path.md)，本文件定义信号与顺序。

## 1. tile 身份、布局与 RAM

tile 由 `(consumer_job_id[15:0],tile_index[15:0])` 唯一识别，同时核对 mode[1:0]、n_grp[3:0]、k_steps[15:0]。配置不符不是“读别的 bank 试试”，而是流身份错误。

每个 ping/pong bank，每组 j 存 `weight_lo/hi[j]`，各 128 bit×WBUF_K_STEPS；另有 `scale[j]`，128 bit×ceil(WBUF_K_STEPS/4)。IO 写、core 同步读，禁止同 bank 所有权重叠。

外部连续帧：先 j-major、s-minor 的权重区，每 `(j,s)` 两个 128 bit 低/高半字；再 j-major 的 scale 区，每个 128 bit 保存四个 step 的 32 bit scale，组尾补齐。W16 无 scale 区。

```text
W4:  W_bus=32*g*S + 16*g*ceil(S/4)
W16: W_bus=32*g*S
weights(j,s,h)=16*(2*(j*S+s)+h)，h=0/1
scale_base=32*g*S
scale_word(j,q)=scale_base+16*(j*ceil(S/4)+q)
```

四个 E8M0 的原始字节按 PE0..PE3 从低位到高位；不进行数值转换。末尾补齐 scale 不参与计算。

W4组内位序以当前pe.v为准，特别注意每16bit行的输出列0在最高nibble：

```text
group_weight[64*p+16*i+12-4*c +:4] = W[16*s+4*p+i, 4*j+c]
group_scale[8*p +:8] = scale_for_K4_block[p]
p=0..3（PE／K四元素块），i=0..3，c=0..3（组内输出列）
```

INT4为4bit二补码，范围-8..7。每PE的E8M0 scale作用于这个K四元素块的四个输出列，不是每列独立FP16 scale。K/N不足的位置在离线帧补0；禁止把w_i0放低nibble。W16则是每PE一输出列、低到高四个16bit K元素，完整公式见16。

## 2. weight_loader 端口（io 域）

| 信号 | 方向 | 位宽/类型 | 含义 |
|---|---|---|---|
| clk_io / rst_io_n | I | 1 / 1 | 时钟 / 复位 |
| wload_valid / wload_ready / wload_cmd | I/O/I | 1/1/512 | LOAD_W 指令，握手时选空 bank |
| data_valid / data_ready / data | I/O/I | 1/1/128 | DMA 已校验 tag/keep 的输入 |
| data_last / data_error | I | 1 / 1 | 帧结束 / 错误 |
| done_valid / done_ready / done_status | O/I/O | 1/1/16 | 提交完成 / 拒绝，不是 bank release |
| bank_wr_en | O | 2 | 每拍最多一个 ping/pong bank 写 |
| ram_kind / group_idx / half_sel | O | 1/3/1 | 0 权重、1 scale；权重低/高 |
| ram_word_addr / ram_wdata | O | RAM_AW/128 | 对应逻辑子 RAM 写入 |
| commit_valid / commit_ready / commit_desc | O/I/O | 1/1/wtile_desc_t | 送各 bank CDC mailbox |
| commit_done_valid/ready/status | I/O/I | 1/1/16 | mailbox源端完整req/ack回零结束，非仅本地提交fire |
| bank_free[1:0] | I | 2 | 本域所有权状态，不是直接看 core 信号 |

`wtile_desc_t` 字段：bank_id 1、consumer_job_id 16、tile_index 16、mode 2、n_grp 4、k_steps 16、weight_bytes 32、scale_bytes 32、layout_version 8（固定1）。这是逻辑结构，不继承 cmd 中保留位。

## 3. loader 状态机

| 状态 | 操作 / 转移 |
|---|---|
| IDLE | wload_ready 要求空 bank；握手后锁存并验证 g/S/长度 |
| WEIGHTS | data_fire 写一个 lo/hi，更新 half→step→group；最后权重后 W4→SCALES，W16→PUBLISH |
| SCALES | 每 data_fire 写一个 scale RAM 字，q→group；最后一个→PUBLISH |
| PUBLISH | 全部 RAM 写沿完成后发布 commit，payload 保持 |
| WAIT_COMMIT | 等 core 已捕获 descriptor 的完整提交握手结束 |
| DONE | 保持 done_valid/status，接受后回 IDLE；bank 仍非 FREE |
| BAD_FRAME | 禁止提交，丢弃到可信帧末；未发布 bank 可本域回收；无可信帧末则 fatal |

最后一个 data_fire 才允许设置“全部写完”，不能同拍组合穿透 commit 绕过 RAM 写沿。data_last 必须精确对应最终字；长度错/错误拍不得发布半 tile。重复身份在 core/IO 活跃 bank 中拒绝，避免一个消费者拿到两份同一 tile。

PUBLISH中的commit_fire只表示本地mailbox接管descriptor，随后进入WAIT_COMMIT；必须等commit_done_fire且status=OK才进入成功DONE。完成接口的背压不能使mailbox重复提交；非零status走错误终结，不能发布成功LOAD_W。

## 4. bank controller 与 CDC

每 bank 分别实例化一个提交 mailbox（io→core）和释放 mailbox（core→io）。建议采用完整四相 req/ack 电平协议，避免把 req 脉冲直接跨域：

```text
源：IDLE → payload 锁存、req=1 → 等同步 ack=1 → req=0
    → 等同步 ack=0 → 本次 handshake 完成 → IDLE
目的：等同步 req=1 → 在有容量时采样 payload、ack=1
      → 等同步 req=0 → ack=0 → IDLE
```

payload 从 req 提高前已稳定，并保持到源端完整握手结束。只同步 req/ack 控制，宽 payload 用受握手约束的稳定多周期路径采样。不得使用“每 bit 独立同步后拼成 descriptor”。实现要给足稳定/采样窗口并在平台 CDC 检查中识别此结构。

| 域 | 状态 | 可做的事情 |
|---|---|---|
| io | FREE | 可以分配并写入 |
| io | FILLING | 仅当前 loader 写 |
| io | COMMITTING | 保持提交元数据，不再写 |
| io | CORE_OWNED | 禁止写，等待释放 |
| io | RELEASE_ACK | 接收释放，完成握手后回 FREE |
| core | EMPTY | 接受新 commit |
| core | READY | descriptor 稳定，可匹配消费者 |
| core | READING | 仅当前 reader 可访问 |
| core | RELEASING | 保持 release，完整握手后回 EMPTY |

提交 ack 表示 core 已捕获 tile，不释放 bank。IO 只有释放握手完成才回 FREE。core 若尚在 RELEASING，下一次 commit 保持 pending，直到 EMPTY 才确认，不能靠两个同步链的延迟巧合排序。

## 5. weight_reader 端口（core 域）

| 信号 | 方向 | 位宽/类型 | 含义 |
|---|---|---|---|
| clk_core / rst_core_n | I | 1 / 1 | 时钟 / 复位 |
| select_valid / select_ready | I / O | 1 | MPU 请求一个 tile |
| select_job / select_tile / select_mode / select_g / select_s | I | 16/16/2/4/16 | 精确匹配 |
| b_valid / b_ready / b_data | O/I/O | 1/1/2304 | MPU 的一个 K-step B beat |
| b_step / b_last | O | 16/1 | 诊断 metadata，不送现有 gemv_stream |
| tile_done_valid / tile_done_ready / tile_done_status | O/I/O | 1/1/16 | 最后 B beat 已被接受且释放握手完成 |
| query_job / query_tile / query_match / query_fault | I/I/O/O | 16/16/1/1 | scheduler/MPU 查询 tile0 |
| query_mode/query_g/query_s | I | 2/4/16 | 同身份tile还必须匹配这些配置 |
| ram_rd_en / ram_rd_addr | O | 各组 enable / RAM_AW | 同步读请求 |
| ram_rd_valid / weight_lo_hi / scale_word | I | 1 / 2048 / 1024 | RAM 响应，按组聚合 |
| release_valid / release_ready / release_desc | O/I/O | 1/1/bank+job+tile | 控制 mailbox |
| release_done_valid/ready/status | I/O/I | 1/1/16 | 完整释放握手结束，才发tile_done |

scale_word 聚合为八个 128 bit RAM 输出，共 1024 bit，reader 依 step[1:0] 选择每组 32 bit。reader 的物理 RAM 响应端口没有 ready，必须在发读时预约 holding register。

## 6. reader 状态机与背压

最小可实现版本只有一个在途 RAM 读和一个 B holding register：

```text
WAIT_TILE → ISSUE_READ → WAIT_RAM → HOLD_B
HOLD_B --b_fire 且非末 step--> ISSUE_READ
HOLD_B --b_fire 且末 step--> RELEASE → WAIT_RELEASE → TILE_DONE → WAIT_TILE
```

RAM 响应时将选中的权重/scale 和 step metadata 一起锁存。无效列组填零权重，scale 用单位位型 0x7F7F7F7F；W16 不发 scale RAM enable。`b_valid&&!b_ready` 保持全部 B 与 metadata。

该保守 reader 不能达到每 core 拍一个 B beat。高吞吐版本需要 pipelined read metadata FIFO 和 B FIFO，按“空槽 >= 在途响应数+新请求”预约后连续发读；带宽文档中的 II=1 是该流水化版本的上限，不能宣称上述最小 FSM 已达到。

初版 bank 在最后 B beat 被 MPU 输入 FIFO 接受后不再使用；完成释放握手后才返回 tile_done。即使 MPU A 比 B 慢，B 的副本已在 MPU FIFO，原 RAM 可释放。不能在最后 RAM 请求发出时提前 release。

RELEASE中的release_fire仅表示本地mailbox接管释放请求；WAIT_RELEASE等待release_done_fire，status=OK后进入TILE_DONE。若status非零，不得将bank视为可重用，须锁存fatal并等待协调恢复。

## 7. 复位、错误与交互

reset 不清大 RAM，只清本地有效、指针和 handshake；必须两域协调。若读到不匹配配置或中途 RAM error，不能伪造零权重补齐现有 MPU，须 fatal 并协调恢复。

读完 tile0 后 MPU 可等待 tile1，期间其它 bank 可装填。多 tile 的 B 顺序必须是 tile-major/K-step-minor，不因 bank0/bank1 哪个先 READY 而乱序。

当前单消费者 bank 不支持跨多个 M job 自动复用；需要复用时增加显式引用计数或 keep/release 指令，不能暗中延迟 release 造成 loader 永久等空 bank。

## 8. 验收

g=1/2/8，S=1/3/4/5/容量上限，四 mode；位序、scale 组尾对齐、data_last、两个 bank 反复切换、任意频比、源和目的停顿、release 延后、同身份重复、缺 tile、配置不匹配、输出背压和协调 reset。记分板按原始外部帧重建 b_data，逐 beat 比较，不只检查最后矩阵结果。

控制原语参考：AMD XPM_CDC_HANDSHAKE（完整请求/确认），仅用于核对协议和 CDC 实现，不要求平台无关代码直接实例化厂商宏。

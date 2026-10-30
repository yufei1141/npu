# 02 指令格式与控制语义

状态：v0.1 建议编码；不是现有 RTL 已有 ISA。所有未定义 bit、word、opcode、subop 必须为零或返回错误，不静默忽略。

## 1. 传输与公共头

一条逻辑指令 512 bit，主机侧 `cmd_data[511:0]` 单次握手提交。首版不做可变长取指。字节序为低地址对应低 bit；P0 指 payload 最低 32 bit。

| bit | 字段 | 规则 |
|---|---|---|
| 7:0 | version | 0x01 |
| 15:8 | opcode | 见下表 |
| 23:16 | subop | 按指令定义 |
| 31:24 | flags | bit0 DEP_EN，bit1 INPLACE，bit2 IRQ；其余为 0 |
| 47:32 | job_id | 在存活事件槽、排队指令及未消费完成记录中唯一 |
| 63:48 | dep_job_id | DEP_EN 时匹配依赖槽内 job_id，否则 0 |
| 71:64 | src0_buf | 未使用时 0 |
| 79:72 | src1_buf | 未使用时 0 |
| 87:80 | dst_buf | 未使用时 0 |
| 95:88 | formats | [1:0] src0，[3:2] src1，[5:4] dst，[7:6] accumulator |
| 103:96 | event_slot | 普通指令分配槽；EVENT_RELEASE 用 0xFF |
| 111:104 | dep_slot | DEP_EN 时为有效槽，否则 0 |
| 127:112 | reserved | 0 |
| 511:128 | P0..P11 | 每个 word 32 bit，`Pi=cmd[128+32*i +: 32]` |

格式编码：`00 FP16`、`01 BF16`、`10 FP32`、`11 INT4`。MPU 的 src0=A、src1=W；W4 的 src1=INT4，其 scale 不占此格式字段。DMA 纯字节搬运要求 formats=0，不解释为 FP16 转换。

| opcode | 名称 | subop | 完成条件 |
|---:|---|---|---|
| 0x01 | LOAD | 0 普通 buffer，1 权重 tile | 普通写确认 / 权重提交确认 |
| 0x02 | STORE | 0 | 外存写确认 |
| 0x10 | MATMUL | 0 | 所有 tile 目的写确认 |
| 0x20 | VECTOR | 见第 4 节 | 所有有效元素写确认 |
| 0x21 | REDUCE | bit0 SUM/MAX，bit7 mask_enable | 标量结果写确认 |
| 0x22 | SFU | 见第 5 节 | 全部输出元素写确认 |
| 0x23 | LAYOUT | 0 TRANSPOSE2D，1 PACK_W16 | 所有重排／padding写确认 |
| 0x30 | FENCE | 0 | 所有更早已发射任务可见完成 |
| 0x31 | EVENT_RELEASE | 0 | 清除指定已完成事件槽；本指令不分配事件槽 |

WAIT 以 DEP_EN + FENCE 表达，不再另分配重复 opcode。首版不提供取消、跳转或任意单域复位指令。

## 2. LOAD / STORE

普通 LOAD/STORE 的 P 字段：

| word | 含义 |
|---|---|
| P0 / P1 | 外部起始地址低 / 高 32 bit |
| P2 | 本地 buffer 字节偏移 |
| P3 | row_bytes，1..2^32-1 |
| P4 | rows，1..65535，其他高位为 0 |
| P5 | 外部 row stride，字节 |
| P6 | 本地 row stride，字节 |
| P7..P11 | 0 |

LOAD 使用 dst_buf，STORE 使用 src0_buf，其他 buffer 头字段为 0。首版外部每行基址 16 B 对齐，本地每行基址 B=MEM_DATA_W/8 对齐；row_bytes 可非整字。rows=1 时 P5/P6=0；多行时 stride 必须不小于 row_bytes，且保持相应对齐。`row_bytes*rows` 不得超过完成计数的 32 bit 上限。地址结束计算使用 65 bit / 64 bit 扩位检查。

LOAD subop=1 为 LOAD_W，header src/dst buffer 均为 0，改用：

| word | 含义 |
|---|---|
| P0 / P1 | 紧邻 weight 区和 scale 区的外部起始地址，16 B 对齐 |
| P2 | [15:0] consumer_job_id，[31:16] tile_index |
| P3 | [1:0] MPU mode，[11:8] n_grp，[31:16] k_steps，其余 0 |
| P4 | 精确传输总字节数，按 [07](07_weight_buffer.md) 布局计算 |
| P5..P11 | 0 |

bank 由 loader 在空闲 ping/pong 中分配，不由软件猜测 bank。LOAD_W.job_id 是搬运命令自己的 ID；consumer_job_id 是未来/当前 MATMUL 的 ID。完成依赖只能指向已发射指令，但 consumer_job_id 可以预先指定。未来 job 的第一 tile 可先加载，计算开始后加载下一 tile。

## 3. MATMUL：一次 M tile、连续 N tiles、完整 K

| word | 含义 |
|---|---|
| P0 | [7:0] m_len 1..4，[15:8] n_grp 1..8，[17:16] mode，其余 0 |
| P1 | [15:0] tiles 1..RESULT_SLOTS，[31:16] k_steps 1..WBUF_K_STEPS |
| P2 | K_valid，真实 K 元素数，必须 `ceil(K_valid/K_step)=k_steps` |
| P3 | N_valid，job 总有效列，`4*g*(tiles-1)<N_valid<=4*g*tiles` |
| P4 | A_base，src0_buf 内字节偏移 |
| P5 | A_row_stride，字节；至少 2*K_valid，首行元素对齐即可 |
| P6 | A_tile_stride，首版必须 0，N tile 间重用相同 A |
| P7 | D_base，dst_buf 内字节偏移 |
| P8 | D_row_stride，至少 N_valid*dst_element_bytes |
| P9..P11 | 0 |

src1_buf=0；权重以 `(consumer_job_id=job_id,tile_index)` 从 weight_reader 获取。A / D 不允许同 buffer。mode=00 W4FP16，01 W16FP16，10 W4BF16，11 W16BF16；必须与 formats 相符，accumulator=FP32，dst 为 FP16/BF16/FP32。

每 job 不跨 M tile。更大 M 由软件发多个 job；每个 job 的配置保持到最后写确认。完整 K 放不下时拒绝，不将 K 分块伪装成不相关 job 后丢失部分和。

## 4. VECTOR / REDUCE

VECTOR subop低6bit：`00 ADD`、`01 SUB`、`02 MUL`、`03 MAXIMUM`、`04 SELECT`、`05 CONVERT`、`06 MUL_SCALAR`、`07 ADD_SCALAR`、`08 COMPARE_GT`、`09 ROTATE_PAIR`。bit7为src1_scalar_broadcast，bit6为src1_pair_broadcast，两者互斥。

ADD/SUB/MUL/MAXIMUM支持两种广播；SELECT仅支持bit7，表示false候选src1为标量；其余操作必须bit7:6=0。广播地址索引：标量始终0；成对为floor(i/2)；不广播为i。src0总按i递增。广播源的范围检查按真实元素数1或ceil(count/2)，不能仍要求src1保存count个元素。成对广播要求count偶数。

| word | 含义 |
|---|---|
| P0 | elem_count，1..2^32-1 |
| P1 / P2 / P3 | src0 / src1 / dst 起始字节偏移 |
| P4 | 标量 FP32 原始位型；仅 scalar 操作使用 |
| P5 | SELECT 的 mask buffer_id 低 8 bit，其余 0 |
| P6 | SELECT 的 mask 字节偏移，每元素 1 B，0=false，非零=true |
| P7..P11 | 0 |

CONVERT、ROTATE_PAIR和scalar立即数操作不读src1，src1_buf/对应格式/P2必须0。VECTOR accumulator=FP32，普通源／目的支持FP16/BF16/FP32，不允许INT4。P4仅立即数操作使用，其余0；P5/P6仅SELECT使用，其余0。COMPARE_GT的输出是每元素一字节0/1，formats.dst必须0，属该subop特殊字节语义。SELECT使用显式第三mask buffer，不把src1当成mask又当成候选值。其他算术输出按formats.dst转换。

ROTATE_PAIR要求count偶数，输入相邻(x[2j],x[2j+1])，输出(-x[2j+1],x[2j])，用于interleaved RoPE；无效数仍遵守12的数值合同。首版要求V_LANES为偶数且≥2以支持该能力，否则返回UNSUPPORTED。

仅 CONVERT 同宽或逐元素 ADD/SUB/MUL/MAXIMUM/scalar 可申请 INPLACE，且必须精确同地址同跨度；错位别名和 SELECT 原地首版拒绝。源与目的元素宽度不同时一律不允许原地。

广播模式不允许目的与广播src1同buffer，避免覆盖要复用的系数；合法原地只允许dst精确覆盖src0，且其余源不被覆盖。ROTATE_PAIR首版禁止原地。

REDUCE subop bit0=0 SUM／1 MAX，bit7=mask_enable，bit6:1=0。使用P0=count、P1=src偏移、P3=dst偏移；启用mask时P5低8bit=mask buffer、P6=mask byte偏移，否则P5/P6=0，其他P字段均0。mask每元素1B，非零参与；src1_buf及格式为0，src0允许FP16/BF16/FP32，accumulator/dst=FP32，恰好输出一个标量，completed_units=1。

空count不合法。masked MAX只对有效元素维护has_value，不制造-Inf；若全被mask，返回NUMERIC且不写标量。masked SUM全被mask时输出+0；无效元素不读取／不参与数值异常。REDUCE不允许原地。

### LAYOUT 补充

opcode=0x23复用VPU执行资源；完整P字段和地址算法见[16](16_tensor_layout_and_kv.md)。TRANSPOSE2D的P0..P5依次为rows、cols、src_base、dst_base、src_row_stride、dst_row_stride；PACK_W16依次为K_valid、N_valid、src_base、dst_base、src_row_stride、n_grp。P6..P11=0；均禁止INPLACE，src1_buf及其格式为0、accumulator位为0。

TRANSPOSE2D源目的格式相同，FP16/BF16/FP32原始元素置换；PACK_W16只允许相同FP16/BF16，输出07规定的group-major原始B帧，不做W4在线量化。完成单位为源有效元素，不含padding。

## 5. SFU

逻辑 subop：`00 EXP2`、`01 RCP`、`02 RSQRT`、`03 SINCOS`、`04 LOG2`、`05 SQRT`、`06 SIN`、`07 COS`、`08 TANH`。这些是新命令层编码，不假定与当前 sfu_top.op 数值一致（映射见 [11](11_sfu_adapter.md)）；`00`-`03` 的编码与早期版本一致，后加的功能不复用它们的编号。subop>08 按 BAD_FIELD 拒绝。硬件功能不在 capability 表中则返回 UNSUPPORTED；capability 为每个 subop 一位（[8:0]）。

P0=count、P1=src 偏移、P3=dst 偏移；P2=第二输出偏移，仅 SINCOS 使用；P4 低 2 bit 为 function_mode，其余为 0；P5..P11=0。SINCOS 两个输出平面都在 dst_buf：dst 偏移为 sin，第二偏移为 cos，不允许重叠。src1_buf=0，INPLACE=0。

最终 SFU 的 op/fmt/mode、SINCOS 帧形式和异常合同尚待对齐；[11](11_sfu_adapter.md) 定义对齐位置及适配条件，不将当前占位数值路径当成已支持。

## 6. FENCE 与 EVENT_RELEASE

FENCE 的全部 P 字段、buffer、formats 为 0，INPLACE=0。可设依赖，但即使未设，也等待所有更早已发射任务完成；不等待主机消费完成记录。FENCE 普通占事件槽，适合构造多个生产者的汇合依赖。

FENCE同时汇总自上一次合法FENCE提交以来的错误：区间内任一任务／拒绝记录失败则本FENCE返回DEP_FAILED，后继不能绕过失败生产者。详细区间规则见04；不是只看当前仍在运行的任务，也不因目标槽已释放就忘记先前失败。

EVENT_RELEASE：event_slot=0xFF，DEP_EN 可用；P0=[7:0] target_slot、[23:8] target_job_id、[31:24]=0，其余 P/buffer/formats=0。目标必须匹配已完成槽，且目标完成记录已经入完成 FIFO；仍运行则等待，匹配失败则返回 BAD_FIELD，不清除其他 job。软件保证所有依赖目标的指令已被发射；首版不为未来消费者保留已释放事件。

job_id 在主机尚未收走其完成记录时仍不能复用。EVENT_RELEASE 也产生一条行政完成，但不创建依赖槽；不能成为 DEP 的生产者。

## 7. 发射例子与队头阻塞

```text
LOAD_W  job=10 slot=0  consumer=11 tile=0
MATMUL  job=11 slot=1  DEP=(slot0,job10) tiles=2
LOAD_W  job=12 slot=2  consumer=11 tile=1
FENCE   job=13 slot=3
EVENT_RELEASE job=14 slot=FF target=(slot0,job10)
```

调度器接受 MATMUL 到引擎后立即继续下一条，故 LOAD_W tile1 能与计算重叠。不能把 MATMUL 设置为依赖 tile1 的 LOAD 完成再将该 LOAD 排在它后面；这会造成程序级循环等待。超过结果槽容量需拆 N job，不能把 tiles 编成大数期望自动无限缓存。

## 8. 验证

检查所有 bit/word 越界、保留位、格式与 mode 一致性、字段乘法扩位、事件引用生存期；为每 opcode 构造一个最小合法和一个非法指令；对 LOAD_W 验证 P4 与布局公式一致。不同 word 的默认零不能掩盖一个必填长度为零的错误。

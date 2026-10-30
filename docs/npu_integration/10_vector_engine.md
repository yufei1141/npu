# 10 VPU、归约与向量访存

状态：v0.1 新模块设计。本文定义基础浮点运算，不以整数 ALU 替代 FP16/BF16/FP32。

## 1. 子模块与任务口

vector_engine = task_control + vector_load_store + lane_alu + reduction_unit + format_converter + tensor_layout。每个任务只执行一个VECTOR/REDUCE/LAYOUT，首版一个计算job在途，不与MPU/SFU并发计算。LAYOUT的端口／索引／FSM见[16](16_tensor_layout_and_kv.md)，由锁存的task类型选择本地memory与event mux。

| 信号 | 方向 | 类型 | 说明 |
|---|---|---|---|
| clk_core/rst_core_n | I | 1/1 | core 时钟 / 复位 |
| task_valid/ready/cmd | I/O/I | 1/1/512 | VECTOR、REDUCE或LAYOUT |
| event_valid/ready/data | O/I/O | 1/1/128 | 最后写确认后完成 |
| mem_req_valid/ready/req | O/I/O | mem_req_t | VPU 客户端 |
| mem_rsp_valid/ready/rsp | I/O/I | mem_rsp_t | 按序单在途 |
| fp_exception | O | 8 | 粘滞数值诊断，本任务开始时清 |
| fatal | O | 1 | 存储通路不再能可靠推进 |

向量任务字段、格式、原地限制见 [02](02_instruction_set.md)。默认 V_LANES=4 是待实现建议；所有 lane 先扩展到 FP32，输出由显式转换处理。

## 2. vector_load_store 端口与地址

内部向量包为 `vector_t={data[V_LANES*32], lane_mask[V_LANES], base_index[31:0]}`。mask 表示有效元素，不对应 memory wstrb。

| 信号 | 方向 | 类型 | 说明 |
|---|---|---|---|
| load_cmd_valid/ready | I/O | 1 | 锁存 buffer/base/fmt/count/which_source |
| load_valid/ready/vec | O/I/O | vector_t | 已扩展 FP32 的一组有效元素 |
| store_valid/ready/vec | I/O/I | vector_t | 待按 dst_fmt 压缩的结果 |
| store_done_valid/ready/status | O/I/O | 1/1/16 | 本组全部写确认 |
| mem 通道 | O/I | 公共协议 | 元素对齐地址→对齐字+slice/strobe |

地址为 `base+index*E`，count 尾部 mask 只有低 count mod V_LANES 个有效。SELECT 第三 mask 源是 byte buffer；非零为真，不做浮点解释。

load_cmd逻辑字段为buf_id[7:0]、base[31:0]、fmt[1:0]、count[31:0]、base_index[31:0]、source_kind[1:0]（SRC0/SRC1/MASK）、broadcast[1:0]（NONE/SCALAR/PAIR）、active_mask[V_LANES-1:0]。store配置从锁存task取得dst_buf/base/fmt/count；load/store不能同时占用本地单在途memory口。

广播由subop高bit解码后下发load配置，ALU只接低6bit原语。标量广播读取一次src1[0]并保存到任务结束；pair广播按floor(index/2)取系数，每两个相邻lane复用。count／跨度验证按02，src1_broadcast与src1_pair_broadcast不能同时置1。ROTATE_PAIR先收齐相邻两个元素再交换并取反；不通过浮点乘-1改变舍入或异常约定。

首版每个源向量收集完才读下一个源，不能同时复用唯一响应寄存器保存 src0 与 src1。可缓存一个完整 src0_vec、src1_vec、mask_vec、result_vec，容纳当前组即可，不缓存完整模型向量。

## 3. lane_alu 与 reduce 端口

| 接口 | 方向 | 类型 | 合同 |
|---|---|---|---|
| alu_valid/ready | I/O | 1 | 当前向量输入 |
| alu_op | I | 8 | VECTOR subop，见 02 |
| src0_vec/src1_vec | I | 各 V_LANES×32 | FP32 bit 型 |
| select_bits / lane_mask | I | 各 V_LANES | SELECT 条件 / 尾部有效 |
| scalar | I | 32 | FP32 标量 |
| alu_out_valid/ready/data/status | O/I/O/O | 1/1/V_LANES×32/8 | 可变延迟，背压保持 |
| red_in_valid/ready/vec | I/O/I | vector_t | 当前任务的归约包 |
| red_start/red_last/red_op | I | 1/1/1 | 新任务/最后包/SUM或MAX |
| red_active_mask | I | V_LANES | 尾部mask与可选byte mask的交集 |
| red_out_valid/ready/value/status | O/I/O/O | 1/1/32/8 | 仅一标量结果 |

CONVERT 跳过算术但经过输入/输出类型转换；COMPARE_GT 每 lane 产生0/1 byte，其存储不使用浮点转换器。SELECT 采用 mask ? src0 : src1，只检查被使用候选值的有效格式和范围，不把 mask 当数据相乘。

## 4. task_control 状态机

```text
IDLE → CHECK → [LOAD_MASK] → LOAD_SRC0 → [LOAD_SRC1]
     → EXEC_ISSUE → EXEC_WAIT → [RED_ACCUM / STORE]
     → WAIT_STORE_ACK → NEXT_GROUP → COMPLETE
```

不需要的源状态旁路；REDUCE 不每组写回，只在最后 red_out 后写一个 FP32 标量。EXEC_WAIT 依据实际 out_valid，不用固定计时器猜测浮点模块完成。
COMPLETE 仅在所有元素/标量写确认后产生事件，保持到接受。

SELECT和masked REDUCE先取mask。REDUCE的无效元素不发有效元素读取／转换；若按整字读到了邻接无效元素，其bit不送分类器或归约。SELECT先明确候选选择，再屏蔽未选择候选的数值flags，不能把未选择NaN引发的错误当成输出错误。内存地址合法性仍对声明使用的源范围完整校验，不以mask规避越界。

## 5. SUM/MAX 的确定顺序

为了可复现，首版 SUM 在每个向量包内按 lane0→lane(V_LANES-1) 的顺序将有效值加到一个 FP32 running_sum，从初始 +0 开始；包按 index 递增。可以多周期共享 FP32 adder，不承诺树形归约吞吐。参考模型必须使用相同 FTZ 与逐次舍入顺序。

MAX 用 `has_value` 和 running_max；第一次有效值初始化，其后按 index 比较。尾部无效 lane 不参与。±0 相等时保留较早输入；输入 NaN/Inf 的首版策略见 12，不能依赖一个未实现的全 IEEE compare。

REDUCE.subop bit7启用byte mask。全被mask的SUM输出+0；MAX返回NUMERIC且不写结果。归约仅在整个count处理完才写一次标量，成功completed_units=1；VECTOR按确认输出元素数计，LAYOUT按16定义。

若以后换成树/多累加器归约，数值顺序改变，需要版本化参考和重新评估误差，不能只修改性能参数。

## 6. 原地操作与数值错误

合法原地任务在写当前组前，必须已读完该组所有会被覆盖的源；严格同跨度、同元素宽度，故不会破坏后续组。错位别名/宽度改变拒绝，不能靠执行顺序侥幸通过。

可终止的数值错误停止后续组，排空本地算术与 memory response，返回 NUMERIC，completed_units 为已写确认元素。目的 generation 标为失败，后继依赖失败；允许新任务在显式新写后重建数据。存储失去事务边界等不可终止错误则 fatal，不能释放隔离锁。

## 7. Transformer 组合说明

- RMSNorm：MUL(x,x)→SUM→MUL_SCALAR(1/N)→ADD_SCALAR(eps)→SFU.RSQRT→MUL(src1标量广播)→MUL(gamma)。动态标量在scratchpad只保存一个元素，不需要展开成完整向量。
- Softmax：masked MAX→SELECT(mask,score,max标量广播)→减max→乘log2(e)→EXP2→SELECT(mask,exp,0标量广播)→SUM→RCP→MUL归一化。被mask位置在指数之前替换为max，指数之后设0，保证不进入分母；不能仅在归一化后清零。全mask行按上层约定显式输出0，不能继续RCP(0)。
- RoPE：MUL(x,cos成对广播)、ROTATE_PAIR(x)、MUL(rotated_x,sin成对广播)、ADD。相邻维配对和split-half之间的TRANSPOSE2D见16；先对齐模型定义再选布局。
- SiLU：MUL_SCALAR(-log2(e))→EXP2→ADD_SCALAR(1)→RCP→MUL原x。若真实输入导致指数溢出则按12返回NUMERIC；本版不宣称该分解已对极端输入数值稳定，扩大范围需要另外设计分支／近似并验收。

这些是任务组合，不是本模块隐藏执行整算子微程序。完整调度例子见 [15](15_validation_and_sequences.md)。

## 8. 验收

各subop，count=1/V_LANES-1/V_LANES/V_LANES+1，混合格式转换，源／目的跨memory字，比较mask，标量／成对广播和范围检查，ROTATE_PAIR奇数长度拒绝，原地允许／拒绝，SUM顺序、MAX首值和全负值、masked归约及全mask行，数值错误、输出背压、最后写确认和完成数量。LAYOUT单独按16验收。

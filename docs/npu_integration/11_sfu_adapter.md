# 11 SFU 接入与非线性任务控制

状态：外围 v0.1 合同；保留用户已有 SFU 算术设计。当前目录 sfu_top 接口已核对，最终数值核与下述 capability 对齐后才能启用对应任务。

## 1. 外层端口（core 域）

| 信号 | 方向 | 位宽/类型 | 说明 |
|---|---|---|---|
| clk_core/rst_core_n | I | 1/1 | 时钟 / 复位 |
| task_valid/ready/cmd | I/O/I | 1/1/512 | SFU 指令 |
| event_valid/ready/data | O/I/O | 1/1/128 | 所有输出平面写确认 |
| mem_req_valid/ready/req | O/I/O | mem_req_t | SFU 客户端 |
| mem_rsp_valid/ready/rsp | I/O/I | mem_rsp_t | 按序单在途 |
| capability_ops | I | 9 | 每个逻辑 subop 一位的验证能力位（bit0=EXP2 … bit8=TANH） |
| capability_formats | I | 3 | FP16/BF16/FP32 的已验收位 |
| core_in_valid / core_in_ready | O/I | 1/1 | 接真实 SFU |
| core_op / fmt_in / fmt_out / mode | O | 4/2/2/2 | 映射后的控制 |
| core_din | O | 256 | 当前快照 8×32 bit lane 槽 |
| core_out_valid / core_out_ready | I/O | 1/1 | 结果握手 |
| core_dout / core_status | I | 256/8 | 结果 / 异常 |
| core_flush | O | 1 | 只用于协调恢复，不用于正常逐任务清零 |
| fatal | O | 1 | 帧/数量无法可靠恢复 |

capability 是构建期验证结果，不是用户可随意置1的运行配置；当前占位 slice 不能作为 EXP2 等已验收实现，因此默认功能能力位为0。最终核接入后按真实测试启用，不改写用户计算数据通路。

## 2. 现有 sfu_top 接口与编码映射

实际端口：clk/rst_n/flush、in_valid/in_ready、op[3:0]、fmt_in/out[1:0]、mode[1:0]、din[WIDTH*DATA_W-1:0]、out_valid/out_ready、dout、status[7:0]。当前默认 WIDTH=8、DATA_W=32，内部固定两个4元素 slice。

当前 sfu/arch/spec.md 的原语 op 为 EXP2=0、LOG2=1、RCP=2、RSQRT=3、SQRT=4、SIN=5、COS=6、TANH=7、SINCOS=8。新指令 SFU.subop 与它们的对应关系：

| subop | 功能 | core op |
|---|---|---|
| 0 | EXP2 | 0 |
| 1 | RCP | 2 |
| 2 | RSQRT | 3 |
| 3 | SINCOS | 8 |
| 4 | LOG2 | 1 |
| 5 | SQRT | 4 |
| 6 | SIN | 5 |
| 7 | COS | 6 |
| 8 | TANH | 7 |

适配表必须显式映射到该 op；不能把两层 opcode 原样连接。0-3 沿用早期编码，新增功能不复用它们。subop>8 与 P4[1:0]>1 均按 BAD_FIELD 拒绝。

fmt、mode、status 的最终位含义以最终核合同为准。建议建立只读 `sfu_core_profile`：每逻辑格式对应的 core fmt、fast/accurate mode、异常位掩码、SINCOS 每输入的输出帧数。未知项禁止启用，而非假定01一定BF16或某statusbit一定div0。

## 3. 数值与 lane 打包

首版优先向最终 SFU 提供 FP32 lane 输入并接收 FP32 输出，由公共格式转换器处理存储16bit格式；若最终核只接受16bit，则 profile 明确改用对应格式和每lane低16bit打包，不能把两种打包方式混用。

每批有效元素 L=min(8,remaining)。当前核无 lane_mask 口，无效 lane 按函数填安全有限值：EXP2/SINCOS/SIN/COS/TANH 填0，RCP/RSQRT/LOG2/SQRT 填1（log2(1)=0、sqrt(1)=1，不触发 invalid）。尾部无效 lane 不写回。若 status 是所有 lane 聚合，安全填充值避免尾部触发假 div0；不能填全0统一处理所有函数。

输入 din、op、fmt、mode 从任务接受时锁存（mode 取自 P4[1:0]），保守保持到该批所有输出写确认。当前目录 slice 有依赖实时 din 的占位路径，此保持有助于接口仿真，但不能修复未完成的数值或内部握手合并问题。

## 4. adapter 状态机

| 状态 | 行为 |
|---|---|
| IDLE | 接收任务并清批次/异常计数 |
| CHECK | 检查 capability、范围、格式、SINCOS 两平面不重叠 |
| LOAD | memory gather 一批元素，保留输入和原index |
| ISSUE | core_in_valid 保持至握手，只允许一个批次在途 |
| WAIT_OUTPUT | 预约输出 holding register 后拉 core_out_ready |
| STORE | 转换并写当前输出平面，等所有写确认 |
| NEXT_OUTPUT | SINCOS 若还有第二帧，返回 WAIT_OUTPUT |
| NEXT_BATCH | 本批全部输出确认后推进 index |
| COMPLETE | 保持事件；completed_units 为各平面确认元素总和 |
| ERROR_DRAIN | 不再发新批，消费当前可枚举的剩余输出，报告失败 |
| FATAL | 出现不可确定帧数、内部协议错误；协调 reset |

out_valid 没到不能靠预计延迟计数直接 STORE；core_out_valid 背压时核心必须保持 dout/status。若核心不能保持，必须在边界设置无条件可捕获的结果槽并限制在途数，不能直接声称兼容 ready/valid。

## 5. SINCOS 两种最终接入方式

本外围建议采用同输入先sin后cos的两帧模式，计数 `output_phase=0/1`。每批只接一次输入，收到两帧并分别写入 `D_sin+index*E` / `D_cos+index*E` 后才推进。

若最终核为双端口同时输出，则改由 boundary wrapper 将两平面先捕获到两个holding槽，再串行写回；若只有独立 SIN/COS，则 wrapper 对保持的相同输入发两次实际调用。三种模式由构建期 profile 选择，功能验收前不能混用或猜测。

当前顶层没有 phase/tag，所以采用两帧时必须保证严格顺序、一次仅一个批次。若未来允许多个输入在途，必须补外部顺序队列或明确tag，不能根据输出数简单交替猜job。

## 6. 数值错误、flush 与完成

除经过 profile 认可的 underflow→0 外，invalid/div0/overflow 导致任务 NUMERIC；停止新批并排空已知剩余输出，目的 generation 标失败。若无法确认剩余帧数则 fatal。RCP/RSQRT 对真实有效0/负值的处理必须测试，不能仅靠尾部填充掩盖。

正常任务之间不 flush。用户取消/异常恢复须先协调存储和全部引擎；不能只拉 flush 就宣称 MPU 和 DMA 一起取消。当前 SFU flush 与复位合并，物理复位释放仍需核对。

成功事件：普通函数 count 个输出、SINCOS 2*count 个输出均确认写入；2*count 不得溢出 completed_units。SINCOS 若两平面任一失败则整个任务失败。

## 7. 验收

接受输入后改变上游配置、核心长延迟、输出背压、8lane尾部、RCP/RSQRT无效lane不触发假异常、原语opcode映射、SINCOS两帧/双口profile、各平面写确认、运行中意外输出、占位实现能力位关闭、最终核按各格式的独立数值参考验收。

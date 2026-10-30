# 12 类型转换、浮点基础单元与数值合同

状态：新外围 v0.1 提案；现有 MPU 和最终 SFU 各有自己的数值边界，不因本文件而宣称具备完整 IEEE-754 功能。

## 1. 数据表示

FP16/BF16/FP32 都按其原始浮点 bit 表示存储，低地址保存低字节。INT4 仅用于 MPU 权重，由离线 pack 定义4bit二补码顺序。E8M0 scale 作为原始8bit位型传送，不用有符号整数转换器解释。

新 VPU 统一以 FP32 为 lane 内部格式；存储读取16bit时扩展，只有写目的格式时缩窄。不能在每个调度步骤自动把临时量变16bit，RMSNorm/softmax 的中间归约和归一化系数保留 FP32。

## 2. format_converter 端口

每实例一个元素在途，多个引擎可实例化各自转换器，避免把未规划的全局共享端口当作无限资源。

| 信号 | 方向 | 位宽 | 含义 |
|---|---|---|---|
| clk/rst_n | I | 1/1 | core 域 |
| in_valid/in_ready | I/O | 1/1 | 接受一个元素 |
| in_bits | I | 32 | 16bit源在低16bit，其余0 |
| src_fmt/dst_fmt | I | 2/2 | 00FP16、01BF16、10FP32；11拒绝 |
| in_tag | I | 8 | 当前元素/位置关联 |
| out_valid/out_ready | O/I | 1/1 | 输出握手 |
| out_bits | O | 32 | 16bit目的放低16bit，其余0 |
| out_tag | O | 8 | 原 tag |
| fp_flags | O | 8 | [0]invalid，[1]overflow，[2]underflow，[3]inexact，其余0 |

输入全部锁存后进入 `CLASSIFY→CONVERT→HOLD`，HOLD 直到接受。旁路 FP32→FP32 也经过同一握手，不能组合破坏背压稳定性。

## 3. 首版外围数值选择

- 正常有限数与±0为基本工作范围，舍入为 RNE（round nearest ties to even）。FP16/BF16→FP32 的正常数扩展不先做近似。
- 外围输入/输出 subnormal 统一 FTZ 为保留符号的0；记 underflow/inexact。此选择为本外围建议固定profile，不覆盖最终 SFU 可配置 FTZ。
- 输入 NaN/Inf 设置 invalid；缩窄溢出或基础运算溢出设置 overflow。引擎将这类结果视为 NUMERIC，不把非有限结果静默作为合法下一任务输入。
- RNE 保留 guard/round/sticky，处理尾数进位引起的指数变化；不能用简单截高16bit替代 BF16 RNE。
- Underflow/普通 inexact 为允许近似，可记录 fp_status 而仍返回 OK；invalid/overflow 导致任务失败。除非未来显式增加profile，不能静默改用饱和最大值。
- ±0 的符号在转换和乘法中按相应运算约定保留；MAX 相等时保留先到元素，SUM起点为+0，参考顺序由10定义。

## 4. FP32 基础运算模块

`fp32_add/sub/mul/compare` 采用 ready/valid 请求和响应，payload 包含 operand、操作枚举、lane有效位、tag，返回结果与 flags。延迟是实现参数，不预先填成“1拍”，接收方依据结果 valid。

| 信号 | 方向 | 位宽 |
|---|---|---|
| clk/rst_n | I | 1/1，core域 |
| in_valid/in_ready | I/O | 1/1 |
| a/b | I | 32/32 |
| op | I | 3，ADD=0/SUB=1/MUL=2/MAX=3/GT=4；5..7拒绝 |
| tag | I | 8 |
| out_valid/out_ready | O/I | 1/1 |
| result / predicate | O | 32/1，GT用predicate |
| out_tag / flags | O | 8/8 |

首版可以不共享不同lane运算器，也可以时间复用一个单元；资源选择由综合评估，必须与 VPU FSM/ready 对应。不将整数 `+/*` 直接对32bit位型操作当成浮点实现。现有 sfu/fp32_add.v 是否可复用需单独数值验收，不因文件名就当作可直接集成。

## 5. MPU 与 SFU 的独立边界

MPU 按现有PE/reduce/acc路径验收，存在 normal/zero 范围和 W4 BF16 宽指数跨度近似，不能用严格全IEEE FP32 FMA 逐位比较后强行宣称其实现错误或等价。
输入数据合法性由离线权重转换和 activation生成规则保证；若运行时 packer加分类检查，要在 core start前完成或将中途失败纳入 fatal，不能少送beat后仍等正常done。

SFU 的函数误差、特殊值、mode和FTZ以最终核profile为准，外围统一格式转换不替代EXP2/RCP/RSQRT本体。没有数值核时可用行为参考验证握手，但报告必须区分参考模型与可综合实现。

## 6. 数值验证矩阵

1. FP16/BF16 全65536输入位型分类与扩展，对 NaN/Inf/FTZ按profile核对flags。
2. FP32缩窄的正负0、指数边界、恰好半ULP、半ULP上下、尾数进位、最大有限数、溢出和最小normal。
3. 浮点基础算术的异号抵消、相等值比较、乘法符号、极端指数、随机正常输入。
4. SUM用固定顺序参考；不得仅用高精度实数结果掩盖实际舍入顺序。
5. 模型级采用绝对/相对误差和必要的余弦指标，阈值在具体模型/profile选定后定义，不发明统一“模型无损”阈值。

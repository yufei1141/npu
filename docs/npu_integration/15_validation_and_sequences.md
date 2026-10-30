# 15 验收计划、指令实例与 Transformer 调度

状态：设计验收规范和可复现的期望值，不是已经运行的RTL／模型测试报告。当前新增外围尚未实现，SFU数值能力仍依赖最终核接入。

## 1. 验证层级与完成标准

| 层级 | 测试对象 | 通过标准／观测 |
|---|---|---|
| 协议 | 所有ready/valid、固定延迟RAM、mailbox／FIFO | 背压稳定；每fire只前进一次；不丢不重 |
| 位序 | A/B打包、W4 scale、结果、TRANSPOSE/PACK | 与独立索引模型逐元素／逐beat一致；padding不越界 |
| 单任务 | DMA、MPU adapter、VPU、SFU wrapper | 接收／读／写／确认／完成各自计数一致 |
| 调度 | 事件、锁、拒绝、FENCE、释放 | 无提前读、重复alloc、错误释放或队头协议死锁 |
| 算子 | RMSNorm、masked softmax、RoPE、SiLU | 使用指定数值profile／相同归约顺序；阈值按实现与模型冻结 |
| 整层 | 小维度prefill、decode、GQA KV | 与独立逐层参考比对中间张量、KV追加、最终输出 |
| 物理实现 | 综合、STA、CDC／reset审查 | 另行提供指定FPGA／ASIC报告；行为仿真不替代 |

参考模型与DUT不得共用同一个错误位序函数作为唯一判据；至少增加可人工解释的标签矩阵和已知答案。数值参考分为现有MPU路径参考、外围RNE/FTZ参考、最终SFU函数参考三部分。

## 2. 最小端到端 MPU 指令实例

构建参数使用01默认值；启用ACT0/ACT1，各至少32B且物理范围不重叠。两个外部W16帧分别在0x1000和0x1080，A在0x3000，输出在0x4000。外存地址是仿真空间例子，非板级地址。

矩阵采用FP16整数：A=[1,2,3,4,5]，B[k,n]=k+n+1，k,n从0开始各5个。期望D=[55,70,85,100,115]，FP32写回。m=1，g=1，tiles=2，S=ceil(5/4)=2，第一tile有效4列、第二tile有效1列。每帧64B，缺K／N元素由打包补+0。

| 顺序 | 命令／job／slot | 字段／依赖 | 目的 |
|---:|---|---|---|
| 1 | LOAD 0x10 / 0 | dst=ACT0，ext=0x3000，P2=0，P3=10，P4=1，P5=P6=0 | A可见 |
| 2 | LOAD_W 0x11 / 1 | ext=0x1000，P2=0x00000013，P3=0x00020101，P4=64 | consumer=0x13的tile0 |
| 3 | FENCE 0x12 / 2 | P全部0 | 汇合A和第一权重；失败也传播 |
| 4 | MATMUL 0x13 / 3 | DEP=(2,0x12)，src0=ACT0，dst=ACT1 | 开始后可继续发命令5 |
| 5 | LOAD_W 0x14 / 4 | ext=0x1080，P2=0x00010013，P3=0x00020101，P4=64 | tile1与计算重叠 |
| 6 | FENCE 0x15 / 5 | P全部0 | 等MATMUL写确认及DMA完成 |
| 7 | STORE 0x16 / 6 | DEP=(5,0x15)，src0=ACT1，ext=0x4000，P2=0，P3=20，P4=1 | 外存写确认后完成 |

MATMUL详细payload：P0=0x00010101，P1=0x00020002，P2=5，P3=5，P4=0，P5=32，P6=0，P7=0，P8=32，P9..P11=0。formats=0xA0（A/W FP16、D/acc FP32），flags.DEP_EN=1。LOAD/STORE格式=0，全部指令version=1，未使用字段0。

主机从开始持续消费CPL；完成顺序可能LOAD_W tile1早于MATMUL，以job匹配而非按发射位置匹配。七条命令各一条记录，MPU completed_units=2，STORE=20字节；完成后检查外存五个FP32值及相邻哨兵未变。清理时逐slot发EVENT_RELEASE（独立新job，slot=FF），继续消费其行政完成；槽清理不删除已有CPL。

### 关键波形断言

1. FENCE 0x12成功前不可start；第一权重失败时MATMUL因DEP_FAILED不start。
2. core busy下降后，配置仍保持到整个job写确认终结。
3. tile0与tile1的A各2个beat、B各2个beat，总数均4；每acc_valid有预留槽。
4. LOAD_W完成时bank仍不可覆盖；最后B握手及release完整结束才FREE。
5. 外存写last已接受但wr_rsp未回时STORE不能完成。

## 3. 逐模块测试矩阵

| 模块／文档 | 正常与边界 | 压力／故障 |
|---|---|---|
| 03前端 | FIFO空／满、同拍进出、所有opcode、保留位 | held与旧队列ID重名、payload接收后改变 |
| 04调度 | 一个计算+一个DMA、读共享、写独占、原地去重 | slot冲突拒绝、缺依赖、失败依赖、barrier_error跨已释放槽保持 |
| 05存储 | 五客户端、不同bank同拍、strobes、最后字 | bank争用公平性、rsp背压、ALIGN/RANGE、未知tag |
| 06DMA | 长度1/15/16/17/31/32/33、2D stride、三类传输 | keep/last/tag错、写确认延迟、中途错误、FIFO满空 |
| 07权重 | g=1/2/8，S=1/3/4/5/上限、W4/W16 | 两bank乱到达但按tile顺序消费、commit与release交错、重复身份 |
| 08MPU | 四mode、m=1/4、g=1/8、T=1/上限、K/N尾部 | A/B独立停顿、busy早降、缺数据、额外beat禁止 |
| 09写回 | FP16/BF16/FP32、D非字对齐、多row/tile | acc突发、无ready捕获、最后写ack延迟、转换溢出 |
| 10VPU | 全subop、标量／pair广播、SUM顺序、masked MAX | 全mask、无效lane、合法／非法原地、数值中途失败 |
| 11SFU | 原语映射、8lane尾部、两平面SINCOS | 未验收capability拒绝、输出背压、状态异常、帧丢失 |
| 12数值 | 分类、RNE ties、FTZ、符号0、指数进位 | NaN/Inf、overflow、极端指数、随机正常数 |
| 13事件 | 五源竞争、CPL同拍入出、IRQ、显式释放 | CPL满、重复完成、无槽拒绝、fatal先于错误提交 |
| 14顶层 | boot/run/drain、配置、两域ready | 域停钟／复位、旧外存响应隔离、CPL长背压 |
| 16布局/KV | 矩形转置、K/N尾部、QK与PV帧回读 | 越界、GQA映射、KV追加未确认、PACK与LOAD_W长度不符 |

建议随机回归记录seed、所有构建参数、指令流、时钟计划、错误注入位置和首个不匹配张量；“跑过一次随机”不构成所有CDC相位覆盖。

## 4. RMSNorm、Softmax、SiLU、RoPE 序列

以下每个箭头都是显式指令和可见完成依赖，不是隐藏FSM。每个临时量应分配不同源／目的buffer或经过合法原地规则，必要时STORE spill；只有8个逻辑buffer，不能假定无限临时buffer。标量可放TMP buffer不同偏移，锁仍按整个buffer保守处理。

### RMSNorm 一行

| 步骤 | 指令 | 输出 |
|---:|---|---|
| 1 | VECTOR.MUL x,x，dst FP32 | x² |
| 2 | REDUCE.SUM | FP32 s |
| 3 | VECTOR.MUL_SCALAR(s,1/N)，count=1 | mean square |
| 4 | VECTOR.ADD_SCALAR(eps)，count=1 | v |
| 5 | SFU.RSQRT(v)，count=1 | r |
| 6 | VECTOR.MUL subop=0x82，src0=x，src1=r | FP32 x*r |
| 7 | VECTOR.MUL gamma，显式dst格式16bit | 下一MPU激活 |

eps、1/N使用经过软件FP32编码的立即数；gamma和x都需有有效生产者。多行分别归约，不把多行flatten成一个SUM。SFU能力未启用时该序列应UNSUPPORTED，不能用透传占位核宣称通过。

### 有效mask的softmax一行

mask由软件按query/key绝对位置生成byte 0/1并LOAD，或后续另行增加mask生成硬件；本版没有隐式causal-mask指令。

```text
s = score * attention_scale
m = REDUCE.MAX(mask_enable=1, s)
z = SELECT(mask, s, scalar_broadcast(m))
u = (z - scalar_broadcast(m)) * log2(e)
e = SFU.EXP2(u)
e_mask = SELECT(mask, e, scalar_broadcast(+0))
l = REDUCE.SUM(e_mask)
r = SFU.RCP(l)
p_fp32 = e_mask * scalar_broadcast(r)
p_16 = CONVERT(p_fp32)        # 仅在进入现有MPU A口前显式转换
```

SELECT的标量false候选为subop=0x84，masked MAX为0x81，动态SUB为0x81（opcode不同，不能混淆），动态MUL为0x82。mask全0时MAX返回NUMERIC；上层若需要全0行，提前识别并显式写零，不能把失败MAX当正常结果继续。分母只包含有效key；单有效key应得到概率1（按实际profile舍入）。

### SiLU 与 gated MLP

SiLU(x)=x/(1+exp(-x))按10的EXP2/RCP序列实现；与up分支逐元素MUL后进入down投影。首版仅验收有限安全输入区间，极负数导致指数溢出必须失败，不能将Inf绕过有限数合同。更广范围可另做稳定sigmoid实现后启用，不在文档中伪称已经完成。

### RoPE

每head按模型的配对规则取得sin/cos；interleaved形式计算x*cos_pair+ROTATE_PAIR(x)*sin_pair。MUL subop=0x42表示pair广播。split-half先转置成相邻对，结束再逆转置。频率参数、位置偏移和实际系数来源由模型确定；若SFU.SINCOS未验收，用软件预置表验证其它路径，不把查表替代当作SFU数值测试。

## 5. Transformer 层级调度

prefill：每M tile≤4行，Norm逐行→Q/K/V投影→RoPE→K/V按token STORE追加→FENCE确认→按head读取历史KV→16的QK/PV重排路径→每query行softmax→O投影和residual→Norm→gate/up/SiLU/down→residual。

decode：M=1；n_grp可选1..8，按真实供数／容量预算调节；先把当前K/V追加成功，再读取覆盖当前位置的历史。causal有效key为≤当前query位置；GQA按16映射。部分列组只是当前输入置零策略，不能据此报告已实现时钟门控节能。

建议先用人工小配置 Hq=2、Hkv=1、head_dim=8、prefill=3个token、再decode 1个token，逐张量记录参考结果；这只是功能fixture，不是选定实际模型。scratchpad、权重bank、外存和临时B帧容量分别按该fixture校验。更长上下文和实际0.5B–3B权重必须在模型确定后另做容量／数值／性能评估。

完整层测试必须记录B帧STORE→LOAD_W的额外流量，不能用只有模型权重的带宽估算代替基线实际流量。online attention未实现时，不宣称任意长上下文可执行。

## 6. CDC、reset 与错误注入

- 以io:core约0.37、0.73、1、1.41、2.7的频比和不同初始相位作行为用例，叠加长暂停；这些是仿真激励，不是推荐工作频率。CDC物理正确性仍需专门检查。
- 在commit请求高／ack高／req回低／release各阶段断言协调reset，确认新任务不继承旧descriptor或半帧。
- 外存平台在reset后返回旧tag时先由平台隔离；不要指望重置tag为0自动区别旧响应。
- fatal错误发生同拍完成提交时，断言scheduler先进入停发／锁隔离，不能被retire误清锁。
- 若当前事务不能恢复可靠边界，则只要求sticky fatal和安全停止；不能捏造“每条旧任务都有完成”。正常无reset运行才要求恰好一条。

## 7. 当前文档检查与后续报告

本轮检查只覆盖Markdown链接／章节合同、编码边界、权重长度与PACK索引的静态算术；不会产生RTL仿真、综合、时序或模型精度的通过结论。后续报告至少带日期、工具版本、源代码版本、profile、已跑用例、失败项和待确认SFU合同，未跑项目明确标注未验证。

### 2026-10-29 实际静态检查记录

工具：Python 3.11.9标准库；对象为本目录v0.1文档，尚无对应外围RTL版本。以下结果是文档结构与公式检查，不是对DUT的测试。

| 检查项 | 实际结果 |
|---|---|
| 19份Markdown、53个相对文件链接、代码围栏配对 | 通过；文件链接目标均存在 |
| 指令位域覆盖／重叠、payload和opcode | 512bit无缺口／无重叠；12个32bit payload；9个不同opcode |
| 头部与payload组装后拆解 | 32组位型通过；不代表已实现指令解码器 |
| PACK_W16地址、元素回读与padding | 30组K/N组合通过；K=1/3/4/5/17，N=1/3/4/5/31/32 |
| W4帧长度／scale区覆盖及nibble顺序算术 | 15组g/S组合通过；g=1/2/8，S=1/3/4/5/256 |
| 第2节指令参数和矩阵期望值 | m=1、g=1、mode=1、T=S=2；D=[55,70,85,100,115] |
| 小矩阵QK转置等式和P·V软件索引 | 通过；只检查布局关系，未验收硬件浮点误差 |
| 默认逻辑容量与带宽算术 | 权重144KiB、结果1KiB；128bit×250MHz=4GB/s |
| RTL仿真／综合／STA／物理CDC／模型精度 | 未运行，仍需后续实现与验证 |

交叉审阅同时补齐：RUN关闭接收后DRAIN继续发射旧队列、scheduler与memory共用buffer配置、mailbox本地提交与完整往返完成分离，以及可恢复失败释放锁与fatal隔离的区别。

### 2026-10-30 RTL 仿真记录：真实 Qwen3-0.6B AWQ 权重

上文“外围尚未实现”只适用 v0.1 文档版本；本轮已有对应 RTL 并在两套仿真器上运行。

工具：iverilog（msys2 mingw64）、Verilator 4.218、Python 3.10。模型为
`downloads/qwen3-0.6b-awq`（hidden 1024、28 层、16 head / 8 KV head、head_dim 128，
tied embedding）。参考值是 `scripts/qwen3_ref.py` 的真实 fp32 前向输出，不是硬件行为模型；
硬件只负责一次 GEMV，因此比对给出的是 int4 + E8M0 再量化误差，不能当作位精确结论。

| 用例 | 结果 |
|---|---|
| `scripts/regress.sh`（iverilog，17 个 testbench） | 17/17 通过 |
| `t_ones`：K=1024 全 1 权重与激活 | 1024.0，与理想值一致 |
| `scripts/real_gemv.py` 七个投影，layer 0，32 列 | 见下表 |
| `q` 投影层间扫描 L0 / L13 / L27 | 9.2% / 8.6% / 11.4% RMS，cos 0.998 / 0.996 / 0.994 |
| `q` 投影 L27 token 0 | 40.4% RMS，cos 0.919（该 token 激活离群，量化误差放大） |

| 投影 | RMS 误差（占参考 RMS） | cos |
|---|---:|---:|
| q | 9.2% | 0.998 |
| k | 7.0% | 0.9985 |
| v | 30.9% | 0.953 |
| o | 5.5% | 0.9986 |
| gate | 6.5% | 0.9979 |
| up | 10.0% | 0.995 |
| down | 6.2% | 0.998 |

所有用例中 RTL 输出与“用打包后权重做精确 fp32 矩阵乘”的结果打印位一致，
即残差全部来自再量化，不来自 RTL。`v` 与 L27 token 0 的误差偏大属于激活分布问题，
需要按量化预算单独评估，不是本轮的 RTL 缺陷。

本轮修复的两个问题：

1. LOAD_W 的总线地址取成 `{P1,P0}`，而该指令的 P0/P1 是两个独立地址（权重区、scale 区），
   正确值只有 P0。修复后 Verilator 发出的请求为 `addr=0x4000 bytes=2304`。
2. `scripts/hw_pack.py` 把 16 bit 权重字按大端写上线，而 PE 按 nibble `3-j` 取第 j 列，
   必须小端（见[权重路径](weight_path.md)的组内位序）。错序等价于列旋转两位，
   全 1 权重测试看不到，只有真实权重会暴露：修复前 `cos=0.045`，修复后 `cos=0.998`。

Verilator 限制：`tensorcore/rtl/fp32_acc.v` 在 `for` 循环里对数组做延迟赋值
（BLKLOOPINIT，Verilator 4.x 不支持）。抑制该告警可以编译，但赋值被静默丢弃，
累加器停在复位值——端到端表现为“PE 输出 4.0、reduce 输出 16.0、acc 恒为 0”。
`tensorcore/` 未修改；`scripts/verilate.sh` 用 `tb/vsim/fp32_acc_sim.v`
替代该文件（`fadd()` 原样复制，只把逐 lane 更新改成每 lane 一个 always 块）。
iverilog 仍编译原始文件并作为参考仿真器。若将来改写 `fp32_acc.v`，删除该替代与其映射。

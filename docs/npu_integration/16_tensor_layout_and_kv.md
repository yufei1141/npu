# 16 张量重排、Attention 与 KV cache 路径

状态：v0.1 外围提案；新增 LAYOUT 指令及 tensor_layout 子模块，尚未实现。它补齐普通张量与现有 MPU B 口之间的布局转换，不改变已有乘加数据通路。

## 1. 必须显式解决的路径

Q/K/V 投影结果在 core scratchpad，当前权重 bank 只有 io 写入口。QKᵀ 和 P·V 的 B 都是动态张量，不能把普通 KV buffer 的地址直接作为 MPU 权重地址。

首版功能基线采用以下闭合路径：

```text
外存 K/V → LOAD → 普通 scratchpad
        → [TRANSPOSE2D] → PACK_W16 → 普通 scratchpad 的 B 帧
        → STORE → 外存临时 B 帧 → LOAD_W → io 写双时钟权重 bank
        → weight_reader → MPU B 口
```

这会额外读写外存，不能把它的性能当成纯权重流式 GEMV。未来可增加 core-local 权重 bank 写入口／双源 B adapter，但要另定义 bank 所有权、仲裁和提交；本版没有这个旁路。模型离线 W4 权重继续由离线打包工具形成原生帧，不经过本模块在线量化。

## 2. tensor_layout 模块端口和资源

本模块放在 vector_engine 内，与 VECTOR/REDUCE 共用一个计算任务和 memory 客户端1。一次任务仅选 vector datapath 或 layout datapath，不能两者同时驱动 memory。

| 信号 | 方向 | 位宽/类型 | 含义 |
|---|---|---|---|
| clk_core/rst_core_n | I | 1/1 | core 时钟／复位 |
| task_valid/ready/cmd | I/O/I | 1/1/512 | LAYOUT 指令 |
| event_valid/ready/data | O/I/O | 1/1/128 | 复用 VPU 完成源 |
| mem_req_valid/ready/req | O/I/O | mem_req_t | 已获 VPU 本地访问权 |
| mem_rsp_valid/ready/rsp | I/O/I | mem_rsp_t | 每客户端一在途 |
| active/element_index | O | 1/32 | 诊断，不作为完成 |
| fatal | O | 1 | tag／事务边界无法恢复 |

顶层 scheduler 将 opcode=0x23 路由 VPU。vector_engine 锁存执行选择，直到完成事件被接受才释放；VPU 内的 req/response/event mux 依锁存选择路由，不能按当前上游 opcode 组合选择。

## 3. 指令字段

opcode=0x23；subop=0 TRANSPOSE2D，1 PACK_W16。INPLACE=0，src1_buf=0，accumulator 格式位为0。源／目的必须是不同 buffer，禁止隐藏重叠。

### TRANSPOSE2D

| word | 含义 |
|---|---|
| P0/P1 | rows/cols，均>0 |
| P2/P3 | src_base/dst_base，字节 |
| P4/P5 | src_row_stride/dst_row_stride，字节 |
| P6..P11 | 0 |

src0/dst 格式相同，只支持 FP16/BF16/FP32，对原始2或4字节元素置换，不做算术。源范围为 rows×cols，目的形状 cols×rows；P4≥cols*E，P5≥rows*E，基址和stride按E对齐。completed_units=rows*cols，扩位检查不超过32bit。

### PACK_W16

| word | 含义 |
|---|---|
| P0/P1 | K_valid/N_valid，均>0 |
| P2/P3 | src_base/dst_base，字节 |
| P4 | src_row_stride，≥2*N_valid |
| P5 | n_grp，1..8，其余高位0 |
| P6..P11 | 0 |

源为行主序 B[K_valid,N_valid]，src0/dst 格式相同且只允许 FP16/BF16。目的属于原始16bit B帧，formats.dst 表示其中元素格式，不意味着普通矩阵布局。S=ceil(K_valid/4)≤WBUF_K_STEPS，1≤N_valid≤4*g；本指令仅生成一个 N tile，多个 tile 用独立 LAYOUT 命令和源列偏移。

目的精确字节数为32*g*S，不另附scale。目的基址按16B对齐；completed_units 为实际源矩阵元素数 K_valid*N_valid，不把padding算有效元素。末端payload容量以及源地址最后一行最后一元素均在CHECK扩位验证。

## 4. 地址和位序

TRANSPOSE2D 对每个源(r,c)：

```text
src = src_base + r*src_row_stride + c*E
dst = dst_base + c*dst_row_stride + r*E
```

PACK_W16 按组j、K-step s、组内输出列p、K元素i遍历：

```text
j = 0..g-1; s = 0..S-1; p = 0..3; i = 0..3
k = 4*s+i; n = 4*j+p
src = src_base + k*src_row_stride + 2*n
dst = dst_base + 32*(j*S+s) + 8*p + 2*i
weight_group[64*p+16*i +:16] = B[k,n]
```

k≥K_valid 或 n≥N_valid 时直接生成+0，不发源请求。该排列对应 gemv_unit 的每PE一输出列／每PE四个K权重，不是把普通4×4行主序片段直接拷贝过去。每32B组拆两个128bit lo/hi，与[07](07_weight_buffer.md)相同；STORE/LOAD_W 的长度必须正好32*g*S。

## 5. 状态机与memory交互

| 状态 | 行为及转移 |
|---|---|
| IDLE | 接受并锁存task，清循环与成功计数 |
| CHECK | 验证格式、维度、全部源目的范围、能力；失败→COMPLETE(error) |
| INDEX | 生成当前源／目的元素地址；padding→BUILD，其他→READ_REQ |
| READ_REQ | 发一个对齐读字，保持到req_fire |
| READ_WAIT | 等rsp_fire，提取目标2/4字节到element_hold |
| BUILD | 准备目的对齐字的数据和字节strobes |
| WRITE_REQ | 发带strobes的单字写；不覆盖其余字节 |
| WRITE_ACK | 收到成功确认后累计有效元素，进入NEXT |
| NEXT | 按既定索引推进；结束→COMPLETE，否则INDEX |
| COMPLETE | 保持event至被接受；成功仅在全部padding和有效元素都写确认后 |
| ERROR_DRAIN | 停新请求，消费当前已接受响应；可终止错误→失败事件 |
| FATAL | 未知tag／额外响应等需要协调reset |

最小实现允许每元素一个读和一个带掩码写，即使同字也不自动合并，保证小缓存即可实现。后续增加转置tile缓存或字合并时仍须保持所有写ack再完成；不能据此保守FSM声称每拍一个元素。

## 6. KV 外存地址合同

首版由软件生成DMA地址，没有硬件KV页表、分配器或自动逐token追加。约定一个可替换的参考布局：K、V独立平面；每层每batch各自有基址，平面内部为[kv_head,token,head_dim]。

```text
addr_K(h,t,d) = K_base(layer,batch) + h*K_head_stride + t*K_token_stride + d*2
addr_V(h,t,d) = V_base(layer,batch) + h*V_head_stride + t*V_token_stride + d*2
kv_head(q_head) = floor(q_head / (Hq/Hkv))
```

head_dim、Hq、Hkv、最大token、层数和地址容量从最终模型读入。要求Hq%Hkv=0，且层/batch平面不相交；不满足的结构需另定义映射。K/V为16bit存储profile，不声称覆盖量化KV。每token物理stride对齐至少B，头stride覆盖所有token，必要padding不算逻辑head_dim。

DMA首版行基址要求外存16B／本地B对齐。因此token/头分片DMA要么从对齐基址搬入整行，再由layout选有效元素，要么软件生成有padding的行；不能任意把维度偏移2B后直接LOAD。PACK接受元素对齐列偏移不改变DMA对齐规则。

## 7. QKᵀ 与 P·V 的执行

### QKᵀ

1. 将连续token的 K[L,D] LOAD到KV_TILE0。
2. TRANSPOSE2D得到 B[D,L]，再按≤4*g列用PACK_W16生成一个或多个帧。
3. 每帧STORE到外存临时区，确认后LOAD_W，consumer指向对应score MATMUL。
4. Q 是A[m,D]；mode为对应W16FP16/BF16，K_valid=D，N_valid是当前key分块长度。MPU结果建议FP32。
5. 多N tile以正确D_base/stride写回逻辑score行，按query行分别执行缩放和mask。不同key分块不是K维累加任务。

### P·V

1. B为 V[L,D]，按D列tile PACK_W16。P是A[m,L]，现有MPU A必须16bit，因此归一化FP32概率需显式CONVERT；该精度损失进入模型数值验收。
2. K_valid=L，N_valid为本次输出维度tile。完整归约长度L必须≤4*WBUF_K_STEPS。
3. 若采用更长L，增大权重bank或将L分块成独立MATMUL，分别得到FP32部分O后用VPU.ADD显式累加；不能指望现有core跨job自动保持部分和。

Q与K等值范围、概率FTZ、W16乘法和累加误差均遵守[12](12_format_and_numeric.md)。不同heads／query位置分开编排；本版无隐式广播head或batch维。

## 8. prefill、decode和长上下文边界

prefill每次最多4个query；每行mask判定为key_position≤query_position并处于有效历史范围。decode为1个query，先确认当前K/V写入，再读合法历史；锁与外存STORE依赖确保不读取尚未写确认的token。

功能基线只对能驻留的score/probability行做显式softmax。0.5B–3B是目标权重规模，不保证任意上下文均能放入该仿真profile。长上下文需要分块online softmax：每行保持(m,l,o)，对每块更新m、对l/o重缩放、累加，再最终除以l；如采用此路径，必须另外验收masked max、指数缩放、P转16bit和部分O累加的数值误差。当前保留ATTN_STATE不等于online attention已实现。

RoPE可用[10](10_vector_engine.md)的ROTATE_PAIR及成对系数广播。原模型若为split-half表示，先把[2,D/2]转置为[D/2,2]，旋转后反向转置；不修改模型对维度配对的定义。sin/cos可由软件预置有限表或经过验收的SFU.SINCOS生成。

## 9. 验收

转置矩形／方阵／1元素、跨字元素、四种非零stride、周边哨兵保护；PACK以B[k,n]=可区分标签逐项回读，覆盖K=1/3/4/5和N=1/3/4/5/31/32；不足K/N补零、g=1/8、长度与LOAD_W完全一致；QKᵀ与P·V的独立小矩阵；多头GQA映射和首/末token；STORE未确认不得LOAD_W、KV未写确认不得读；容量不够明确拒绝。数值／RTL测试尚待实现。

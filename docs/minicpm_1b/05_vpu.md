# 05 VPU/SFU与模型专用前后处理

## 1. 资源与接口

VPU16独立FP32 MUL、ADD/compare、SUM/MAX reducer；SFU4请求lane，EXP/RECIP/RSQRT/SOFTPLUS；标量接口可用SFU低占用时隙但必须保留credit。RPU依赖beta/r，不能被长SwiGLU占满SFU队列后永远无法启动。

VpuCmd含op、src/dst TensorRef、length32、segmentLength16、gammaMode（WEIGHT/ONE_PLUS_WEIGHT）、ACT_FMT、roundingFence、scalarParams、TxnTag。普通任务单active，actor内部phase允许独立read/write pipeline；未来多actor扩展需显式资源scheduler。

vector beat16个FP32，mask/segmentLast/query/head/tag跟随；无readyFP IP返回预留FIFO，source read双源同bank组只能时分，不能凭不同base自动并行。

## 2. 数值与rounding fences

默认保持框架A16边界而不是全部融合成FP32：
- 零中心RMSNorm：FP32mean/rsqrt与(1+weight)，最后castA16。
- GDN gatedNorm：RPU o先castA16；norm内部FP32求mean/rsqrt，normalized castA16；gamma相乘按原dtypecastA16；再乘FP32 SiLU(z)，最终castA16。与普通Norm不可互换。
- FFN：SiLU(gate)按框架输出A16，再与up相乘castA16，送down。
- full attention：attention output A16，与sigmoid(gate)的A16结果相乘castA16，再o_proj。
- lm_head：内部FP32acc，默认cast为框架A16logits后argmax；FP32 argmax可作诊断模式，改变tie语义需声明。

GPU kernel内部FMA/规约次序不一定可逐bit复现；端到端验收需定义与原runtime的容差/任务指标。FP32全融合可以减少舍入，但不是自动更忠实。

## 3. L2Norm vs RMSNorm

Qwen线性attention L2Norm：q/k * rsqrt(sum(x²)+eps)，不除D；q随后乘1/sqrt128。RMSNorm用mean，因此差一个sqrtD，不能复用同descriptor却忘记设置模式。

普通zero-centered weight为(1+w)，gatedNorm为w。gamma/norm参数离线导出可预生成FP32gamma但必须保留所需A16fence；1+w使用FP32，不将其先低精度round。segment长度128/256/1024分别为linearhead/fullhead/hidden，每query/head独立。

reducer16→1平衡树4层；长segment多个blockpartial存储后固定树归约，不单标量每拍反馈。latency未知用partial context/scoreboard。gamma fetch与x fetch需要独立bank组或两阶段读。

## 4. 短卷积单元 DCU4（Depthwise Conv Unit，与全局DCU区分命名）

正式模块名ConvUnit4，避免DCU缩写冲突。6144channel，每channel4tap，权重A16，输出累加FP32后A16，再SiLU后A16。history保存最近3个raw in_proj_qkv输入，非SiLU输出。第t输出=sum_j w[j]*x[t-3+j]，在initialhistory0时左补零。

每layer coefficient48KiB、minimalhistory36KiB，18layerhistory648KiB放DDR或专门cache。原runtime可能保留4步cache，导出器映射为3历史＋current，不改变输出；snapshot包含history schema，不能直接拷贝runtime4stepcache却按3解析。

实现16channel/beat、tap4次复用MUL16，ADD partial，系数与history本地bank；pre/postfoldconv次序固定。source输出dtype与SiLUfence独立。所有channel完成后Q/K/V分拆，blockprefill连续更新history，chunk边界不清零。

## 5. beta/decay：必须支持softplus

β=sigmoid(b)，b是未量化16×H投影输出；g=-A*softplus(a+dt_bias)，A=exp(A_log)可host按runtime精度预计算并记录hash；r=exp(g)。a/b运行时变化，不能预计算整条r表代替网络。

softplus稳定形式max(x,0)+log1p(exp(-abs(x)))。为了288个head scalar/token，不为LOG1P造16lane大管线；首版SFU scalar SOFTPLUS采用独立分段近似，范围如[-16,16]为候选，尾部exp/x旁路必须有误差报告及实际输入分布证据。不因低工作量忽略精度，r影响长期state记忆。

sigmoid稳定负支a=exp(-abs(x)),inv=recip(1+a),negative=a*inv，正支inv；SiLU=x*sigmoid(x)。共享SFU每element可能两次请求，负支需要MUL。性能模型必须按原语数而非opcode个数计数。

## 6. partial RoPE、attention门控

完整headD256仅前64旋转，其余192pass-through。64维生成32频率，NeoX半区旋转并遵循MRoPE interleaved frequency layout。纯文本position ids通常三个轴相同，仍由原processor/model纯文本路径测试验证，不假设把多模态position全设0。

首版host生成每token32组sin/cos并cast至框架ACT_FMT，传256B级别系数（实际2×64? expanded cos/sin各64A16=256B），cache双buffer；FP32计算RoPE是否保留framework multiply/add A16fence由manifest定义。视觉入口将来需三轴positions，不在本v1实现。

full q_proj输出按head组织[Q256,gate256]，不是简单整个前2048Q、后2048gate；exporter可重排成canonical separateQ/gate，TensorDesc必须说明layout。

## 7. online softmax及argmax

fullattention6layer使用FP32 m/l/o，tile32key（每head32score）先max，exp(score*scale-mnew)、更新l与old o，A16概率送MPU PV。每head/query state顺序commit，next QK可lookahead但概率/alpha依赖上一版本。l可FP32 p，PV A16 p是候选近似；支持严格reference时应匹配framework边界。

finalnormalize o/l→A16，再gate。argmax流式处理每row headlogit，tail词id<V，平局最小id，NaN fault；greedy为裸金属基线。top-k/top-p可CPU选项需收集必要logits且额外流量/时延明确，不当已支持随机采样。

## 8. FSM与验收

各op FSM显式READ→REDUCE/SFU→SCALE→WRITE→ACK，不把流水首输入fire当完成。scalarSfu输出r/beta先ready，RPU才发；normgate等待RPU完整head输出。覆盖分段边界、eps近零、softplus大正负、FP16溢出/下溢、qGate reshape、partial非旋转维不变、layerconv history不串扰与跨block连续性。

# 01 模型事实、算子与公开设计依据

## 1. 模型事实

目标：https://huggingface.co/openbmb/MiniCPM-V-4.7-1B-GPTQ 。配置、量化配置、索引、revision快照和来源hash存于sources。模型主分支可能变化，以保存的revision与hash复现。

| 属性 | 已读取配置 |
|---|---|
| H / F / layers / vocab | 1024 / 3584 / 24 / 248094 |
| layer类型 | 18 linear_attention，6 full_attention，每4层一full |
| full attention | 8 Q heads、2 KV heads、D256，输出gate开启 |
| linear attention | 16 K/V heads，Dk128/Dv128，conv kernel4 |
| GPTQ | bits4、group128、sym true、desc_act false、checkpoint_format gptq |
| RoPE | rotary64（256×0.25），theta1e7，MRoPE interleaved [11,11,10] |
| Norm | eps1e-6；常规Norm零中心权重（1+w），GDN输出Norm为独立gated版本 |
| 视觉 | 不进入本设计，text-only运行不发视觉输入 |

hidden不能被假定等于Q总宽；这里H1024但Q总宽2048。full q_proj还包含gate，输出4096（按head分成Q256与gate256），不是2048。配置max_position_embeddings262144只是软件能力，板级容量和延迟单独约束，首版context4096。

### 1.1 语言矩阵尺寸（输出×输入）

| layer类型 | 矩阵 | shape | 精度/目的 |
|---|---|---:|---|
| linear ×18 | in_proj_qkv | 6144×1024 | W4：Q2048、K2048、V2048 |
| linear ×18 | in_proj_z | 2048×1024 | W4：gated norm输入 |
| linear ×18 | in_proj_a/b，各 | 16×1024 | 未列入GPTQ量化列表，原生A16 |
| linear ×18 | out_proj | 1024×2048 | W4 |
| full ×6 | q_proj(Q+gate) | 4096×1024 | W4 |
| full ×6 | k/v_proj，各 | 512×1024 | W4 |
| full ×6 | o_proj | 1024×2048 | W4 |
| 所有×24 | gate/up_proj，各 | 3584×1024 | W4 |
| 所有×24 | down_proj | 1024×3584 | W4 |
| final | embedding/lm_head矩阵 | 248094×1024 | 按native16预算，是否共享待loader核验 |

以上W4矩阵共497,025,024参数；payload=248,512,512B，若scale16bit/G128加7,766,016B；serialized GPTQ zero另按每group4bit加1,941,504B。canonical signed4可不保留zero，但仅在真实zero解码验证后。scale实际dtype、qzeros编码不能由sym=True推断。

native16矩阵508,096,512B；仅embedding查一行=2048B，但lm_head每token读取整矩阵。模型卡称1B不等于每decode token执行1B矩阵产品。完整文件索引total_size=1,865,984,480B，包含本设计不使用的视觉部分。

### 1.2 混合注意力的实际运算

参考Transformers Qwen3.5实现：linear先depthwise causal conv4+SiLU；q/k执行L2Norm（sum而不是mean）；q额外乘1/sqrt128；β=sigmoid(b)；g=-exp(A_log)*softplus(a+dt_bias)；r=exp(g)。FP32状态S[head,128,128]按token递归更新：

```
Sd = r * S
u[j] = sum_i k[i] * Sd[i,j]
delta[j] = beta * (v[j] - u[j])
Snew[i,j] = Sd[i,j] + k[i] * delta[j]
o[j] = sum_i q[i] * Snew[i,j]
```

最后per-head RMSNormGated(o,z)，再out_proj。GDN中的Norm乘weight，不使用普通Norm的1+weight；源实现还有输入dtype中间cast，必须记录rounding fence。

full attention先Q/K零中心RMSNorm、partial RoPE，再causal attention，输出乘sigmoid(gate)，最后o_proj。若忽略该gate，不能叫兼容此网络。

### 1.3 新增算子覆盖

原MPU/VPU已具备Linear、Norm、SiLU、softmax、RoPE的组合能力；本模型另外必须实现depthwise conv4、softplus、L2Norm、FP32 recurrent state read/update、RMSNormGated、sigmoid attention gate与partial RoPE。不实现GELU/视觉LayerNorm、ViT、image patch encoder。

## 2. 参考公开设计：可借鉴什么

| 来源 | 可核查的公开内容 | 本设计借鉴 | 不做的推断 |
|---|---|---|---|
| [Rockchip RKNN/RKLLM](https://github.com/airockchip/rknn-llm) | 模型转换/量化、runtime分层，RK3588支持；README给出Qwen3.5 w8a8及视觉/语言分离示例 | host编译图、离线pack、宏任务、共享DDR的有效带宽预算 | 没有公开证据证明内部PE数量、GDN原语或本模型全部在NPU上执行；不复制宣称6TOPS推算本W4A16速度 |
| [Rockchip RKNN Toolkit2](https://github.com/airockchip/rknn-toolkit2) | 图转换、算子与MatMul接口、runtime/driver职责 | 单元级任务与host边界明确 | 算子支持名单不等于无CPU fallback或特定芯片特定精度都支持 |
| [NVIDIA NVDLA](https://raw.githubusercontent.com/nvdla/doc/master/doc/hw/v1/hwarch.rst) | 模块化compute/SDP/CDP/BDMA、independent/fused模式、FIFO绕过外存 | 独立MPU/VPU/DMA、融合短向量出口、参数化实例数量 | NVDLA主要面向CNN，不自带本模型Gated DeltaNet；不声称直接移植即可运行 |
| [Gemmini](https://github.com/ucb-bar/gemmini) | output/weight-stationary、banked scratchpad、独立accumulator、load/store/execute控制 | 显式scratchpad/accumulator、双dataflow与分离控制队列 | 原始整数systolic默认配置不能等同本A16/FP32运算与资源成本 |

参考RK3588时保留“模型转换器+runtime+硬件能力表”的分层；本ZCU102裸金属没有Linux driver，A53通过CSR/descriptor控制，但cache/DMA一致性仍需软件处理。分核/多集群是可扩展方向，不把RK3588未公开内部细节写成spec事实。

## 3. 来源与版本限制

provenance.json保存成功读取URL及内容SHA-256；研究阶段读到的第三方代码留在工作区research而未拷贝进本仓库。参考的upstream main不是已执行过的Transformers环境；实际loader版本应固定到checkpoint声明版本并验证，不能只看类名。

权重header的Range读取曾被us.aws.cdn.hf.co代理403阻止，完整权重没有下载。环境草稿已追加该域名并保留huggingface.co，保存不等于当前实例已应用/发布。metadata API显示I32/F16计数，但不是可靠的逐tensordtype/offset替代；导出门槛保留。

source `_tied_weights_keys` 映射lm_head到embedding，config却tie_word_embeddings=false，索引不列独立lm_head。这里不盲目补随机head或自动强制共享；必须用原始loader核验missing/unexpected keys、实际head storage alias及forward logits后导出。

板卡目标为ZCU102 ZU9EG、200MHz裸金属；具体DDR控制器频率、HP/HPC端口和物理地址按用户板卡/BSP填写。AMD UG1182本次网络读取未成功，不把未核验的DDR规格当实测。board profile先保留待测字段。

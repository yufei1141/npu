# 08 layer调度、prefill与扩展

## 1. decode linear layer

```
hidden/residual
 -> zero-centered RMSNorm
 -> MPU W4 in_proj_qkv + in_proj_z
 -> MPU A16 in_proj_a/b
 -> ConvUnit4 + SiLU (QKV)
 -> VPU per-head L2(q,k), q/sqrt128
 -> SFU beta=sigmoid(b), r=exp(-A*softplus(a+dt))
 -> RPU state heads 0..15 (DDR old -> local two-pass -> DDR new)
 -> RMSNormGated(o,z)
 -> W4 out_proj -> residual ADD
 -> zero-centered RMSNorm
 -> W4 gate/up -> SiLU fence -> MUL up fence
 -> W4 down -> residual ADD -> next layer
```

MPU单active macro下projections依序执行，VPU可处理已ready的qkv blocks，普通compiler保守等全部QKV后conv以简化。a/b和qkv/z都要ready后beta/r及headRPU才运行。RPU output可能在DDRstateB返回前可供gatedNorm，但tokenDone必须等待全部18layer state/convjournal完成。

stateDMA可并行prefetchhead(i+1)、writehead(i-1)、computehead i，三slot避免生命周期冲突；若DDR端口共享总带宽有限，scheduler按age与credit限速。RPU与VPU互不共享MUL时可处理前head gatedNorm，输出bank不与RPU正在写的head冲突。

## 2. decode full layer

norm→q_proj(Q+gate)/k/v→Q/K norm→partialRoPE→KV append→attention32keytile→sigmoid(qGate)*attnOut→o_proj→residual→FFN→nextlayer。

append以Bresponse为可见，初版当前token不绕过DDR；后续currentK/V localbypass可优化但相同mask/data验证。fullattention沿context顺序onlineStatecommit，nextQK最多lookahead1，probability/alpha不得提前使用未commitstate。

fullLayerID0..5不是原layerId，KVmanager统一映射。perheadgamma256按zero-centeredNorm处理。门控logits与query来自每headq_proj split，不能flatten切半错位。

## 3. FFN与出口融合

gate/up按outputtile成对pack，当前pair成型后VPU执行SwiGLU同时MPU算后续pair。down先等全部Factivated ready；先不把down改成跨全部output rows的group-majorglobalacc，避免额外容量与循环变更。VPU吞吐应该按EXP/RECIP/MUL次数实际计量，不宣称隐藏为0。

native lm_head选DOT32 row-major持续读，多个row在飞后按词id输出A16logit→argmax；无需vocab完整logitsRAM。最后argmaxDone、全部state/KV/convB完成才TOKEN_COMMIT。

## 4. prefill推荐层内多阶段

v1完整prefill是必选，但实现先decode再prefill不是功能取消。layer-major保证一个layer完整prompt输出才能下一layer，input/outputhidden HBM ping-pong。projection/FFN用M16四query×16row权重复用，不默认大mesh。

linear推荐HEAD_MAJOR_FULL_PROMPT：
1. 逐M16block完成norm、qkv/z/a/b投影、conv，并存本layerconv后的A16QKV、Z、beta/r于DDR scratch；convhistory跨block连续。
2. RPU按head0..15：oldstate读一次，在localheadcache连续处理整个新增prompt S token，q/k当前tokenL2Norm输出FP32，valuecastFP32；输出o逐token写scratch；最后一次写newstate。GDN数学依赖只在同headtoken序列，head间可重新排序。
3. gatedNorm(o,z)→out_proj/residual→FFN逐M16block处理，输出nextlayerhidden；释放本layer QKV/Z/O scratch。

S4096下QKV48MiB、Z16MiB、O16MiB、coef≈0.5MiB，input/outputhidden各8MiB，peak约96.5MiB外存scratch（未计alignment/其他临时）；同层生命期释放后可复用。RPU stateDDR一次读写36MiB，而非S×36MiB。计算4stateMUL/token没有减少。

低scratch选BLOCK_MAJOR_M16：每block跨head更新stateworkregion，stateDMA每16token约36MiB；committedoriginalstate区保持不变，后续block读写workregion，整个prefill完成前不交换global表。出错可discardwork重新从原committedprefix做prefill，不半恢复。

fullprefill：所有Q/K/V projection/norm/RoPE完成并append新增KV（可分block写），causalmask保证不看future；queryM16按keytile32处理。Q/K positions是绝对T0+q；final only最后prompttoken lm_head，其他tokenlogitsdebug可选。

## 5. 事务与缓存一致性

session committedLen、state table、convhistory version同时提交。未提交KV位置已写也不会被旧validLen消费。prefill layerworking lengths不同，用per-layer visibleLen/workingVersion，但globalsession不提前提交。abort丢workingtensor、排空writes；不能把layer0已prefill完成当全模型token完成。

DCU event依赖明确success/fault；fetch完成与tensorformatpack完成区分。meta<->dataqueue顺序以tag匹配，无ready期间不推进计数器。软件不得复用尚未ack的DMAbuffer。

## 6. 拓展路径

从32到64W4先确认payloadBW约3.3→6.6GB/s；从64到128需要约13.2GB/s，否则body仅更闲。提高prefillM减少weighttraffic但可能计算已主导，要同时增加可用产品位置才明显改善TTFT。

多cluster时每cluster有localpartial/scratchpad/queue，权重按输出行partition，input广播复制；RPU按headpartition，allDone barrier汇总state提交。输出head可按vocab行partition再全局argmax。PSDDR bandwidth总量不随cluster数量自动增长。

如将来加视觉，定义ExternalEmbedding入口及三轴positionid buffer，独立扩vision编译图/算子；当前能力位textOnly=1，不声称完整MiniCPM图像推理。

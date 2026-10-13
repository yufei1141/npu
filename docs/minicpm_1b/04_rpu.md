# 04 RPU：FP32 Gated DeltaNet递归状态引擎

RPU=Recurrent Processing Unit。独立于MPU，并非用BF16矩阵路径将FP32状态偷偷降精度。首版16FP32 MUL lane+16FP32 ADD lane、norm/reduce共享VPU或局部partial上下文；可扩8/16/32lane。

## 1. 数据与state容量

每linear layer16head，每head128×128 FP32=64KiB；每layer1MiB，18layer18MiB。state固定容量，不随context线性增长；只有6full layers拥有完整KV。

状态存DDR，片上至少3个head slot，每64KiB，role分别current compute、next fetch、previous writeback。采用固定head序也不保证三slot完全隐藏latency，但允许read/compute/write各持不同slot而无环等待。slot含epoch、layerId、headId、tokenVersion、dirty、remainingDMA。

q/k各128 FP32，v128由A16转FP32；beta按框架sigmoid输出A16 fence后转FP32；r=exp(g) FP32。q已L2Norm并除sqrt128，k仅L2Norm。r∈[0,1]正常，负值/NaN按NUMERIC_FAULT，不能clamp掩盖错误。g的计算属于05 Scalar/SFU。

## 2. 保留源语义的两遍处理

pass1逐i/k维读取S[i,j]（j方向16lane并行）：MUL(r,S)得Sd，写回本地slot；MUL(k_i,Sd)累加u[j]。所有i完成及partial归约退休后，delta[j]=beta*(v[j]-u[j])。

pass2逐i读本地Sd：MUL(k_i,delta[j])，ADD成Snew写回slot；MUL(q_i,Snew)累加o[j]。最后o归约完成，castA16交给gated norm，同时DIRTY slot发DDR写回。每state元素4MUL、3ADD；shared16MUL每pass需2发射拍/16元素，理想算术吞吐8state元素/拍。

localSd不写DDR；每token每head只一次64KiB读、一次64KiB写，不做三次DDR扫描。headDMA read/write共128KiB，18×16head共36MiB（37,748,736B）/token。layer级写完成是state commit必要条件，不能只等o产生。

重排候选 `u_old=k^T S,o_old=q^T S` 后再decay是数学等价但FP32不同舍入；不作为当前基线。外积修正的Snew必须用于o，不能从oldS读o漏掉当前token更新。

## 3. row-major state bank

logical S[i,j]，bank=j mod16，row=i*8+j//16；每bank1024 FP32word，16bank合计64KiB。扩展通用公式row=i*ceil(Dv/lanes)+j//lanes；Dk/Dv或lane改变必须重新算slot容量和mask，不复用常数8。每beat固定i与jBatch，16个不同bank同时read/write。q_i、k_i、r广播；delta/o/u按j向量寄存或banked small RAM。

每j累加u/o存在feedback：一个i有8个jBatch×2MUL阶段，同行j下次更新至少16实际发射拍。若Ladd>16，用C个partial上下文或显式stall。不能只认为16lane就自然解决RAW。首版partial按i modC，以固定平衡树归约，C由latency检查产生。

pass1两MUL必须保留Sd与对应k_i/tag，不能让state回写读取到下一i的value。pass2MUL/ADD/MUL/ADD异步latency，用StepAligner和per-bank scoreboard。read-after-write同址必须等Sd/Snew writeAck或经过已完成bypass。

## 4. 模块与端口

RpuCmd：TxnTag、layer/head、stateSlotRef、qRef/kRef/vRef、beta32/r32、stateVersion、outRef。输入cmd为Stream；外部stateDDR搬运由DMA，不在RPU内藏AXI。

子模块：HeadSlotManager、StateReadWriter、DecayPipeline、KeyReadReduce、DeltaBuilder、RankOneUpdate、QueryReadReduce、OutputCaster、CommitTracker、FpExceptionCollector。物理MUL/ADD可由phase mux复用；表中模块不是每个都复制16lane。

StateBeat含tag、i7、jBase7、S[16]FP32、mask16、last；U/O向量写回含head、jBase、FP32/A16结果。输出headDone区分ARITH_DONE与STATE_DDR_COMMITTED，DCU最终只消费后者做token完成。

## 5. FSM与事务

WAIT_SLOT_READY→LOAD_QKV_SCALARS→PASS1_ISSUE→PASS1_DRAIN→REDUCE_U→BUILD_DELTA→PASS2_ISSUE→PASS2_DRAIN→REDUCE_O→EMIT_O→QUEUE_STATE_STORE→WAIT_B→COMMIT。

允许前headDDR写与下一head计算并行；RPU当前head o供后续norm时，整layer stateDone仍未完成。各head的stateVersion从t-1到t，不允许同head下一token在上一写未确认时从DDR读取旧版。prefill可连续在片上更新同head多个token，最后一次统一写并提交block版本。

globaltoken层间异常：不可能在18MiB state更新中靠一个寄存器回滚。支持两个明确模式：
- FAST_NO_ROLLBACK：任何中途fault使session INVALID，必须reset state并重prefill，不继续使用部分更新状态。
- JOURNALED：可选DDR old/new state双region，写新region并在完整token结束原子交换所有layer版本；额外18MiB容量但不额外复制旧state（计算本来读旧写新）。full KV用validLen防未提交append。

基线选JOURNALED以可恢复语义，token0读zeroInitial逻辑标志而非未初始化DDR；state snapshot保存version及layout。缓存dirty不得跨epoch错误flush。

## 6. Prefill扩展

M16 block时，一个head读取state一次，连续执行该block所有validtoken，缓存o[16,128]及state在slot，最终一次write。DDR state读写每block36MiB，而非每token；并不减少每token的FP32 recurrence算术。

q/k/v/z来自该block已完成conv和投影；每token独立norm/beta/r，序号不可跳。head-major处理完16token再下一head，输出scatter至[token,head,dim]，该token完整head输出ready后才gated norm/out_proj。能否跨head overlap取决于资源独立，不由软件线程数量决定。

## 7. 验证

独立reference逐token更新；compare两遍tile实现与高精度oracle及固定FP32舍入模型。覆盖r0/r1、beta0/beta1、非零initialstate、不同tokenblock拆分、lane8/16/32、反压/写B延迟、故障新旧版本不混合、state-bank所有地址唯一。source或matrixFMA实现变化必须重验，不能对比仅最终top1而忽略state漂移。

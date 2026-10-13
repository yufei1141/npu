"""MiniCPM text-only roofline/stage model. Not RTL, not a board benchmark."""
from pathlib import Path
import json,math
ROOT=Path(__file__).parents[1]
c=json.loads((ROOT/'sources/model_config.json').read_text())['text_config']
H,F,L=c['hidden_size'],c['intermediate_size'],c['num_hidden_layers']
FA=c['layer_types'].count('full_attention');LA=L-FA
QH,KH,D=c['num_attention_heads'],c['num_key_value_heads'],c['head_dim']
LH,DK,DV=c['linear_num_value_heads'],c['linear_key_head_dim'],c['linear_value_head_dim']
V=c['vocab_size'];HZ=200e6
ffn=3*H*F
w4_linear=H*(2*LH*DK+LH*DV)+H*LH*DV+LH*DV*H
w4_full=H*(2*QH*D)+2*H*(KH*D)+(QH*D)*H
w4_params=LA*w4_linear+FA*w4_full+L*ffn
head_params=H*V
state_elems=LA*LH*DK*DV
# Native small projections a,b, convolution weights and norm/gates (norm estimate).
small_bytes=2*(LA*(2*H*LH + 4*(2*LH*DK+LH*DV))+L*3*H+FA*2*D)
base={
 'hidden':H,'ffn':F,'full_layers':FA,'linear_layers':LA,
 'w4_matrix_parameters':w4_params,'native_head_parameters':head_params,
 'w4_payload_scale_bytes':w4_params*(.5+2/128),
 'w4_payload_scale_gptq_zero_bytes':w4_params*(.5+2/128+.5/128),
 'native_head_bytes':head_params*2,'small_param_bytes_estimate':small_bytes,
 'fp32_recurrent_state_bytes':state_elems*4,
 'decode_state_read_write_bytes':state_elems*8,
 'minimal_conv_history_bytes':LA*3*(2*LH*DK+LH*DV)*2,
 'kv_bytes_per_token_all_full_layers':FA*2*KH*D*2}
archs=[
 dict(name='A_unified32',w4=32,bf16=32,state=8,matrix_eff=.70,state_eff=.65,head='native16',state_shared=True),
 dict(name='B_stream64_recommended',w4=64,bf16=32,state=16,matrix_eff=.80,state_eff=.80,head='native16',state_shared=False),
 dict(name='C_mesh256_prefill',w4=256,bf16=128,state=16,matrix_eff=.45,state_eff=.80,head='native16',state_shared=False),
 dict(name='D_stream64_headW4',w4=64,bf16=32,state=16,matrix_eff=.80,state_eff=.80,head='w4_candidate',state_shared=False),
 dict(name='E_W4A8_128_candidate',w4=128,bf16=32,state=16,matrix_eff=.80,state_eff=.80,head='w4_candidate',state_shared=False)]

def estimate(a,bw_gbs=6,context=4096,head_bw_gbs=None):
 bw=bw_gbs*1e9; hb=(head_bw_gbs or bw_gbs)*1e9; eff=a['matrix_eff']
 # Per layer roofline rather than single whole-model max. Counts include the Q output gate.
 stages=[]
 for kind in c['layer_types']:
  mat=w4_full if kind=='full_attention' else w4_linear
  for label,n in [('projection',mat),('FFN',ffn)]:
   compute=n/(a['w4']*HZ*eff)
   mem=n*(.5+2/128+.5/128)/bw
   stages.append(max(compute,mem))
 body=sum(stages)
 head_w4=a['head']=='w4_candidate'
 hp=a['w4'] if head_w4 else a['bf16']
 headcomp=head_params/(hp*HZ*eff)
 headmem=head_params*(.5+2/128+.5/128 if head_w4 else 2)/hb
 head=max(headcomp,headmem)
 # Source-order FP32 recurrence: decay, k read, outer correction, q read -> 4 MULs/state.
 state_compute=4*state_elems/(a['state']*HZ*a['state_eff'])
 state_memory=state_elems*8/bw
 state=max(state_compute,state_memory)
 att_products=2*FA*QH*D*context
 attention_compute=att_products/(a['bf16']*HZ*eff)
 attention_memory=base['kv_bytes_per_token_all_full_layers']*context/bw
 attention=max(attention_compute,attention_memory)
 # Explicit placeholder budget, not measured: VPU, launches, drain, local packing, small params.
 misc=.003 + small_bytes/bw
 total=body+head+state+attention+misc
 return dict(architecture=a['name'],context=context,effective_GBps=bw_gbs,
  body_ms=body*1000,head_ms=head*1000,state_ms=state*1000,attention_ms=attention*1000,
  misc_budget_ms=misc*1000,total_ms=total*1000,tokens_per_second=1/total,
  qualifies='conditional analytical budget; no FPGA resource-fit/accuracy guarantee')

rows=[estimate(a,bw,t) for a in archs for bw in (4,6,8) for t in (1024,4096,8192)]
# Prefill has repeated states inside a block, not decode's external state transfer each token.
prefill=[]
a=archs[1]
for S in (128,512,4096):
 compute=(w4_params*S)/(a['w4']*HZ*a['matrix_eff'])
 attention=2*FA*QH*D*(S*(S+1)/2)/(a['bf16']*HZ*a['matrix_eff'])
 recurrence=4*state_elems*S/(a['state']*HZ*a['state_eff'])
 # Weight reuse M16, including GPTQ zeros. Roofline still dominated by compute in this profile.
 weightmem=math.ceil(S/16)*w4_params*(.5+2/128+.5/128)/6e9
 core=max(compute,weightmem)+attention+recurrence
 prefill.append(dict(prompt_tokens=S,core_seconds_floor_under_eff_assumptions=core,
                      prompt_tokens_per_second=S/core,note='excludes scratch, norm, state transfers, launches and final head'))
assert w4_params==497025024
assert base['fp32_recurrent_state_bytes']==18*1024*1024
assert base['kv_bytes_per_token_all_full_layers']==12288
result=dict(model_revision=json.loads((ROOT/'sources/model_revision.json').read_text()),base=base,
 assumptions=dict(clock=HZ,gb='decimal',state_precision='FP32',context_includes_current=True,
                 misc_ms_placeholder=3,small_params_rough=True,
                 head_alias='tensor-header/loader unresolved, native16 head traffic retained',
                 dense_mesh_utilization='chosen comparison assumption, NOT measured',
                 omitted=['resource fit','exact IP latency/II','bank/DDR arbitration','SFU error','tokenizer time']),
 architectures=archs,decode=rows,prefill=prefill)
(ROOT/'design_space_results.json').write_text(json.dumps(result,indent=2))
print('Matrix count',w4_params,'native16 head bytes',base['native_head_bytes'])
for a in archs:
 r=estimate(a);print(a['name'],round(r['total_ms'],2),'ms',round(r['tokens_per_second'],2),'tok/s')
for r in prefill:print('Prefill',r['prompt_tokens'],round(r['core_seconds_floor_under_eff_assumptions'],2),'sec')

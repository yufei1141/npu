"""Analytic resource/phase model, NOT an RTL or end-to-end timing simulation."""
import json, math
from pathlib import Path
C=lambda n,d:(n+d-1)//d
H,F,NQ,NKV,D,L,V=4096,12288,32,8,128,36,151936
hz=200_000_000
linear=H*H+2*H*NKV*D+H*H+3*H*F
rows=[]
for M in (1,32):
 for width in (16,32,64):
  el=M*F
  gateup=2*M*H*F//1024; down=M*H*F//1024
  # exp and reciprocal share one SFU array; three stable-sigmoid/SwiGLU multiplies share MUL.
  shared_sfu=2*C(el,width)
  mul=3*C(el,width)
  add=C(el,width)
  resource=max(shared_sfu,mul,add)
  serial=shared_sfu+mul+add
  rows.append(dict(M=M,width=width,gateup_issue=gateup,down_issue=down,
                   swiglu_resource_floor=resource,swiglu_serial_phase_issue=serial,
                   fraction_of_down=resource/down))
attention=[]
for width in (16,32,64):
 E=NQ*128; O=NQ*D
 # max scan, exp+sum (assuming distinct reducer), old-o scale, combine.
 phases=dict(max_scan=C(E,width),exp_sum=C(E,width),old_o_scale=C(O,width),combine=C(O,width))
 extra=sum(phases.values())
 attention.append(dict(width=width,QK_issue=512,PV_issue=512,vector_phases=phases,
                       serialized_tile_issue=1024+extra,
                       relative_extra=extra/1024,
                       extra_4k_36layers_ms=extra*32*L/hz*1000))
bandwidth=[]
for eta in (1,.9,.8,.7):
 wneed=528*hz
 channels=C(math.ceil(wneed/eta),64*hz)
 lm=V*H*2/(12*64*hz*eta)
 bandwidth.append(dict(efficiency=eta,min_weight_channels=channels,
                       lm_head_12channels_ms=lm*1000,
                       linear_weight_floor_ms=max(linear*L/1024/hz,
                           linear*L*(.5+2/128)/(10*64*hz*eta))*1000))
result=dict(assumptions='issue/resource lower bounds; excludes pipelines, bank stalls, causal tails, and actual IP initiation intervals',
            swiglu=rows,attention=attention,bandwidth=bandwidth,
            qk_output_burst=dict(elements=128,producer_interval=16,consumer_width=32,
                                 minimum_stall_free_capacity=128,
                                 recommendation='two 128-element FP32 skid/result banks plus reserved in-flight capacity'),
            scale=dict(elements_per_group=128,group_period=16,scale_width=8,
                       utilization=128/(16*8),note='100% utilization: no recovery margin'))
assert linear==192937984
assert rows[4]['swiglu_resource_floor']==36864
assert attention[1]['serialized_tile_issue']==1536
path=Path(__file__).with_name('npu_bottleneck_results.json')
path.write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))

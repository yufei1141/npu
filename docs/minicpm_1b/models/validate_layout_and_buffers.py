"""Logical layout, descriptor packing and buffer occupancy checks."""
import math,struct,json
from pathlib import Path
from itertools import product
# Descriptor ABI sizes and zero reserved encodings.
formats={'macro':'<BBH8H9I8s','tensor':'<QQ3I3I4BQI8s','gdn_ext':'<16H5I12s'}
assert all(struct.calcsize(fmt)==64 for fmt in formats.values())
# Every state entry and KV tile element has a unique bank/row address.
state=[(j%16,i*8+j//16) for i,j in product(range(128),range(128))]
assert len(set(state))==128*128
K=lambda h,p,d:((h,p%4),(p//4)*256+d)
V=lambda h,p,d:((h,d%8),(d//8)*32+p)
for func in (K,V):
 addrs=[func(*x) for x in product(range(2),range(32),range(256))]
 assert len(set(addrs))==2*32*256
for batch,d in product(range(8),range(256)):
 req={}
 for qh,pl in product(range(8),range(4)):
  bank,row=K(qh//4,batch*4+pl,d);req.setdefault(bank,set()).add(row)
 assert all(len(r)==1 for r in req.values())
for h,db,p in product(range(2),range(32),range(32)):
 req={}
 for d in range(db*8,db*8+8):
  bank,row=V(h,p,d);req.setdefault(bank,set()).add(row)
 assert len(req)==8 and all(len(r)==1 for r in req.values())
# First-generation DOT group and new output-stationary buffer lifetimes differ.
slots_os=math.ceil((128+14+64)/128)
slots_dot=math.ceil((16+30+16)/16)
slots_pv=math.ceil((32+14+32)/32)
slots_head=math.ceil((32+54+32)/32)
assert slots_os==2 and slots_dot==4 and slots_pv==3 and slots_head==4
for latency in (1,8,16,20):
 C=1<<(latency-1).bit_length()
 assert C>=latency
 if latency>16:assert C>16
# Pool journaling never publishes incomplete working state.
old={'version':7,'region':0};work={'version':8,'region':1}
committed=old.copy();heads_done=287
if heads_done==18*16:committed=work.copy()
assert committed==old
heads_done+=1
if heads_done==18*16:committed=work.copy()
assert committed==work
r=dict(result='passed',descriptor_sizes={k:struct.calcsize(v) for k,v in formats.items()},
       state_entries=16384,kv_entries_each=16384,group_slots=dict(output_stationary=slots_os,dot_old_example=slots_dot,pv_short_tile=slots_pv,head_contexts=slots_head),
       scope='logical addresses/ABI/invariants; physical RAM ports and real DMA still unvalidated')
Path(__file__).parents[1].joinpath('layout_validation_results.json').write_text(json.dumps(r,indent=2))
print('PASS: 64-byte ABIs, state/KV uniqueness/read-bank limits, group slots, feedback contexts and commit invariant.')

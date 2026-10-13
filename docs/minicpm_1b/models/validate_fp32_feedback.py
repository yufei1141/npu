"""Latency-aware RAM/ADD feedback simulation. Integer payloads test hazards, not FP accuracy."""
from pathlib import Path
import json,math

def run(C,Lram,Ladd,Lwrite,II,count=128,guard=1):
    sums=[0]*C; events={}; busy_until=[-1]*C
    t=accepted=stalls=0;next_issue=0;last_accept=None
    while accepted<count or events:
        for ctx,val in events.pop(t,[]):
            sums[ctx]=val
        if accepted<count and t>=next_issue:
            ctx=accepted%C
            # Same-cycle write/read unspecified: guard enforces a later read edge.
            if t < busy_until[ctx]+guard:
                stalls+=1
            else:
                old=sums[ctx]
                commit=t+Lram+Ladd+Lwrite
                events.setdefault(commit,[]).append((ctx,old+accepted+1))
                busy_until[ctx]=commit
                accepted+=1;next_issue=t+II;last_accept=t
        t+=1
        assert t<10000
    assert sum(sums)==sum(range(1,count+1))
    d=Lram+Ladd+Lwrite+guard
    return dict(C=C,Lram=Lram,Ladd=Ladd,Lwrite=Lwrite,II=II,guard=guard,
                loop_safe_spacing=d,min_contexts=math.ceil(d/II),stall_cycles=stalls,
                accepted=count,cycles_with_drain=t,issue_cycles=last_accept+1)

cases=[run(8,2,8,1,1),run(16,2,8,1,1),run(16,2,20,1,1),
       run(32,2,20,1,1),run(4,2,8,1,4),run(16,2,8,1,8)]
assert cases[0]['stall_cycles']>0 and cases[1]['stall_cycles']==0
assert cases[2]['stall_cycles']>0 and cases[3]['stall_cycles']==0
assert cases[4]['stall_cycles']==0 and cases[5]['issue_cycles']>=8*127
# Partial row banks can read one ctx each cycle; transposed snapshot gathers C times.
R,C=64,16
partial=[[r*1000+c for c in range(C)] for r in range(R)]
snapshot=[[None]*C for _ in range(R)]
for c in range(C):
    beat=[partial[r][c] for r in range(R)]
    for r in range(R):snapshot[r][c]=beat[r]
assert snapshot==partial
# Every selected row now exposes all C values without requesting C reads from one RAM.
assert all(sum(snapshot[r])==sum(partial[r]) for r in range(R))
hold=128+14+2+16+1
assert math.ceil(hold/128)==2
result=dict(cases=cases,partial_rows=R,contexts=C,capture_cycles=C,
 snapshot_storage_per_slot_bytes=R*C*4,partial_hold_example=hold,
 scope='cycle-level feedback/transpose invariants; not vendor FP IP/RTL or full-NPU timing')
Path(__file__).parents[1].joinpath('fp32_feedback_results.json').write_text(json.dumps(result,indent=2))
for x in cases:print(f"C={x['C']:2} Ladd={x['Ladd']:2} II={x['II']} stalls={x['stall_cycles']:3} issue_cycles={x['issue_cycles']}")
print('PASS: 6 feedback cases and 16-cycle partial-to-snapshot transpose.')

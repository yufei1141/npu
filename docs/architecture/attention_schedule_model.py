"""Finite-slot, dependency-aware attention stage model; not arithmetic RTL."""
import json
from pathlib import Path

def simulate(tokens=4096,slots=2,lookahead=False,bytes_per_cycle=716,hbm_latency=40,
             qk_cycles=512,pv_cycles=512,prep_cycles=272,merge_cycles=136):
    count=(tokens+127)//128
    jobs=[dict(loaded=False,slot=None,qk_started=False,qk_done=False,
               prep_started=False,prep_done=False,pv_started=False,pv_done=False,
               merge_started=False,committed=False) for _ in range(count)]
    kv=[None]*slots
    dma=mpu=vpu=None
    trace=[]; stats=dict(mpu_busy=0,vpu_busy=0,dma_busy=0,kv_high_water=0,
                       score_high_water=0,prob_high_water=0,mpu_idle=0)
    t=0
    def event(kind,i): trace.append(dict(cycle=t,event=kind,tile=i))
    while not all(j['committed'] for j in jobs):
        if t>1_000_000: raise AssertionError('model timeout/deadlock')
        if dma and dma[1]==t:
            i=dma[0];jobs[i]['loaded']=True;event('KV_READY',i);dma=None
        if mpu and mpu[2]==t:
            phase,i,_=mpu;j=jobs[i]
            if phase=='QK':j['qk_done']=True;event('QK_DONE',i)
            else:
                j['pv_done']=True
                assert kv[j['slot']]==i
                kv[j['slot']]=None;j['slot']=None
                event('PV_DONE_KV_RELEASE',i)
            mpu=None
        if vpu and vpu[2]==t:
            phase,i,_=vpu;j=jobs[i]
            if phase=='PREP':j['prep_done']=True;event('P_SCALE_READY',i)
            else:
                assert i==0 or jobs[i-1]['committed']
                j['committed']=True;event('STATE_COMMIT',i)
            vpu=None
        # All requests pre-reserve their finite storage; earliest merge has priority.
        if vpu is None:
            target=next((i for i,j in enumerate(jobs) if j['pv_done'] and not j['merge_started']),None)
            if target is not None:
                jobs[target]['merge_started']=True
                vpu=('MERGE',target,t+merge_cycles);event('MERGE_START',target)
            else:
                target=next((i for i,j in enumerate(jobs)
                             if j['qk_done'] and not j['prep_started']
                             and (i==0 or jobs[i-1]['committed'])),None)
                if target is not None:
                    jobs[target]['prep_started']=True
                    vpu=('PREP',target,t+prep_cycles);event('PREP_START',target)
        if mpu is None:
            target=next((i for i,j in enumerate(jobs) if j['prep_done'] and not j['pv_started']),None)
            if target is not None:
                jobs[target]['pv_started']=True
                mpu=('PV',target,t+pv_cycles);event('PV_START',target)
            else:
                active_qk=sum(j['qk_started'] and not j['pv_done'] for j in jobs)
                score_reserved=sum(j['qk_started'] and not j['prep_done'] for j in jobs)
                for i,j in enumerate(jobs):
                    allowed=(lookahead and active_qk<2 and score_reserved<2) or (
                        not lookahead and (i==0 or jobs[i-1]['committed']))
                    if j['loaded'] and not j['qk_started'] and allowed:
                        assert j['slot'] is not None
                        j['qk_started']=True;mpu=('QK',i,t+qk_cycles);event('QK_START',i);break
        if dma is None and None in kv:
            target=next((i for i,j in enumerate(jobs) if not j['loaded'] and j['slot'] is None),None)
            if target is not None:
                slot=kv.index(None);kv[slot]=target;jobs[target]['slot']=slot
                nvalid=min(128,tokens-target*128)
                duration=hbm_latency+(nvalid*4096+bytes_per_cycle-1)//bytes_per_cycle
                dma=(target,t+duration);event('KV_FETCH',target)
        stats['mpu_busy']+=int(mpu is not None)
        stats['mpu_idle']+=int(mpu is None)
        stats['vpu_busy']+=int(vpu is not None)
        stats['dma_busy']+=int(dma is not None)
        stats['kv_high_water']=max(stats['kv_high_water'],sum(x is not None for x in kv))
        stats['score_high_water']=max(stats['score_high_water'],sum(j['qk_started'] and not j['prep_done'] for j in jobs))
        stats['prob_high_water']=max(stats['prob_high_water'],sum(j['prep_started'] and not j['pv_done'] for j in jobs))
        assert stats['score_high_water']<=2
        assert stats['prob_high_water']<=2
        for i,j in enumerate(jobs):
            assert not j['pv_started'] or j['prep_done']
            assert not j['prep_started'] or i==0 or jobs[i-1]['committed']
            assert not j['committed'] or j['pv_done']
        t+=1
    assert stats['mpu_busy']==count*(qk_cycles+pv_cycles)
    return dict(tokens=tokens,slots=slots,lookahead=lookahead,bytes_per_cycle=bytes_per_cycle,
                cycles=t,us=t/200,mpu_utilization=stats['mpu_busy']/t,stats=stats,first_events=trace[:24])

if __name__=='__main__':
    scenarios=[]
    for bw in (512,716,896):
        for slots,overlap in ((2,False),(2,True),(3,True)):
            scenarios.append(simulate(slots=slots,lookahead=overlap,bytes_per_cycle=bw))
    checks=[]
    for n in (1,127,128,129,4096):
        a=simulate(tokens=n,slots=2,lookahead=False)
        b=simulate(tokens=n,slots=3,lookahead=True)
        checks.append(dict(tokens=n,serial=a['cycles'],lookahead=b['cycles']))
    payload=dict(scope='serial DMA fills, fused max, abstract PREP and MERGE, finite KV/score/prob slots; excludes bank conflicts, IP numerical operations and complete model program',
                 scenarios=scenarios,boundary_checks=checks)
    Path(__file__).with_name('attention_schedule_results.json').write_text(json.dumps(payload,indent=2))
    for x in scenarios:
        print(f"BW={x['bytes_per_cycle']:3} slot={x['slots']} lookahead={x['lookahead']} cycles={x['cycles']:5} MPU_util={x['mpu_utilization']:.3f}")
    print('Dependency/storage assertions and 5 token-length boundary cases completed.')

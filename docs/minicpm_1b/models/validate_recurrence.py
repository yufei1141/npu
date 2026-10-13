"""Independent algorithm validation. Does not load GPTQ tensors or emulate vendor IP."""
import json
from pathlib import Path
import numpy as np

def oracle(s,q,k,v,b,r):
    sd=s*r
    u=np.einsum('i,ij->j',k,sd)
    delta=b*(v-u)
    sn=sd+np.outer(k,delta)
    out=np.einsum('i,ij->j',q,sn)
    return sn,out

def scan_two_pass(s,q,k,v,b,r,lanes):
    dk,dv=s.shape
    sd=np.empty_like(s);u=np.zeros(dv,dtype=s.dtype)
    # Separate local state decay/write and k read, then delta before any update.
    for i in range(dk):
        for j in range(0,dv,lanes):
            sl=slice(j,min(j+lanes,dv))
            sd[i,sl]=s[i,sl]*r
            u[sl]=u[sl]+k[i]*sd[i,sl]
    delta=b*(v-u);sn=np.empty_like(s);out=np.zeros(dv,dtype=s.dtype)
    for i in range(dk):
        for j in range(0,dv,lanes):
            sl=slice(j,min(j+lanes,dv))
            sn[i,sl]=sd[i,sl]+k[i]*delta[sl]
            out[sl]=out[sl]+q[i]*sn[i,sl]
    return sn,out

rng=np.random.default_rng(20261013)
cases=[]
for dtype in (np.float64,np.float32):
 for lanes in (1,4,16):
  s=rng.normal(0,.05,(8,12)).astype(dtype)
  for b,r in ((0,0),(0,1),(1,0),(1,1),(.6,.93)):
   q=rng.normal(0,.1,8).astype(dtype);k=rng.normal(0,.1,8).astype(dtype)
   v=rng.normal(0,.1,12).astype(dtype)
   s1,o1=oracle(s,q,k,v,dtype(b),dtype(r))
   s2,o2=scan_two_pass(s,q,k,v,dtype(b),dtype(r),lanes)
   np.testing.assert_allclose(s2,s1,rtol=2e-5 if dtype==np.float32 else 1e-12,atol=2e-7 if dtype==np.float32 else 1e-12)
   np.testing.assert_allclose(o2,o1,rtol=2e-5 if dtype==np.float32 else 1e-12,atol=2e-7 if dtype==np.float32 else 1e-12)
  cases.append(dict(dtype=dtype.__name__,lanes=lanes,corner_cases=5))
# Actual state dimensions and stream/chunk equivalence of the same operation order.
seq=[]
for _ in range(33):
 q=rng.normal(0,.1,128).astype('float32');k=rng.normal(0,.1,128).astype('float32')
 q=q/np.sqrt(np.sum(q*q)+np.float32(1e-6))/np.sqrt(np.float32(128))
 k=k/np.sqrt(np.sum(k*k)+np.float32(1e-6))
 seq.append((q,k,rng.normal(0,.1,128).astype('float32'),np.float32(.7),np.float32(.9)))
s0=np.zeros((128,128),dtype='float32');sa=s0.copy();outs=[]
for q,k,v,b,r in seq:sa,o=scan_two_pass(sa,q,k,v,b,r,16);outs.append(o)
for chunk in (1,8,16,32):
 sb=s0.copy();ob=[]
 for start in range(0,len(seq),chunk):
  for q,k,v,b,r in seq[start:start+chunk]:sb,o=scan_two_pass(sb,q,k,v,b,r,16);ob.append(o)
 np.testing.assert_array_equal(sb,sa);np.testing.assert_array_equal(ob,outs)
# Minimal three-sample conv history vs an independent full prefix dot.
w=rng.normal(0,.1,(4,7)).astype('float32');x=rng.normal(0,.1,(17,7)).astype('float32')
hist=np.zeros((3,7),dtype='float32')
for t in range(len(x)):
 win=np.vstack([hist,x[t]]);got=np.sum(win*w,axis=0)
 expected=np.zeros(7,dtype='float32')
 for tap in range(4):
  source=t-3+tap
  if source>=0:expected=expected+x[source]*w[tap]
 np.testing.assert_allclose(got,expected,rtol=1e-6,atol=1e-7)
 hist=win[1:].copy()
result=dict(seed=20261013,cases=cases,corner_cases=30,chunk_cases=4,conv_steps=17,
            result='passed',scope='algorithm and stream/chunk invariants; no checkpoint accuracy or RTL claim')
Path(__file__).parents[1].joinpath('recurrence_validation_results.json').write_text(json.dumps(result,indent=2))
print('PASS: 30 recurrence corner cases, 4 chunk cases at 128x128, 17 conv-history steps.')

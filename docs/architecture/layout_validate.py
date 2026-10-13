"""Check logical address uniqueness, compute-bank reads and reduction counts."""
from itertools import product

def key_addr(h,p,d):return (h,d//8),p//32,(p%32)*8+d%8

def value_addr(h,p,d):return (h,d%32),(d//32)*16+p//8,p%8

for func in (key_addr,value_addr):
    addresses=[func(h,p,d) for h,p,d in product(range(8),range(128),range(128))]
    assert len(addresses)==len(set(addresses))==131072
# Decode QK reads four positions per head, all lanes share a bank word.
for batch,step in product(range(32),range(16)):
    words={}
    for qh,pl,dl in product(range(32),range(4),range(8)):
        bank,row,element=key_addr(qh//4,batch*4+pl,step*8+dl)
        words.setdefault(bank,set()).add(row)
    assert all(len(rows)==1 for rows in words.values())
# Prefill QK: one KV head, 32 positions; four queries reuse each operand.
for h,batch,step in product(range(8),range(4),range(16)):
    words={}
    for p,dl in product(range(batch*32,(batch+1)*32),range(8)):
        bank,row,element=key_addr(h,p,step*8+dl)
        words.setdefault(bank,set()).add(row)
    assert all(len(rows)==1 for rows in words.values())
# Both decode and prefill PV: 32 dims share V across 4 heads/queries.
for h,db,step in product(range(8),range(4),range(16)):
    words={}
    for d,p in product(range(db*32,(db+1)*32),range(step*8,(step+1)*8)):
        bank,row,element=value_addr(h,p,d)
        words.setdefault(bank,set()).add(row)
    assert len(words)==32 and all(len(rows)==1 for rows in words.values())
assert sum([64,32,16,8])==8*15==120
assert 128*16*4*2==16384
print('K/V each 131072 unique element addresses; decode/prefill compute read-word contracts and group-store capacity verified.')
print('This verifies logical banking only, not physical FPGA RAM inference, timing, write packing or RTL.')

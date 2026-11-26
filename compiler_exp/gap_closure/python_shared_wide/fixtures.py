"""Static source and independent modulo64 references for typed shared arrays."""
import random
import struct
MASK=(1<<64)-1
SENTINEL=0x13579bdf2468ace0
HEADER='from openassembler.pyfront import *\n'

def cases():
    for dtype in('i64','u64'):
        for mode in('exchange','literal','mixed','loop','maximum','padded_maximum','nested','if_else','phases','legacy32'):
            name=dtype+'_'+mode
            count=6144 if mode=='maximum'else 6143 if mode=='padded_maximum'else 128
            before='    prefix = shared_array(i32, 1)\n'if mode in('mixed','padded_maximum','legacy32')else''
            after='    suffix = shared_array(i32, 3)\n'if mode=='mixed'else''
            prefix_write='    if t == 0:\n        prefix[0] = 0x12345678\n'if before else''
            suffix_write='    if t == 0:\n        suffix[0] = 0x76543210\n        suffix[1] = -1\n        suffix[2] = -2147483648\n'if after else''
            reader='    v = shared[t ^ 64]\n'
            body='    shared[t] = a[i]\n'
            if mode=='literal':
                body='    if t == 127:\n        shared[127] = a[i]\n';reader='    v = shared[127]\n'
            elif mode=='loop':
                reader='''    v = a[i]
    for step in range(3):
        v = shared[t ^ 64] + delta
        syncthreads()
        shared[t] = v
        syncthreads()
'''
            tail=''
            if before:tail+='        out[i * 4 + 1] = '+dtype+'(prefix[0])\n'
            if after:tail+='        out[i * 4 + 2] = '+dtype+'(suffix[0])\n        out[i * 4 + 3] = '+dtype+'(suffix[2])\n'
            if mode in('maximum','padded_maximum'):
                body='    shared[t + '+str(count-128)+'] = a[i]\n';reader='    v = shared[(t ^ 64) + '+str(count-128)+']\n'
            if mode=='nested':
                body=f'''    if t < 64:
        if t % 2 == 0:
            shared[t] = a[i] + delta
        else:
            shared[t] = a[i] - delta
    else:
        shared[t] = a[i]
'''
            elif mode=='if_else':
                body='    if t % 2 == 0:\n        shared[t] = a[i] + delta\n    else:\n        shared[t] = a[i] - delta\n'
            elif mode=='phases':
                reader='''    v = shared[t ^ 64]
    syncthreads()
    if t % 2 == 0:
        shared[t] = v + delta
    else:
        shared[t] = v - delta
    syncthreads()
    v = shared[t ^ 64]
'''
            elif mode=='legacy32':
                local_type='i32'if dtype=='i64'else'u32'
                body='    shared[t] = '+local_type+'(a[i])\n'
                reader='    v = '+dtype+'(shared[t ^ 64])\n'
            source=HEADER+f'''@kernel
def {name}(a: Array({dtype}), out: Array({dtype}), delta: {dtype}, n: i32):
{before}    shared = shared_array({dtype}, {count})
{after}    t = tid_x()
    i = ctaid_x() * 128 + t
{prefix_write}{suffix_write}{body}    syncthreads()
{reader}    if i < n:
        out[i * 4] = v
{tail}'''
            if mode=='legacy32':source=source.replace('shared = shared_array('+dtype+', 128)','shared = shared_array('+local_type+', 128)')
            shared_bytes=516 if mode=='legacy32'else 49152 if mode in('maximum','padded_maximum')else 1044 if mode=='mixed'else 1024
            yield dict(name=name,dtype=dtype,mode=mode,source=source,shared_bytes=shared_bytes)

def values(seed,lanes):
    rng=random.Random(seed)
    edges=[0,1,MASK,1<<63,(1<<63)-1,1<<32,0xffffffff,0x8000000000000001,0x123456789abcdef0,0xfedcba9876543210]
    return[edges[i%len(edges)]if i<len(edges)*2 else rng.getrandbits(64)for i in range(lanes)]

def expected(data,case,n,delta):
    result=[SENTINEL]*(len(data)*4)
    for i in range(n):
        base=i//128*128;t=i%128;peer=base+(t^64)
        value=data[base+127]if case['mode']=='literal'else data[peer]
        if case['mode']=='loop':value=(value+3*delta)&MASK
        if case['mode']=='nested'and (t^64)<64:value=(value+delta if t%2==0 else value-delta)&MASK
        if case['mode']=='if_else':value=(value+delta if t%2==0 else value-delta)&MASK
        if case['mode']=='phases':value=(data[i]+delta if t%2==0 else data[i]-delta)&MASK
        if case['mode']=='legacy32':
            value&=0xffffffff
            if case['dtype']=='i64'and value&(1<<31):value=(value-(1<<32))&MASK
        result[i*4]=value
        if case['mode']in('mixed','padded_maximum','legacy32'):result[i*4+1]=0x12345678
        if case['mode']=='mixed':result[i*4+2]=0x76543210;result[i*4+3]=(-2147483648)&MASK
    return struct.pack('<'+str(len(result))+'Q',*result)

def delta(case,seed):return(-0x123456789abcdef if case['dtype']=='i64'else 0xfedcba9876543210)+(seed&1)

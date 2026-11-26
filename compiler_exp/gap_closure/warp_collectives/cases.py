"""Independent lane-index semantics, register/literal operands and reconvergence."""
HEADER='.version 8.7\n.target sm_80\n.address_size 64\n'


def cases():
    for mode in ('idx','up','down','bfly'):
        for source_kind in ('literal','parameter','varying','aliased'):
            name=f'{mode}_{source_kind}'
            b='33'if source_kind=='literal'else'%delta'if source_kind=='parameter'else'%value'if source_kind=='aliased'else'%b'
            destination='%value'if source_kind=='aliased'else'%result'
            yield dict(name=name,mode=mode,source_kind=source_kind,source=HEADER+f'''.visible .entry {name}(.param .u64 output,.param .u64 input,.param .u32 delta) {{
.reg .u64 %out,%input,%offset,%src,%dst;
.reg .u32 %t,%lane,%value,%result,%b,%delta; .reg .pred %branch;
ld.param.u64 %out,[output]; ld.param.u64 %input,[input]; ld.param.u32 %delta,[delta];
mov.u32 %t,%tid.x; and.b32 %lane,%t,31;
mul.wide.u32 %offset,%t,4; add.u64 %src,%input,%offset; ld.global.u32 %value,[%src];
setp.lt.u32 %branch,%lane,16; @%branch bra LOW;
add.u32 %value,%value,19; bra JOIN;
LOW: xor.b32 %value,%value,0x76543210;
JOIN: mad.lo.u32 %b,%lane,13,7;
shfl.sync.{mode}.b32 {destination},%value,{b},{0 if mode=='up'else 31},0xffffffff;
mul.wide.u32 %offset,%t,4; add.u64 %dst,%out,%offset; st.global.u32 [%dst],{destination};
ret; }}
''')


def expected(case,values,delta):
    transformed=[(value^0x76543210)if index%32<16 else(value+19)&0xffffffff for index,value in enumerate(values)]
    result=[]
    for index in range(len(values)):
        lane=index%32
        b=(33 if case['source_kind']=='literal'else delta if case['source_kind']=='parameter'else transformed[index]if case['source_kind']=='aliased'else lane*13+7)&31
        if case['mode']=='idx':source=b
        elif case['mode']=='up':source=lane-b if lane>=b else lane
        elif case['mode']=='down':source=lane+b if lane+b<32 else lane
        else:source=lane^b
        result.append(transformed[index-lane+source])
    return result

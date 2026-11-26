"""Wide/scalar sums and boolean fan-out expose avoidable allocation pressure."""
HEADER='.version 8.7\n.target sm_80\n.address_size 64\n'
SENTINEL=0x13579bdf


def source(name,count,limit,width=1,predicates=False,local=False,loop=False):
    lines=[HEADER+f'''.visible .entry {name}(.param .u64 output,.param .u64 input,.param .u32 n)
.maxnreg {limit}
{{
{'.local .align 16 .b8 preserve[32];'if local else''}
.reg .u64 %out,%input,%offset,%src,%dst;
.reg .u32 %t,%ct,%i,%n,%sum,%tmp,%r<{count}>,%iter;
.reg .pred %end,%even,%again,%p<{count}>;
ld.param.u64 %out,[output];ld.param.u64 %input,[input];ld.param.u32 %n,[n];
mov.u32 %t,%tid.x;mov.u32 %ct,%ctaid.x;mad.lo.u32 %i,%ct,128,%t;
setp.ge.u32 %end,%i,%n;@%end bra END;
mul.wide.u32 %offset,%i,{count*4};add.u64 %src,%input,%offset;
and.b32 %t,%i,1;setp.eq.u32 %even,%t,0;
mov.u32 %sum,0;mov.u32 %iter,0;
''']
    if loop:lines.append('LOOP:')
    for start in range(0,count,width):
        if width==1:lines.append(f'ld.global.u32 %r{start},[%src+{4*start}];')
        else:
            values=','.join(f'%r{i}'for i in range(start,start+width))
            lines.append(f'ld.global.v{width}.u32 {{{values}}},[%src+{4*start}];')
        if local and start==0:lines.append('st.local.u32 [preserve],%r0;')
    if predicates:
        for index in range(count):lines.append(f'and.b32 %tmp,%r{index},1;setp.ne.u32 %p{index},%tmp,0;')
        for index in range(count):lines.append(f'@{("!"if index%2 else"")}%p{index} add.u32 %sum,%sum,{index+1};')
    else:
        for index in range(count):
            lines.append(f'xor.b32 %tmp,%r{index},{index+1};@%even add.u32 %sum,%sum,%tmp;@!%even xor.b32 %sum,%sum,%tmp;')
    if loop:lines.append('add.u32 %iter,%iter,1;setp.lt.u32 %again,%iter,3;@%again bra LOOP;')
    if local:lines.append('ld.local.u32 %tmp,[preserve];xor.b32 %sum,%sum,%tmp;')
    lines+=['mul.wide.u32 %offset,%i,4;add.u64 %dst,%out,%offset;st.global.u32 [%dst],%sum;','END:ret;}']
    return '\n'.join(lines)+'\n'


def cases():
    for name,count,limit,width,predicates,local,loop in(
        ('scalar40',40,32,1,False,False,False),
        ('scalar96',96,32,1,False,False,False),
        ('pair96',96,32,2,False,False,False),
        ('quad96',96,32,4,False,False,False),
        ('pred24',24,32,1,True,False,False),
        ('local_loop48',48,32,1,False,True,True),
    ):
        yield dict(name=name,count=count,limit=limit,width=width,predicates=predicates,local=local,loop=loop,output_words=1,
                   source=source(name,count,limit,width,predicates,local,loop))


def expected(values,case,n,lanes):
    output=[SENTINEL]*lanes
    for lane in range(n):
        words=values[lane*case['count']:(lane+1)*case['count']];result=0
        for _ in range(3 if case['loop']else 1):
            for index,word in enumerate(words):
                if case['predicates']:
                    if bool(word&1)!=(index%2!=0):result=(result+index+1)&0xffffffff
                else:
                    value=word^(index+1)
                    result=(result+value)&0xffffffff if lane%2==0 else result^value
        if case['local']:result^=words[0]
        output[lane]=result
    return output

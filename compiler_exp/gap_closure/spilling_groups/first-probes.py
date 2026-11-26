"""Aligned vector groups, mutable predicates and their simultaneous pressure."""
HEADER='.version 8.7\n.target sm_80\n.address_size 64\n'
SENTINEL=0x13579bdf


def source(name,count,limit,width=0,predicates=0,local=False):
    output_words=count+4 if width else 1
    lines=[HEADER+f'''.visible .entry {name}(.param .u64 output,.param .u64 input,.param .u32 n)
.maxnreg {limit}
{{
{'.local .align 16 .b8 preserve[32];' if local else ''}
.reg .u64 %out,%input,%offset,%src,%dst;
.reg .u32 %t,%i,%n,%ct,%r<{count}>,%sum,%iter,%tmp;
.reg .pred %end,%even,%loop,%p<{max(1,predicates)}>;
ld.param.u64 %out,[output];ld.param.u64 %input,[input];ld.param.u32 %n,[n];
mov.u32 %t,%tid.x;mov.u32 %ct,%ctaid.x;mad.lo.u32 %i,%ct,128,%t;
setp.ge.u32 %end,%i,%n;@%end bra END;
mul.wide.u32 %offset,%i,{count*4};add.u64 %src,%input,%offset;
mul.wide.u32 %offset,%i,{output_words*4};add.u64 %dst,%out,%offset;
''']
    if width:
        for start in range(0,count,width):
            values=','.join(f'%r{i}'for i in range(start,start+width))
            lines.append(f'ld.global.v{width}.u32 {{{values}}},[%src+{4*start}];')
    for i in range(predicates):
        if not width:lines.append(f'ld.global.u32 %tmp,[%src+{4*i}];')
        lines.append(f'and.b32 %t,{"%r"+str(i) if width else "%tmp"},1;setp.ne.u32 %p{i},%t,0;')
    if local:lines.append('st.local.u32 [preserve],%r0;')
    lines+=['and.b32 %t,%i,1;setp.eq.u32 %even,%t,0;mov.u32 %sum,0;mov.u32 %iter,0;','LOOP:']
    if width:
        for i in range(count):lines.append(f'xor.b32 %sum,%sum,%r{i};')
    for i in range(predicates):
        lines.append(f'@%even not.pred %p{i},%p{i};@%p{i} add.u32 %sum,%sum,{i+1};')
    lines+=['add.u32 %iter,%iter,1;setp.lt.u32 %loop,%iter,3;@%loop bra LOOP;']
    if width:
        # Every tuple is read as a whole after the loop and also as individual
        # scalars inside it. Alternate lanes leave complete vector output intact.
        for start in range(0,count,width):
            values=','.join(f'%r{i}'for i in range(start,start+width))
            lines.append(f'@%even st.global.v{width}.u32 [%dst+{4*start}],{{{values}}};')
    if local:lines.append('ld.local.u32 %tmp,[preserve];xor.b32 %sum,%sum,%tmp;')
    lines+= [f'st.global.u32 [%dst+{4*count if width else 0}],%sum;','END:ret;}']
    return '\n'.join(lines)+'\n'


def cases():
    for name,count,limit,width,predicates,local in (
        ('group2_tight',48,16,2,0,False),
        ('group4_tight',48,16,4,0,False),
        ('group4_auto',260,252,4,0,False),
        ('predicates',24,32,0,24,False),
        ('groups_predicates_local',48,32,4,24,True),
    ):
        text=source(name,count,limit,width,predicates,local)
        if name.endswith('_auto'):text=text.replace('.maxnreg 252\n','')
        yield dict(name=name,count=count,limit=limit,width=width,predicates=predicates,local=local,
                   output_words=count+4 if width else 1,source=text)


def expected(values,case,n,lanes):
    output=[SENTINEL]*(lanes*case['output_words']);count=case['count']
    for lane in range(n):
        words=values[lane*count:(lane+1)*count];base=lane*case['output_words'];result=0
        flags=[bool(value&1)for value in words[:case['predicates']]]
        for _ in range(3):
            if case['width']:
                for value in words:result^=value
            for index in range(len(flags)):
                if lane%2==0:flags[index]=not flags[index]
                if flags[index]:result=(result+index+1)&0xffffffff
        if case['local']:result^=words[0]
        if case['width']:
            if lane%2==0:output[base:base+count]=words
            output[base+count]=result
        else:output[base]=result
    return output

"""Live independent loads retain scalar pressure through a guarded loop."""
HEADER='.version 8.7\n.target sm_80\n.address_size 64\n'


def source(count=40,limit=32,*,guarded=True):
    lines=[HEADER+f'''.visible .entry pressure(.param .u64 output,.param .u64 input,.param .u32 n)
.maxnreg {limit}
{{
.reg .u64 %out,%input,%offset,%src,%dst;
.reg .u32 %t,%i,%n,%ct,%r<{count}>,%sum,%iter;
.reg .pred %end,%even,%loop;
ld.param.u64 %out,[output];ld.param.u64 %input,[input];ld.param.u32 %n,[n];
mov.u32 %t,%tid.x;mov.u32 %ct,%ctaid.x;mad.lo.u32 %i,%ct,128,%t;
setp.ge.u32 %end,%i,%n;@%end bra END;
mul.wide.u32 %offset,%i,{count*4};add.u64 %src,%input,%offset;
''']
    for i in range(count):lines.append(f'ld.global.u32 %r{i},[%src+{i*4}];')
    lines+=['mov.u32 %sum,0;mov.u32 %iter,0;and.b32 %t,%i,1;setp.eq.u32 %even,%t,0;','LOOP:']
    for i in range(count):
        if guarded:lines.append(f'@%even add.u32 %r{i},%r{i},{i+1};')
        lines.append(f'xor.b32 %sum,%sum,%r{i};')
    lines+=['add.u32 %iter,%iter,1;setp.lt.u32 %loop,%iter,3;@%loop bra LOOP;',
            'mul.wide.u32 %offset,%i,4;add.u64 %dst,%out,%offset;st.global.u32 [%dst],%sum;',
            'END:ret;}']
    return '\n'.join(lines)+'\n'


def expected(values,count,n,lanes):
    output=[0x13579bdf]*lanes
    for lane in range(n):
        words=list(values[lane*count:(lane+1)*count]);result=0
        for _ in range(3):
            for i in range(count):
                if lane%2==0:words[i]=(words[i]+i+1)&0xffffffff
                result^=words[i]
        output[lane]=result
    return output



def cases():
    yield dict(name='pressure_tight',count=40,limit=16,source=source(40,16).replace('pressure(','pressure_tight('))

"""Bounded two-CTA epoch publication and CTA-local shared atomic probes."""
HEADER='.version 8.7\n.target sm_80\n.address_size 64\n'
EPOCHS=128
WORDS=16
SPIN_LIMIT=1000000


def publication_source(name='publication',*,release='release',acquire='acquire'):
    lines=[HEADER+f'''.visible .entry {name}(.param .u64 output,.param .u64 payload,.param .u64 flag,.param .u64 ack) {{
.reg .u64 %out,%payload,%flag,%ack;
.reg .b32 %tid,%cta,%epoch,%spin,%old,%value,%expected,%bad,%warm;
.reg .pred %p,%timeout,%wrong;
ld.param.u64 %out,[output]; ld.param.u64 %payload,[payload]; ld.param.u64 %flag,[flag]; ld.param.u64 %ack,[ack];
mov.u32 %tid,%tid.x; setp.ne.u32 %p,%tid,0; @%p bra END;
mov.u32 %cta,%ctaid.x; mov.u32 %epoch,1; mov.u32 %bad,0;
setp.ne.u32 %p,%cta,0; @%p bra CONSUMER;
PRODUCER:
mul.lo.u32 %value,%epoch,0x1f123bb5;''']
    for j in range(WORDS):lines += [f'add.u32 %expected,%value,{j}; st.global.u32 [%payload+{4*j}],%expected;']
    lines += [f'atom.{release}.gpu.global.exch.b32 %old,[%flag],%epoch; mov.u32 %spin,0;',
              'WAIT_ACK:',f'atom.{acquire}.gpu.global.add.u32 %old,[%ack],0;',
              'setp.eq.u32 %p,%old,%epoch; @%p bra NEXT_PRODUCER;',
              f'add.u32 %spin,%spin,1; setp.lt.u32 %p,%spin,{SPIN_LIMIT}; @%p bra WAIT_ACK;',
              'st.global.u32 [%out],0xbad00001; bra END;',
              'NEXT_PRODUCER:',f'add.u32 %epoch,%epoch,1; setp.le.u32 %p,%epoch,{EPOCHS}; @%p bra PRODUCER;',
              'st.global.u32 [%out],0; bra END;',
              'CONSUMER:','mov.u32 %warm,0;']
    # Keep warm-up reads observable, forcing pre-acquire payload accesses to
    # remain present even though their values are not part of the correctness gate.
    for j in range(WORDS):lines += [f'ld.global.u32 %value,[%payload+{4*j}]; xor.b32 %warm,%warm,%value;']
    lines += ['st.global.u32 [%out+8],%warm;', 'WAIT_FLAG:',
              'mov.u32 %spin,0;','POLL_FLAG:',f'atom.{acquire}.gpu.global.add.u32 %old,[%flag],0;',
              'setp.eq.u32 %p,%old,%epoch; @%p bra READ_PAYLOAD;',
              f'add.u32 %spin,%spin,1; setp.lt.u32 %p,%spin,{SPIN_LIMIT}; @%p bra POLL_FLAG;',
              'st.global.u32 [%out+4],0xbad00002; bra END;','READ_PAYLOAD:',
              'mul.lo.u32 %expected,%epoch,0x1f123bb5;']
    for j in range(WORDS):lines += [f'ld.global.u32 %value,[%payload+{4*j}]; add.u32 %old,%expected,{j}; setp.ne.u32 %wrong,%value,%old; @%wrong add.u32 %bad,%bad,1;']
    lines += [f'atom.{release}.gpu.global.exch.b32 %old,[%ack],%epoch;',
              f'add.u32 %epoch,%epoch,1; setp.le.u32 %p,%epoch,{EPOCHS}; @%p bra WAIT_FLAG;',
              'st.global.u32 [%out+4],%bad;','END: ret; }']
    return ('\n'.join(lines)+'\n').replace('%tid','%thread').replace('%thread.x','%tid.x')


def shared_source(name='shared_add',*,operation='add',explicit=False):
    form='atom.'+('relaxed.cta.shared.' if explicit else 'shared.')+('add.u32' if operation=='add' else 'exch.b32')
    text=HEADER+f'''.visible .entry {name}(.param .u64 output) {{
.shared .align 4 .b32 slot[17];
.reg .u64 %out,%off,%dst; .reg .u32 %tid,%cta,%i,%bin,%address,%old,%value;
.reg .pred %init,%take;
ld.param.u64 %out,[output]; mov.u32 %tid,%tid.x; mov.u32 %cta,%ctaid.x;
setp.lt.u32 %init,%tid,17; mul.lo.u32 %address,%tid,4; @%init st.shared.u32 [%address],0;
bar.sync 0; rem.u32 %bin,%tid,17; mul.lo.u32 %address,%bin,4;
mad.lo.u32 %i,%cta,128,%tid; add.u32 %value,%i,1; mov.u32 %old,0x13579bdf;
and.b32 %bin,%tid,1; setp.eq.u32 %take,%bin,0;
@%take {form} %old,[%address],{('1' if operation=='add' else '%value')};
mul.wide.u32 %off,%i,4; add.u64 %dst,%out,%off; st.global.u32 [%dst],%old;
bar.sync 0; setp.lt.u32 %init,%tid,17; @!%init bra END;
mul.lo.u32 %address,%tid,4; ld.shared.u32 %old,[%address];
mad.lo.u32 %i,%cta,17,%tid; mul.wide.u32 %off,%i,4;
add.u64 %dst,%out,%off; st.global.u32 [%dst+2048],%old;
END: ret; }}
'''
    return text.replace('%tid','%thread').replace('%thread.x','%tid.x')

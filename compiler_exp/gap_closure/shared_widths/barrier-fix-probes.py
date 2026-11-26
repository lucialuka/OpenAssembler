"""Shared subword/64-bit snapshots with initialized bytes and two CTA barriers."""
import struct

HEADER='.version 8.7\n.target sm_80\n.address_size 64\n'
SENTINEL=0x13579bdf


def source(name,dtype,width,symbol=False):
    wide=width==8
    write='''st.shared.'''+dtype+(' [arena+8],%wide;'if symbol else' [%addr+8],%wide;')if wide else'st.shared.'+dtype+(' [arena+6],%lo;'if symbol else' [%addr+6],%lo;')
    read='ld.shared.'+dtype+' '+('%loadedwide'if wide else'%loaded')+(','+'[arena+'+('8'if wide else'6')+'];'if symbol else',[%other+'+('8'if wide else'6')+'];')
    store='st.global.b64 [%dst],%loadedwide;'if wide else'st.global.u32 [%dst],%loaded;'
    # Each writer initializes all sixteen bytes. The guarded narrow overwrite
    # must preserve its neighbors, checked through complete 16-byte snapshots.
    if symbol:
        setup='mov.u32 %addr,arena;mov.u32 %other,arena;'
        writer='setp.eq.u32 %writer,%t,0;'
        init='@%writer st.shared.v4.u32 [arena],{%s0,%s1,%s2,%s3};'
        guard='%writer'
    else:
        setup='''mul.lo.u32 %addr,%t,16;xor.b32 %peer,%t,64;mul.lo.u32 %other,%peer,16;
mov.u32 %base,arena;add.u32 %addr,%addr,%base;add.u32 %other,%other,%base;'''
        writer='and.b32 %tmp,%t,1;setp.eq.u32 %writer,%tmp,0;'
        init='st.shared.v4.u32 [%addr],{%s0,%s1,%s2,%s3};'
        guard='%writer'
    return HEADER+f'''.visible .entry {name}(.param .u64 output,.param .u64 input,.param .u32 n){{
.shared .align 16 .b8 arena[2048];
.reg .u64 %out,%input,%offset,%src,%dst;
.reg .b64 %wide,%loadedwide;
.reg .u32 %t,%ct,%i,%n,%peer,%addr,%other,%base,%tmp,%lo,%hi,%loaded,%iter,%j,%byte;
.reg .b32 %s0,%s1,%s2,%s3;
.reg .pred %writer,%active,%loop;
ld.param.u64 %out,[output];ld.param.u64 %input,[input];ld.param.u32 %n,[n];
mov.u32 %t,%tid.x;mov.u32 %ct,%ctaid.x;mad.lo.u32 %i,%ct,128,%t;
mul.wide.u32 %offset,%i,8;add.u64 %src,%input,%offset;
mul.wide.u32 %offset,%i,64;add.u64 %dst,%out,%offset;
ld.global.v2.u32 {{%lo,%hi}},[%src];mov.b64 %wide,{{%lo,%hi}};
{setup}
{writer}
setp.lt.u32 %active,%i,%n;
mov.u32 %s0,0x10203040;mov.u32 %s1,0x50607080;
mov.u32 %s2,0x90a0b0c0;mov.u32 %s3,0xd0e0f000;
{init}
bar.sync 0;
@{guard} {write}
bar.sync 0;
mov.u32 %loaded,0x2468ace0;mov.b64 %loadedwide,{{%lo,%hi}};
@%active {read}
@%active {store}
bar.sync 0;
// Retain every byte including neighbors around the narrow overwrite. All lanes
// reach barriers, even when output length masks their observable stores.
mov.u32 %j,0;
SNAP:ld.shared.u8 %byte,[%other];
@%active st.global.u8 [%dst+16],%byte;
add.u32 %other,%other,1;add.u64 %dst,%dst,1;add.u32 %j,%j,1;
setp.lt.u32 %loop,%j,16;@%loop bra SNAP;
ret;
}}
'''


def cases():
    for dtype,width in(('b8',1),('u8',1),('s8',1),('b16',2),('u16',2),('s16',2),('b64',8),('u64',8),('s64',8),('f64',8)):
        name='exchange_'+dtype
        yield dict(name=name,dtype=dtype,width=width,symbol=False,source=source(name,dtype,width))
    for dtype,width in(('u8',1),('s16',2),('u64',8)):
        name='symbol_'+dtype
        yield dict(name=name,dtype=dtype,width=width,symbol=True,source=source(name,dtype,width,True))
    for dtype,width in(('s16',2),('b64',8)):
        name='negative_'+dtype;text=source(name,dtype,width)
        text=text.replace('bar.sync 0;\n@%writer','bar.sync 0;\nadd.u32 %addr,%addr,16;add.u32 %other,%other,16;\n@%writer',1)
        old=8 if width==8 else 6;text=text.replace(f'[%addr+{old}]',f'[%addr+-{16-old}]').replace(f'[%other+{old}]',f'[%other+-{16-old}]')
        text=text.replace('mov.u32 %j,0;','sub.u32 %other,%other,16;mov.u32 %j,0;')
        yield dict(name=name,dtype=dtype,width=width,symbol=False,negative_offset=True,source=text)


def expected(data,case,n,lanes):
    original=struct.pack('<'+str(len(data))+'I',*data);output=bytearray(struct.pack('<I',SENTINEL)*(lanes*16))
    seed=bytes.fromhex('4030201080706050c0b0a09000f0e0d0')
    for lane in range(n):
        local=lane%128;cta=lane//128;peer=0 if case['symbol']else local^64
        writer=case['symbol']or peer%2==0;index=cta*128+peer
        snapshot=bytearray(seed);offset=8 if case['width']==8 else 6
        if writer:snapshot[offset:offset+case['width']]=original[index*8:index*8+case['width']]
        loaded=int.from_bytes(snapshot[offset:offset+case['width']],'little')
        if case['dtype'].startswith('s')and case['width']<4 and loaded>>(case['width']*8-1):loaded|=0xffffffff^((1<<(case['width']*8))-1)
        start=lane*64
        output[start:start+(8 if case['width']==8 else 4)]=loaded.to_bytes(8 if case['width']==8 else 4,'little')
        output[start+16:start+32]=snapshot
    return bytes(output)

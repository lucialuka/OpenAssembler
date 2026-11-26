"""Next G12 baseline: shared scalar narrow/64-bit instruction observations."""
import hashlib
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT))
from openassembler import build,oracle
from openassembler.build import split_listing,encoder
from openassembler.ptx.parse import parse_module


HEADER='.version 8.7\n.target sm_80\n.address_size 64\n'


def source(dtype):
    width=8 if dtype.endswith('64')else 4
    input_load='ld.global.b64 %wide,[%src];'if width==8 else'ld.global.u32 %v,[%src];'
    value='%wide'if width==8 else'%v';name='shared_'+dtype
    return HEADER+f'''.visible .entry {name}(.param .u64 output,.param .u64 input){{
.shared .align 16 .b8 arena[4096];
.reg .u64 %out,%input,%offset,%src,%dst;
.reg .u32 %t,%peer,%addr,%other,%base,%v,%loaded;
.reg .b64 %wide,%loadedwide;
ld.param.u64 %out,[output];ld.param.u64 %input,[input];
mov.u32 %t,%tid.x;mul.wide.u32 %offset,%t,{width};
add.u64 %src,%input,%offset;add.u64 %dst,%out,%offset;
{input_load}
mov.u32 %base,arena;mul.lo.u32 %addr,%t,16;xor.b32 %peer,%t,1;mul.lo.u32 %other,%peer,16;
add.u32 %addr,%addr,%base;add.u32 %other,%other,%base;
st.shared.{dtype} [%addr],{value};bar.sync 0;
ld.shared.{dtype} {('%loadedwide'if width==8 else'%loaded')},[%other];
st.global.{('b64'if width==8 else'u32')} [%dst],{('%loadedwide'if width==8 else'%loaded')};ret;
}}
'''


if __name__=='__main__':
    rows=[]
    for dtype in('u8','s8','u16','s16','u64','b64','f64'):
        name='shared_'+dtype;text=source(dtype);path=HERE/(name+'.ptx');path.write_text(text);binary=path.with_suffix('.cubin')
        result=oracle.run_tool('ptxas','-arch=sm_80','-o',oracle.tool_path(binary),oracle.tool_path(path),check=False)
        path.with_suffix('.tool.log').write_text(result.stdout+result.stderr)
        row=dict(dtype=dtype,toolkit_accepted=result.returncode==0,source_sha256=hashlib.sha256(text.encode()).hexdigest())
        if result.returncode==0:
            listing=oracle.disasm_text(binary);path.with_suffix('.sass').write_text(listing)
            row['cubin_sha256']=hashlib.sha256(binary.read_bytes()).hexdigest()
            row['shared_instructions']=[line.strip()for line in listing.splitlines()if' LDS'in line or' STS'in line]
        try:build.generate(parse_module(text).kernels[name],'sm_80','safe');row['oa_accepted']=True
        except Exception as error:row.update(oa_accepted=False,oa_error=type(error).__name__+': '+str(error))
        rows.append(row);print(row,flush=True)
    templates={}
    for instruction in('LDS.U8 R2, [R3]','LDS.S8 R2, [R3]','LDS.U16 R2, [R3]','LDS.S16 R2, [R3]','LDS.64 R2, [R3]',
                        'STS.U8 [R3], R2','STS.U16 [R3], R2','STS.64 [R3], R2'):
        try:templates[instruction]=dict(encoded=True,word=hex(encoder('sm_80').encode(instruction)))
        except Exception as error:templates[instruction]=dict(encoded=False,error=type(error).__name__+': '+str(error))
    report=dict(schema=1,scope='G12 shared-width toolkit/source-admission/encoder baseline; no implementation or independent GPU numerical gate',results=rows,template_probes=templates,
                ptxas=oracle.run_tool('ptxas','--version').stdout,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (HERE/'oracle-probe.json').write_text(json.dumps(report,indent=2)+'\n')

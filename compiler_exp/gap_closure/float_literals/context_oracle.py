"""Locate decimal operands allowed by actual PTX instructions and special bits."""
import hashlib
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT))
from openassembler import oracle


def fixture(name,body):
    return f'''.version 8.7
.target sm_80
.address_size 64
.visible .entry {name}(.param .u64 output,.param .u64 input){{
.reg .u64 %out,%input;
.reg .u32 %bits;
.reg .f32 %a,%b,%value;
.reg .pred %p;
ld.param.u64 %out,[output];ld.param.u64 %input,[input];
ld.global.f32 %a,[%input];mov.f32 %b,0f40000000;setp.lt.f32 %p,%a,%b;
{body}
ret;
}}
'''


def cases():
    store='st.global.f32 [%out],%value;'
    for op in('add.rn.f32','sub.rn.f32','mul.rn.f32','min.f32','max.f32','div.rn.f32'):
        for index,args in enumerate(('1.5,%a','%a,1.5')):
            yield op.replace('.','_')+f'_{index}',fixture(op.replace('.','_')+f'_{index}',f'{op} %value,{args};{store}')
    for index,args in enumerate(('1.5,%a,%b','%a,1.5,%b','%a,%b,1.5')):
        yield f'fma_{index}',fixture(f'fma_{index}',f'fma.rn.f32 %value,{args};{store}')
    for op in('neg.f32','abs.f32','sqrt.rn.f32','rcp.rn.f32'):
        name=op.replace('.','_');yield name,fixture(name,f'{op} %value,1.5;{store}')
    for index,args in enumerate(('1.5,%a','%a,1.5')):
        name=f'setp_{index}';yield name,fixture(name,f'setp.lt.f32 %p,{args};selp.f32 %value,%a,%b,%p;{store}')
        name=f'selp_{index}';yield name,fixture(name,f'selp.f32 %value,{args},%p;{store}')
    yield 'store_literal',fixture('store_literal','st.global.f32 [%out],1.5;')
    yield 'setp_bad_dest',fixture('setp_bad_dest','setp.lt.f32 1.5,%a,%b;')
    yield 'selp_bad_guard',fixture('selp_bad_guard',f'selp.f32 %value,%a,%b,1.5;{store}')
    yield 'packed_literal',fixture('packed_literal','mov.b64 %out,{1.5,%bits};st.global.u64 [%input],%out;')
    for bits in('7ff0000000000000','fff0000000000000','7ff8000000000000','fff8000000000000',
                 '7ff0000000000001','7ffabcdef1234567','0000000000000001','8000000000000000'):
        name='raw_'+bits;yield name,fixture(name,f'mov.f32 %value,0d{bits};{store}')


if __name__=='__main__':
    folder=HERE/'contexts';folder.mkdir(exist_ok=True);records=[]
    for name,source in cases():
        path=folder/(name+'.ptx');path.write_text(source);binary=folder/(name+'.cubin')
        result=oracle.run_tool('ptxas','-arch=sm_80','-o',oracle.tool_path(binary),oracle.tool_path(path),check=False)
        raw=result.stdout+result.stderr;(folder/(name+'.tool.log')).write_text(raw)
        row=dict(name=name,accepted=result.returncode==0,returncode=result.returncode,
                 source_sha256=hashlib.sha256(source.encode()).hexdigest(),tool_log_sha256=hashlib.sha256(raw.encode()).hexdigest())
        if row['accepted']:
            listing=oracle.disasm_text(binary);(folder/(name+'.sass')).write_text(listing)
            row.update(cubin_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),sass_sha256=hashlib.sha256(listing.encode()).hexdigest())
        records.append(row);print(row,flush=True)
    (HERE/'context-oracle.json').write_text(json.dumps(dict(schema=1,scope='Toolkit syntax/operand and raw64-to-f32 observations; no GPU numerical proof',
        results=records,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2)+'\n')

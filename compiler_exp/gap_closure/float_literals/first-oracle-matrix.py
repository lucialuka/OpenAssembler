"""Tool acceptance controls for decimal contexts and boundary spellings."""
import hashlib
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT))
from openassembler import oracle


def source(instruction,register_type='f32',store_type='f32'):
    return f'''.version 8.7
.target sm_80
.address_size 64
.visible .entry literal(.param .u64 output){{
.reg .u64 %out;
.reg .{register_type} %value;
ld.param.u64 %out,[output];
{instruction}
st.global.{store_type} [%out],%value;ret;
}}
'''


if __name__=='__main__':
    fixtures=[]
    for index,literal in enumerate(('0.0','-0.0','+1.5','.5','2.','1e-3','1E+3',
        '3.4028234663852886e38','3.4028235677973366e38','1.401298464324817e-45',
        '1e-5000','1e5000','1.0000000596046448','0d3ff8000000000000')):
        fixtures.append((f'f32_{index}',source(f'mov.f32 %value,{literal};'),literal))
    for name,instruction,register_type,store_type in(
        ('b32_decimal','mov.b32 %value,1.5;','b32','b32'),
        ('u32_decimal','mov.u32 %value,1.5;','u32','u32'),
        ('f64_decimal','mov.f64 %value,1.5;','f64','f64'),
        ('b64_decimal','mov.b64 %value,1.5;','b64','b64'),
        ('u64_decimal','mov.u64 %value,1.5;','u64','u64'),
        ('f32_integer','mov.f32 %value,1;','f32','f32'),
        ('f32_raw64','mov.f32 %value,0d3ff8000000000000;','f32','f32'),
    ):fixtures.append((name,source(instruction,register_type,store_type),instruction))
    records=[];folder=HERE/'oracle';folder.mkdir(exist_ok=True)
    for name,text,spelling in fixtures:
        ptx=folder/(name+'.ptx');ptx.write_text(text);cubin=folder/(name+'.cubin')
        record=dict(name=name,spelling=spelling,source_sha256=hashlib.sha256(ptx.read_bytes()).hexdigest())
        try:
            oracle.ptxas(ptx,cubin,arch='sm_80')
            sass=oracle.disasm_text(cubin);(folder/(name+'.sass')).write_text(sass)
            record.update(accepted=True,cubin_sha256=hashlib.sha256(cubin.read_bytes()).hexdigest(),sass_sha256=hashlib.sha256(sass.encode()).hexdigest())
        except Exception as error:
            diagnostic=str(error);(folder/(name+'.failure.txt')).write_text(diagnostic+'\n')
            record.update(accepted=False,error_type=type(error).__name__,diagnostic=diagnostic)
        records.append(record);print(record,flush=True)
    result=dict(schema=1,scope='Same-toolkit literal syntax/context acceptance and optimized disassembly; no numerical/emulator/GPU claim',results=records,
                oracle_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (HERE/'oracle-matrix.json').write_text(json.dumps(result,indent=2)+'\n')

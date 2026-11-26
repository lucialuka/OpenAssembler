"""Primary-spec and toolkit counterexample for decimal PTX floating literals."""
import hashlib
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT))
from openassembler import build,oracle
from openassembler.ptx.parse import parse_module,parse_operand

SOURCE='''.version 8.7
.target sm_80
.address_size 64
.visible .entry decimal(.param .u64 output,.param .u64 input){
.reg .u64 %out,%input;
.reg .f32 %value;
ld.param.u64 %out,[output];ld.param.u64 %input,[input];
ld.global.f32 %value,[%input];
fma.rn.f32 %value,%value,1.5,0.0;
st.global.f32 [%out],%value;ret;
}
'''


if __name__=='__main__':
    source=HERE/'decimal.ptx';source.write_text(SOURCE)
    reference=HERE/'decimal-reference.cubin'
    oracle.ptxas(source,reference,arch='sm_80')
    (HERE/'decimal-reference.sass').write_text(oracle.disasm_text(reference))
    try:build.generate(parse_module(SOURCE).kernels['decimal'],'sm_80','safe')
    except Exception as error:failure={'type':type(error).__name__,'message':str(error)}
    else:raise AssertionError('baseline no longer reproduces')
    result=dict(schema=1,passed=True,gap='G11',scope='Toolkit accepts valid decimal operands which OA parses as variable names; code-generation counterexample, no GPU numerical claim',
                source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),reference_sha256=hashlib.sha256(reference.read_bytes()).hexdigest(),
                parsed_operands={text:type(parse_operand(text)).__name__ for text in('0.0','1.5','1e-3')},oa_failure=failure,
                compiler_hashes={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest()for path in(ROOT/'openassembler/ptx/parse.py',ROOT/'openassembler/backend/ra.py')},
                primary_spec={'url':'https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-constants',
                              'quote':'Each 64-bit floating-point constant is converted to the appropriate floating-point size based on the data or instruction type at its use.',
                              'implication':'Decimal literals have binary64 representation before use-site conversion; parsing directly as binary32 would lose the specified two-stage rounding.'})
    (HERE/'baseline.json').write_text(json.dumps(result,indent=2)+'\n');print(result)

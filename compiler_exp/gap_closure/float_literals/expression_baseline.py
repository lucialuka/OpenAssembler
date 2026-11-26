"""Next G11 baseline: typed constant expressions still lack parser evaluation."""
import hashlib
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
sys.path[:0]=[str(ROOT),str(HERE)]
from oracle_matrix import source
from openassembler import build,oracle
from openassembler.ptx.parse import parse_module


if __name__=='__main__':
    expressions=('(-1.5)','1.5+0.5','(1.5+0.5)','(1.5+1)','(1.0/3.0)','(1.0/0.0)',
                 '(0.0/0.0)','((1.0e20+1.0)-1.0e20)','(0d3ff0000000000000+1.0)',
                 '(0f3f800000+1.0)','(1+2)','(1.5*2.0)','(1.5<2.0)','(1?1.5:2.5)')
    folder=HERE/'expressions';folder.mkdir(exist_ok=True);records=[]
    for index,expression in enumerate(expressions):
        text=source('mov.f32 %value,'+expression+';');path=folder/f'expr_{index}.ptx';path.write_text(text);binary=path.with_suffix('.cubin')
        result=oracle.run_tool('ptxas','-arch=sm_80','-o',oracle.tool_path(binary),oracle.tool_path(path),check=False)
        (folder/f'expr_{index}.tool.log').write_text(result.stdout+result.stderr)
        row=dict(name=f'expr_{index}',expression=expression,toolkit_accepted=result.returncode==0,source_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        if result.returncode==0:
            row['cubin_sha256']=hashlib.sha256(binary.read_bytes()).hexdigest()
            path.with_suffix('.sass').write_text(oracle.disasm_text(binary))
        try:build.generate(parse_module(text).kernels['literal'],'sm_80','safe');row['oa_accepted']=True
        except Exception as error:row.update(oa_accepted=False,oa_error=type(error).__name__+': '+str(error))
        records.append(row);print(row,flush=True)
    report=dict(schema=1,scope='Pending typed floating constant-expression gap: toolkit/production code-generation comparisons; no GPU numerical or implementation claim',
                parser_sha256=hashlib.sha256((ROOT/'openassembler/ptx/parse.py').read_bytes()).hexdigest(),
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),results=records,
                counterexamples=sum(row['toolkit_accepted']and not row['oa_accepted']for row in records))
    (HERE/'expression-baseline.json').write_text(json.dumps(report,indent=2)+'\n')

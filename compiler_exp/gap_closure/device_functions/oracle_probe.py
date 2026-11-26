"""Observe actual toolkit function syntax, lowered instructions and executable inventory."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;sys.path[:0]=[str(ROOT),str(HERE)]
from fixtures import expanded_cases as cases
from openassembler import oracle
from openassembler.container.elf import Elf

def sha(data):return hashlib.sha256(data).hexdigest()
if __name__=='__main__':
    base=HERE/'oracle';base.mkdir(exist_ok=True);rows=[]
    for case in cases():
        path=base/(case['name']+'.ptx');path.write_text(case['source']);binary=path.with_suffix('.cubin')
        result=oracle.run_tool('ptxas','-arch=sm_80','-o',oracle.tool_path(binary),oracle.tool_path(path),check=False);path.with_suffix('.tool.log').write_text(result.stdout+result.stderr)
        row=dict(name=case['name'],accepted=result.returncode==0,source_sha256=sha(path.read_bytes()))
        if result.returncode==0:
            path.with_suffix('.sass').write_text(oracle.disasm_text(binary));row['cubin_sha256']=sha(binary.read_bytes());elf=Elf.parse(binary.read_bytes())
            row['texts']=[dict(name=s.name,size=s.size)for s in elf.sections if s.name.startswith('.text.')]
        rows.append(row);print(row,flush=True)
    result=dict(schema=1,results=rows,script_sha256=sha(Path(__file__).read_bytes()),toolkit=oracle.run_tool('ptxas','--version').stdout)
    (HERE/'oracle-probe.json').write_text(json.dumps(result,indent=2)+'\n')

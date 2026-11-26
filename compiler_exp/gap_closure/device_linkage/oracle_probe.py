"""Next G16 baseline: original multi-translation-unit CUDA device link commands."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT))
from openassembler import oracle
from openassembler.container.elf import Elf
from openassembler.ptx.parse import parse_module
from openassembler import build

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
if __name__=='__main__':
    rows=[]
    for name in('entry','helper'):
        source=HERE/(name+'.cu')
        for mode,suffix in(('-ptx','.ptx'),('-dc','.o')):
            output=source.with_suffix(suffix)
            args=['-arch=sm_80','-O3','-rdc=true',mode,'-o',oracle.tool_path(output),oracle.tool_path(source)]
            result=oracle.run_tool('nvcc',*args,check=False);log=HERE/(name+suffix+'.tool.log');log.write_text(result.stdout+result.stderr)
            row=dict(name=name,mode=mode,args=args,returncode=result.returncode,source_sha256=sha(source),log_sha256=sha(log))
            if result.returncode==0:row['artifact_sha256']=sha(output)
            rows.append(row);assert result.returncode==0,row
    binary=HERE/'linked.cubin';args=['-arch=sm_80','-dlink','-cubin','-o',oracle.tool_path(binary),oracle.tool_path(HERE/'entry.o'),oracle.tool_path(HERE/'helper.o')]
    result=oracle.run_tool('nvcc',*args,check=False);(HERE/'link.tool.log').write_text(result.stdout+result.stderr)
    link=dict(args=args,returncode=result.returncode,log_sha256=sha(HERE/'link.tool.log'))
    if result.returncode==0:
        link['cubin_sha256']=sha(binary);elf=Elf.parse(binary.read_bytes());link['texts']=[dict(name=s.name,size=s.size)for s in elf.sections if s.name.startswith('.text.')]
        (HERE/'linked.sass').write_text(oracle.disasm_text(binary))
    source=(HERE/'entry.ptx').read_text()
    try:module=parse_module(source);build.generate(next(iter(module.kernels.values())),'sm_80','safe');admission=dict(accepted=True)
    except Exception as error:admission=dict(accepted=False,error=type(error).__name__+': '+str(error))
    report=dict(schema=1,scope='Actual original nvcc separate translation units/relocatable device objects/device linking and unresolved OA baseline; no implementation or independent hardware execution',commands=rows,link=link,oa=admission,compiler_inputs={name:sha(ROOT/name)for name in('openassembler/ptx/parse.py','openassembler/ptx/inlining.py','openassembler/build.py')},script_sha256=sha(Path(__file__)),toolkit=oracle.run_tool('nvcc','--version').stdout)
    (HERE/'oracle-probe.json').write_text(json.dumps(report,indent=2)+'\n');print(link,admission)

"""Actual independent ptxas separate-module linking versus staged source resolver."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4];HERE=Path(__file__).resolve().parent;sys.path[:0]=[str(ROOT),str(HERE)]
from candidate import install
install()
from fixtures import cases
from openassembler import oracle
from openassembler.ptx.link import resolve
from openassembler.container.elf import Elf

def sha(data):return hashlib.sha256(data).hexdigest()
if __name__=='__main__':
 base=HERE/'oracle';base.mkdir(exist_ok=True);records=[]
 for case in cases():
  folder=base/case['name'];folder.mkdir(exist_ok=True);objects=[];commands=[]
  for i,source in enumerate([case['root'],*case['libraries']]):
   path=folder/f'unit{i}.ptx';path.write_text(source);obj=folder/f'unit{i}.o';args=['-arch=sm_80','-c','-o',oracle.tool_path(obj),oracle.tool_path(path)]
   result=oracle.run_tool('ptxas',*args,check=False);(folder/f'unit{i}.tool.log').write_text(result.stdout+result.stderr);commands.append(dict(tool='ptxas',args=args,returncode=result.returncode))
   if result.returncode==0:objects.append(obj)
  binary=folder/'original-linked.cubin';args=['-arch=sm_80','-o',oracle.tool_path(binary),*[oracle.tool_path(p)for p in objects]]
  linked=oracle.run_tool('nvlink',*args,check=False);(folder/'link.tool.log').write_text(linked.stdout+linked.stderr);commands.append(dict(tool='nvlink',args=args,returncode=linked.returncode))
  merged,contract=resolve([('entry',case['root'])]+[(f'library{i}',text)for i,text in enumerate(case['libraries'])]);path=folder/'merged.ptx';path.write_text(merged);control=folder/'merged.cubin'
  result=oracle.run_tool('ptxas','-arch=sm_80','-o',oracle.tool_path(control),oracle.tool_path(path),check=False);(folder/'merged.tool.log').write_text(result.stdout+result.stderr)
  row=dict(name=case['name'],original_objects_accepted=len(objects)==1+len(case['libraries']),original_link_accepted=linked.returncode==0,merged_accepted=result.returncode==0,commands=commands,contract=contract)
  if linked.returncode==0:
   (folder/'original-linked.sass').write_text(oracle.disasm_text(binary));row['original_cubin_sha256']=sha(binary.read_bytes());row['original_texts']=[s.name for s in Elf.parse(binary.read_bytes()).sections if s.name.startswith('.text.')]
  if result.returncode==0:(folder/'merged.sass').write_text(oracle.disasm_text(control));row['merged_cubin_sha256']=sha(control.read_bytes())
  records.append(row);print({k:v for k,v in row.items()if k not in('commands','contract')},flush=True)
 (HERE/'oracle-probe.json').write_text(json.dumps(dict(schema=1,results=records,script_sha256=sha(Path(__file__).read_bytes()),linker_sha256=sha((HERE/'link.py').read_bytes())),indent=2)+'\n')

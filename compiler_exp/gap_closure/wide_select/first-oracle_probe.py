"""Actual toolkit acceptance and observed wide select instructions for next G11 cycle."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;sys.path[:0]=[str(ROOT),str(HERE)]
from probe import cases
from openassembler import oracle

def sha(data):return hashlib.sha256(data).hexdigest()
if __name__=='__main__':
 base=HERE/'oracle';base.mkdir(exist_ok=True);rows=[]
 for case in cases():
  name=case['dtype']+'_'+case['mode'];source=base/(name+'.ptx');source.write_text(case['source']);binary=source.with_suffix('.cubin')
  result=oracle.run_tool('ptxas','-arch=sm_80','-o',oracle.tool_path(binary),oracle.tool_path(source),check=False);source.with_suffix('.tool.log').write_text(result.stdout+result.stderr)
  row=dict(name=name,accepted=result.returncode==0,source_sha256=sha(source.read_bytes()))
  if result.returncode==0:
   listing=oracle.disasm_text(binary);source.with_suffix('.sass').write_text(listing);row['cubin_sha256']=sha(binary.read_bytes());row['selects']=[i.text for i in oracle.parse_listing(listing)if i.text.split()[0]=='SEL']
  rows.append(row)
 result=dict(schema=1,results=rows,script_sha256=sha(Path(__file__).read_bytes()),scope='Real toolkit acceptance/listings for next wide select hypothesis; no independent hardware execution or production promotion')
 (HERE/'oracle-probe.json').write_text(json.dumps(result,indent=2)+'\n');print('Toolkit accepted',sum(r['accepted']for r in rows),'of',len(rows))

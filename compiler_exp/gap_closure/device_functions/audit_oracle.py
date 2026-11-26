"""Read-only custody of direct-function toolkit source and executable observations."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;sys.path[:0]=[str(ROOT),str(HERE)]
from fixtures import expanded_cases
from openassembler.container.elf import Elf

def sha(data):return hashlib.sha256(data).hexdigest()
if __name__=='__main__':
    path=HERE/'oracle-probe.json';report=json.loads(path.read_text());fixtures={c['name']:c for c in expanded_cases()};assert len(report['results'])==len(fixtures)==15
    files={}
    for row in report['results']:
        name=row['name'];assert row['accepted']
        for suffix in('.ptx','.cubin','.sass','.tool.log'):
            p=HERE/'oracle'/(name+suffix);files[str(p.relative_to(HERE))]=sha(p.read_bytes())
        assert(HERE/'oracle'/(name+'.ptx')).read_text()==fixtures[name]['source']
        assert row['source_sha256']==files['oracle/'+name+'.ptx']and row['cubin_sha256']==files['oracle/'+name+'.cubin']
        elf=Elf.parse((HERE/'oracle'/(name+'.cubin')).read_bytes())
        assert row['texts']==[dict(name=s.name,size=s.size)for s in elf.sections if s.name.startswith('.text.')]
    assert report['script_sha256']==sha((HERE/'oracle_probe.py').read_bytes())
    result=dict(schema=1,passed=True,report_sha256=sha(path.read_bytes()),artifact_hashes=files,script_sha256=sha(Path(__file__).read_bytes()),scope='Fifteen exact original PTX sources/tool logs/cubins/listings and executable inventories; no new hardware or timing')
    (HERE/'oracle-audit.json').write_text(json.dumps(result,indent=2)+'\n');print('Passed',len(fixtures),'sources;',len(files),'oracle artifacts')

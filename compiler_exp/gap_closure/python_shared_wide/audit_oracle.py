"""Read-only custody of the original typed shared-layout toolkit observations."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT))
from openassembler.container.elf import Elf
from openassembler.backend.ra import Selector
from openassembler.ptx.parse import parse_module

def sha(data):return hashlib.sha256(data).hexdigest()
if __name__=='__main__':
    path=HERE/'oracle-probe.json';report=json.loads(path.read_text());assert len(report['results'])==12
    assert report['frontend_sha256']==sha((ROOT/'openassembler/pyfront/compile.py').read_bytes())
    assert report['script_sha256']==sha((HERE/'oracle_probe.py').read_bytes())
    files={}
    for row in report['results']:
        prefix=HERE/'oracle'/row['name']
        for suffix in('.py','.ptx','.cubin','.sass','.tool.log'):
            artifact=prefix.with_suffix(suffix);files[str(artifact.relative_to(HERE))]=sha(artifact.read_bytes())
        assert files['oracle/'+row['name']+'.py']==row['source_sha256']
        assert files['oracle/'+row['name']+'.ptx']==row['ptx_sha256']
        assert files['oracle/'+row['name']+'.cubin']==row['cubin_sha256']
        kernel=parse_module(prefix.with_suffix('.ptx').read_text()).kernels[row['name']];selector=Selector(kernel)
        assert selector.smem_bytes==row['oa_shared_bytes']and selector.sym_shared==row['oa_offsets']
        elf=Elf.parse(prefix.with_suffix('.cubin').read_bytes());section=elf.section('.nv.shared.'+row['name'])
        assert section.size==row['oa_shared_bytes']and section.align==8 and section.type==8
        assert row['donor_sections']==[dict(name=section.name,type=section.type,flags=section.flags,size=section.size,align=section.align,info=section.info,link=section.link)]
    result=dict(schema=1,passed=True,report_sha256=sha(path.read_bytes()),artifact_hashes=files,frontend_sha256=report['frontend_sha256'],
                script_sha256=sha(Path(__file__).read_bytes()),scope='Original twelve exact source/PTX/cubin/listing/log identities and declaration/section layout; local toolkit observations, no new hardware or timing')
    (HERE/'oracle-audit.json').write_text(json.dumps(result,indent=2)+'\n');print('Passed',len(report['results']),'oracle sources;',len(files),'raw artifact identities')

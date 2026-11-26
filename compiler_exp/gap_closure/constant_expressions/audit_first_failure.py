"""Preserve and independently diagnose the failed first A100 candidate."""
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import sys
import zipfile

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
sys.path[:0]=[str(ROOT),str(HERE),str(HERE.parent/'float_literals')]


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__=='__main__':
    report_path=HERE/'candidate-a100-20261123T130725Z.json';report=json.loads(report_path.read_text());assert not report['passed']and report['error']is None
    archive_path=HERE/report['archive']['path'];assert sha(archive_path)==report['archive']['sha256']
    assert sha(HERE/'first-hardware-candidate-parse.py')==report['source_hashes']['openassembler/ptx/parse.py']
    assert sha(HERE/'first-hardware-run-modal.py')==report['harness_sha256']
    for name,path in(('reference',HERE/'first-hardware-reference.py'),('first_probes',HERE/'first-hardware-probes.py')):
        spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    probes=sys.modules['first_probes'];fixtures={case['name']:case for case in probes.cases()};rows=[];guard=b'\xa5'*256;bytes_checked=0
    with zipfile.ZipFile(archive_path)as archive:
        for row in report['results']:
            name,compiler,level,n,seed=(row[key]for key in('name','compiler','level','n','seed'));case=fixtures[name]
            prefix=f'{level}/{compiler}/{name}/{n}/{seed}/';before=archive.read(prefix+'before-input.bin');after=archive.read(prefix+'input.bin');actual=archive.read(prefix+'output.bin')
            values=probes.values(seed,384);assert before==after==guard+struct.pack('<384I',*values)+guard
            expected=probes.expected(values,case,n,384);wanted=guard+struct.pack('<'+str(len(expected))+'I',*expected)+guard
            assert actual[:256]==actual[-256:]==guard
            matched=actual==wanted;assert matched==row['passed'];bytes_checked+=len(after)+len(actual)
            if not matched:
                diffs=[]
                for index in range(case['slots']):
                    word=struct.unpack_from('<Q',actual,256+index*8)[0]
                    bits=case['bits'][index]
                    if word!=bits:diffs.append(dict(slot=index,expression=case['expressions'][index],actual_bits=hex(word),reference_bits=hex(bits)))
                rows.append(dict(name=name,compiler=compiler,level=level,n=n,seed=seed,differences_in_first_lane=diffs))
    assert len(report['results'])==384 and len(rows)==48 and all(row['compiler']=='ptxas'for row in rows)
    result=dict(schema=1,passed=False,audit_completed=True,report_sha256=sha(report_path),archive_sha256=sha(archive_path),
                launches=384,failed_control_launches=48,oa_failed_launches=0,bytes_checked=bytes_checked,failures=rows,
                scope='Failed gate stays failed: local oracle/reference/OA agree, worker ptxas differs in remainder signedness and generated-NaN sign. Current admitted domain requires explicit remainder casts and rejects invalid NaN arithmetic; historical outputs/tolerances unchanged')
    (HERE/'first-hardware-failure-audit.json').write_text(json.dumps(result,indent=2)+'\n');print('Retained failed gate:',len(rows),'control failures;',bytes_checked,'complete bytes')

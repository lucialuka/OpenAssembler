"""Keep the failed shared64 optimized-level hardware gate as a counterexample."""
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import struct
import sys
import zipfile

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
sys.path[:0]=[str(ROOT),str(HERE)]


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__=='__main__':
    path=HERE/'candidate-a100-20261123T143647Z.json';report=json.loads(path.read_text());assert not report['passed']and report['error']is None
    archive=HERE/report['archive']['path'];assert sha(archive)==report['archive']['sha256']
    spec=importlib.util.spec_from_file_location('failed_shared_probes',HERE/'failed-execution-probes.py');probes=importlib.util.module_from_spec(spec);spec.loader.exec_module(probes)
    assert sha(HERE/'failed-execution-probes.py')==report['source_hashes']['compiler_exp/gap_closure/shared_widths/probes.py']
    assert sha(HERE/'failed-execution-run-modal.py')==report['harness_sha256']
    assert sha(HERE/'baseline-compile.py')==report['source_hashes']['openassembler/backend/compile.py']
    fixtures={case['name']:case for case in probes.cases()};rows=[];guard=b'\xa5'*256;checked=0
    with zipfile.ZipFile(archive)as z:
        for row in report['results']:
            name,level,compiler,n,seed=(row[key]for key in('name','level','compiler','n','seed'))
            prefix=f'{level}/{compiler}/{name}/{n}/{seed}/';before=z.read(prefix+'before-input.bin');after=z.read(prefix+'input.bin');out=z.read(prefix+'output.bin')
            rng=random.Random(seed);data=[rng.getrandbits(32)for _ in range(768)]
            assert before==after==guard+struct.pack('<768I',*data)+guard
            wanted=guard+probes.expected(data,fixtures[name],n,384)+guard
            assert out[:256]==out[-256:]==guard and(out==wanted)==row['passed']
            checked+=len(out)+len(after)
            if not row['passed']:
                assert compiler=='oa'and level in('mined','sched','opt')and fixtures[name]['width']==8
                # Neighbor snapshots complete after another barrier; this
                # distinguishes stale first reads from bad byte encodings.
                assert all(out[256+lane*64+16:256+lane*64+32]==wanted[256+lane*64+16:256+lane*64+32]for lane in range(n))
                rows.append(dict(name=name,level=level,n=n,seed=seed,complete_later_snapshots=True))
    assert len(report['results'])==960 and len(rows)==90
    result=dict(schema=1,passed=False,audit_completed=True,report_sha256=sha(path),archive_sha256=sha(archive),
                launches=960,failed_oa_launches=90,failed_ptxas_launches=0,bytes_checked=checked,failures=rows,
                scope='Failed optimized shared64 first-read gate stays failed; complete later snapshots identify visibility/issue dependency gap. Safe OA and ptxas controls pass; no tolerance, source or historical output changed')
    (HERE/'failed-execution-audit.json').write_text(json.dumps(result,indent=2)+'\n');print('Retained',len(rows),'hardware failures;',checked,'bytes')

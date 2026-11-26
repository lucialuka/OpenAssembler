"""Read-only resource/timing custody and complete matched output references."""
from pathlib import Path
import hashlib
import json
import math
import random
import statistics
import struct
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT),str(HERE)]
from probes import expected
from openassembler import build
from openassembler.container.cubin import regcount,text_words
from openassembler.ptx.parse import parse_module
from openassembler.native_ownership import validate


def sha(data):return hashlib.sha256(data).hexdigest()


if __name__=='__main__':
    path=HERE/'resources-20261123T100759Z.json';report=json.loads(path.read_text());assert report['passed']
    assert all(sha((ROOT/name).read_bytes())==digest for name,digest in report['source_hashes'].items()if name.startswith('openassembler/'))
    assert report['harness_sha256']==sha((HERE/'run_resources_modal.py').read_bytes())
    archive_path=path.parent/report['archive']['path'];assert sha(archive_path.read_bytes())==report['archive']['sha256']
    comparisons=[];samples={name:[]for name in report['samples_ms']}
    assert len(samples)==4 and len(report['records'])==52
    with zipfile.ZipFile(archive_path)as archive:
        manifest=json.loads(archive.read('manifest.json'));binary=archive.read('oa.cubin')
        validate(manifest['entries'][0],binary)
        source=archive.read('pressure.ptx');module=parse_module(source.decode())
        for name,kernel in module.kernels.items():
            _,_,words,_,registers=build.generate(kernel,'sm_80','opt')
            assert words==text_words(binary,name)and registers==regcount(binary,name)
            comparisons.append(dict(entry=name,registers=registers,passed=True))
        original=archive.read('input.bin');guard=b'\xa5'*256
        assert original[:256]==original[-256:]==guard
        values=struct.unpack('<'+str(384*260)+'I',original[256:-256])
        wanted=guard+struct.pack('<384I',*expected(values,260,259,384))+guard
        ordering=random.Random(17053);keys=list(samples)
        for repetition in range(13):
            names=list(keys);ordering.shuffle(names)
            rows=[row for row in report['records']if row['repetition']==repetition]
            assert [row['name']for row in rows]==names
            for row in rows:
                duration=row['cuda_event_ms'];name=row['name']
                assert math.isfinite(duration)and duration>0 and row['passed']
                raw=archive.read(f'outputs/{repetition}/{name.replace("/","-")}.bin')
                assert raw==wanted and sha(raw)==row['output_sha256']
                samples[name].append(duration)
        assert samples==report['samples_ms']
        assert {name:statistics.median(values)for name,values in samples.items()}==report['medians_ms']
        damaged=bytearray(wanted);damaged[256]^=1;assert damaged!=wanted
    result=dict(schema=1,passed=True,report=path.name,report_sha256=sha(path.read_bytes()),archive_sha256=report['archive']['sha256'],
                source_exact=True,output_comparisons=52,native_comparisons=comparisons,medians_ms=report['medians_ms'],
                scope='Read-only original CUDA-event sample/order and native/output custody; one constructed pressure workload, no new timing run or general policy optimum',
                auditor_sha256=sha(Path(__file__).read_bytes()))
    (HERE/'resource-audit.json').write_text(json.dumps(result,indent=2)+'\n');print(result['medians_ms'])

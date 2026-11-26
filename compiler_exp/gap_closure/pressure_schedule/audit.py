"""Read-only byte, executable, resource and paired timing custody audit."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import statistics
import struct
import sys
import types
import zipfile

ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT),str(HERE)]
from openassembler import build
from openassembler.container.cubin import regcount,text_words
from openassembler.backend.storage import local_frame_inventory
from openassembler.native_ownership import validate
from openassembler.ptx.parse import parse_module


def sha(data):return hashlib.sha256(data).hexdigest()


def install(path):
    name='openassembler.backend.spill';module=types.ModuleType(name);module.__file__=str(path)
    sys.modules[name]=module;exec(compile(path.read_bytes(),str(path),'exec'),module.__dict__)


def audit(path,candidate,heldout,frozen):
    if heldout:from heldout_probes import HEADER,cases,expected,values as fixture_values
    else:from hardware_probes import HEADER,cases,expected
    report=json.loads(path.read_text());assert report['passed']and report['error']is None
    final_hash=None
    if not frozen:
        repair=json.loads((HERE/'terminator-repair.json').read_text())
        old=(HERE/'hardware-spill.py').read_bytes();new=(HERE/'spill.py').read_bytes()
        assert sha(old)==repair['frozen_allocator_sha256']and sha(new)==repair['final_allocator_sha256']
        assert old.count(repair['change']['old'].encode())==1
        assert old.replace(repair['change']['old'].encode(),repair['change']['new'].encode())==new
        admission_path=HERE/'final-a100-20261123T112918Z.json';admission=json.loads(admission_path.read_text())
        assert admission['passed']and admission['source_hashes']['openassembler/backend/spill.py']==sha(new)
        final_hash=sha(admission_path.read_bytes())
    if heldout:
        freeze_record=json.loads((HERE/'heldout-freeze.json').read_text())
        assert all(sha((HERE/('hardware-spill.py'if name=='spill.py'else name)).read_bytes())==digest for name,digest in freeze_record['sources'].items())
        first=json.loads((HERE/'first-heldout-freeze.json').read_text())
        assert all(freeze_record['sources'][name]==first['sources'][name]for name in('pressure.py','spill.py'))
    archive_path=path.parent/report['archive']['path'];assert sha(archive_path.read_bytes())==report['archive']['sha256']
    harness=HERE/('run_heldout_modal.py'if heldout else'run_modal.py')
    assert sha(harness.read_bytes())==report['harness_sha256']
    excluded={'openassembler/kernels/sm80_gemm_dispatch.json','openassembler/kernels/sm80_rowwise_dispatch.json'}
    for name,digest in report['source_hashes'].items():
        if name in excluded:continue # Exact measured-policy continuity is a separate audited gate.
        if name in('openassembler/backend/pressure.py','openassembler/backend/spill.py')and candidate:
            source=HERE/('hardware-spill.py'if frozen and name.endswith('/spill.py')else Path(name).name)
        else:source=ROOT/name
        if name=='openassembler/backend/spill.py'and not frozen:
            assert digest==repair['frozen_allocator_sha256']and sha(source.read_bytes())==repair['final_allocator_sha256']
            continue
        assert source.is_file()and sha(source.read_bytes())==digest,name
    fixtures={case['name']:case for case in cases()};native=[];logical_bytes=0
    domain={(name,level,compiler,n,seed)for name in fixtures for level in('safe','mined','sched','opt')
            for compiler in('baseline','candidate','ptxas')for n in(0,1,259,384)for seed in(13052,13053)}
    assert len(report['results'])==len(domain)
    assert{tuple(row[k]for k in('name','level','compiler','n','seed'))for row in report['results']}==domain
    compiler_resources={};guard=b'\xa5'*256
    with zipfile.ZipFile(archive_path)as archive:
        source=HEADER+'\n'.join(case['source'][len(HEADER):]for case in fixtures.values())
        assert archive.read('pressure.ptx')==source.encode()
        module=parse_module(source)
        for compiler in('baseline','candidate'):
            allocation_path=HERE/'baseline-spill.py'if compiler=='baseline'else(HERE/('hardware-spill.py'if frozen else'spill.py')if candidate else ROOT/'openassembler/backend/spill.py')
            expected_allocator=repair['final_allocator_sha256']if compiler=='candidate'and not frozen else report['allocator_sha256'][compiler]
            assert sha(allocation_path.read_bytes())==expected_allocator
            if compiler=='candidate'and candidate:
                from candidate import install as install_candidate
                install_candidate(frozen=frozen)
            else:install(allocation_path)
            for level in('safe','mined','sched','opt'):
                binary=archive.read(f'{level}/{compiler}/native.cubin')
                manifest=json.loads(archive.read(f'{level}/{compiler}/manifest.json'));entry=manifest['entries'][0]
                validate(entry,binary)
                contracts={item['name']:item for item in entry['kernels']}
                for name,kernel in module.kernels.items():
                    resources={};_,_,words,_,registers=build.generate(kernel,'sm_80',level,resource_info=resources)
                    assert words==text_words(binary,name)and registers==regcount(binary,name),(compiler,level,name)
                    assert local_frame_inventory(binary)[name]==resources['local_bytes']==contracts[name].get('local_bytes',0)
                    assert contracts[name].get('register_allocation',{}).get('spill_slots',0)==resources['spill_slots']
                    compiler_resources[level,compiler,name]=resources
                    native.append(dict(name=name,level=level,compiler=compiler,registers=registers,resources=resources,passed=True))
        def output_check(case,n,artifacts,seed):
            nonlocal logical_bytes
            raw={key:archive.read('blobs/'+digest+'.bin')for key,digest in artifacts.items()}
            assert set(raw)=={'before','input','output'}
            assert all(sha(raw[key])==digest for key,digest in artifacts.items())
            rng=random.Random(seed)
            values=fixture_values(seed,case,384)if heldout else[rng.getrandbits(32)for _ in range(384*case['count'])]
            original=guard+struct.pack('<'+str(len(values))+'I',*values)+guard
            reference=expected(values,case,n,384)
            wanted=guard+struct.pack('<'+str(len(reference))+'I',*reference)+guard
            assert raw['before']==raw['input']==original and raw['output']==wanted
            logical_bytes+=len(raw['input'])+len(raw['output'])
        for row in report['results']:
            assert row['passed'];case=fixtures[row['name']]
            output_check(case,row['n'],row['artifacts'],row['seed'])
            binary=archive.read(f'{row["level"]}/{row["compiler"]}/native.cubin')
            assert sha(binary)==row['cubin_sha256']
            assert regcount(binary,row['name'])==row['resources']['num_regs']
            assert local_frame_inventory(binary)[row['name']]==row['frame']==row['resources']['local_bytes']
        names=[name+'/'+compiler for compiler in('baseline','candidate','ptxas')for name in fixtures]
        expected_order=[];rng=random.Random(17053)
        for repetition in range(21):
            order=list(names);rng.shuffle(order)
            expected_order.extend((repetition,name)for name in order)
        assert[(row['repetition'],row['name'])for row in report['timings']]==expected_order
        assert Counter((row['name'],row['repetition'])for row in report['warmups'])==Counter((name,repetition)for name in names for repetition in range(3))
        samples={name:[]for name in names}
        for row in report['warmups']+report['timings']:
            assert row['passed'];name,compiler=row['name'].split('/')
            output_check(fixtures[name],259,row['artifacts'],17052)
            binary=archive.read(f'opt/{compiler}/native.cubin');attrs=report['timing_resources'][row['name']]
            assert attrs['num_regs']==regcount(binary,name)and attrs['local_bytes']==local_frame_inventory(binary)[name]
            if 'cuda_event_ms'in row:
                assert row['cuda_event_ms']>0;samples[row['name']].append(row['cuda_event_ms'])
        assert samples==report['samples_ms']
        assert {name:statistics.median(values)for name,values in samples.items()}==report['medians_ms']
        resources=[]
        for name in fixtures:
            before=compiler_resources['opt','baseline',name];after=compiler_resources['opt','candidate',name]
            assert after['spill_slots']<=before['spill_slots']and after['local_bytes']<=before['local_bytes']
            resources.append(dict(name=name,before=before,after=after,
                                  candidate_over_baseline=report['medians_ms'][name+'/candidate']/report['medians_ms'][name+'/baseline']))
    return dict(schema=1,passed=True,report=str(path.relative_to(ROOT)),report_sha256=sha(path.read_bytes()),
                archive_sha256=report['archive']['sha256'],audited_backend='isolated candidate'if candidate else'production',
                final_admission_sha256=final_hash,final_allocator_sha256=repair['final_allocator_sha256']if not frozen else None,
                correctness_launches=len(domain),timed_launches=len(report['timings']),warmup_launches=len(report['warmups']),
                bytes_checked=logical_bytes,native_comparisons=native,resource_timing_comparisons=resources,
                scope='Complete deduplicated input/output blobs rechecked for every launch, native words/frames/ownership, exact sample order and medians; resident kernel intervals only',
                auditor_sha256=sha(Path(__file__).read_bytes()))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('report',type=Path);parser.add_argument('output',type=Path)
    parser.add_argument('--candidate',action='store_true');parser.add_argument('--heldout',action='store_true');parser.add_argument('--frozen',action='store_true');args=parser.parse_args()
    result=audit(args.report.resolve(),args.candidate,args.heldout,args.frozen);args.output.write_text(json.dumps(result,indent=2)+'\n')
    print('Audited',result['correctness_launches'],result['timed_launches'],result['bytes_checked'],len(result['native_comparisons']))

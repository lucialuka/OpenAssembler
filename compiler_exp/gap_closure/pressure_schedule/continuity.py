"""Reuse prior numerical observations only after exact current executable checks.

This records changed selector/admission identities explicitly. Original source-
exact auditors, numerical outputs and hardware reports are never rewritten.
"""
import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from openassembler import build
from openassembler.container.cubin import regcount,text_words
from openassembler.modelfront.kernels import canonical
from openassembler.ptx.parse import parse_module


def sha(data):return hashlib.sha256(data).hexdigest()


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('admission',type=Path);args=parser.parse_args()
    admission_path=args.admission.resolve()
    admission=json.loads(admission_path.read_text());assert admission['passed']
    backend={f'openassembler/backend/{name}.py':sha((ROOT/f'openassembler/backend/{name}.py').read_bytes())for name in('ra','contracts','compile','sched')}
    assert all(admission['source_hashes'][name]==digest for name,digest in backend.items())
    backend['openassembler/backend/storage.py']=sha((ROOT/'openassembler/backend/storage.py').read_bytes())
    backend['openassembler/backend/spill.py']=sha((ROOT/'openassembler/backend/spill.py').read_bytes())
    assert all(admission['source_hashes'][name]==digest for name,digest in backend.items())
    old_backend=dict(backend)
    old_backend['openassembler/backend/spill.py']=sha((HERE/'baseline-spill.py').read_bytes())
    validation_path=HERE.parent/'spilling_groups/validation.json'
    validation=json.loads(validation_path.read_text());assert validation['passed']
    changed={name for name,digest in validation['source_hashes'].items()if name.startswith('openassembler/')and sha((ROOT/name).read_bytes())!=digest}
    assert changed=={'openassembler/backend/spill.py','openassembler/kernels/sm80_gemm_dispatch.json','openassembler/kernels/sm80_rowwise_dispatch.json'},changed
    chain_path=HERE.parent/'spilling_groups/application-continuity.json'
    chain=json.loads(chain_path.read_text());assert chain['passed']
    assert chain['backend_hashes']==old_backend
    assert 'openassembler/backend/pressure.py'not in validation['source_hashes']
    backend['openassembler/backend/pressure.py']=sha((ROOT/'openassembler/backend/pressure.py').read_bytes())
    assert admission['source_hashes']['openassembler/backend/pressure.py']==backend['openassembler/backend/pressure.py']
    proofs=[HERE.parent/'local_storage/promotion-audit.json',HERE.parent/'warp_collectives/promotion-audit.json',
            HERE.parent/'atomic_ordering/promotion-audit-20261123.json',
            HERE.parent/'atomic_ordering/promotion-forms-audit-20261123.json',
            HERE.parent/'decode_specialization/promotion-audit.json',
            HERE.parent/'integration_20261123/checkpoint-final-audit.json',
            HERE.parent/'spilling/promotion-audit.json',HERE.parent/'spilling/promotion-tight-audit.json',
            HERE.parent/'spilling_groups/promotion-audit.json']
    records=[];files={};cache={}
    def checked(path):
        files[str(path.relative_to(ROOT))]=sha(path.read_bytes());return json.loads(path.read_text())
    def compare(source,binary,level,names,label):
        module=parse_module(source.decode());assert set(module.kernels)==set(names)
        for name,kernel in module.kernels.items():
            key=sha(canonical({'kernel':__import__('dataclasses').asdict(kernel),'level':level}))
            if key not in cache:
                _,_,words,_,registers=build.generate(kernel,'sm_80',level);cache[key]=(words,registers)
            words,registers=cache[key]
            assert words==text_words(binary,name)and registers==regcount(binary,name),(label,name,level)
            records.append(dict(source=label,entry=name,level=level,registers=registers,passed=True))
    for proof_path in proofs:
        proof=checked(proof_path);assert proof['passed']
        reports=proof['reports']if isinstance(proof.get('reports'),list)else[{'path':proof['report'],'sha256':proof['report_sha256']}] 
        for item in reports:
            report_path=ROOT/item['path'];report=checked(report_path);assert files[item['path']]==item['sha256']
            identities=report.get('compiler_hashes',report.get('source_hashes'))
            if not all(identities[name]==digest for name,digest in old_backend.items()):
                assert chain['retained_hashes'][item['path']]==files[item['path']]
                assert chain['retained_hashes'][str(proof_path.relative_to(ROOT))]==files[str(proof_path.relative_to(ROOT))]
            archive_info=report.get('archive',report.get('artifact_archive'))
            archive_path=report_path.parent/archive_info['path'];assert sha(archive_path.read_bytes())==archive_info['sha256']
            files[str(archive_path.relative_to(ROOT))]=archive_info['sha256']
            with zipfile.ZipFile(archive_path)as archive:
                names=archive.namelist()
                if any(name in names for name in('probes.ptx','fixtures.ptx','shuffle.ptx','local.ptx')):
                    source_name=next(name for name in('probes.ptx','fixtures.ptx','shuffle.ptx','local.ptx')if name in names)
                    source=archive.read(source_name);entries=list(parse_module(source.decode()).kernels)
                    for level in('safe','mined','sched','opt'):
                        if source_name in('probes.ptx','shuffle.ptx','local.ptx'):compare(source,archive.read(level+'/oa.cubin'),level,entries,item['path'])
                        else:
                            module=parse_module(source.decode())
                            for entry in entries:
                                binary=archive.read(f'{level}/{entry}/oa.cubin')
                                kernel=module.kernels[entry];generated=build.generate(kernel,'sm_80',level)
                                assert generated[2]==text_words(binary,entry)and generated[4]==regcount(binary,entry)
                                records.append(dict(source=item['path'],entry=entry,level=level,registers=generated[4],passed=True))
                else:
                    for name in names:
                        if not name.endswith('/native-ledger.json'):continue
                        prefix=name.removesuffix('native-ledger.json');ledger=json.loads(archive.read(name))
                        plan=json.loads(archive.read(prefix+'plan.json'));assert sha(canonical(plan))==ledger['plan_sha256']
                        source=archive.read(prefix+'sources/native.ptx');assert sha(source)==ledger['source_sha256']
                        binary=archive.read(prefix+ledger['cubin']);assert sha(binary)==ledger['cubin_sha256']
                        compare(source,binary,'opt',ledger['kernels'],item['path']+':'+prefix)
    result=dict(schema=1,passed=True,hardware='not run; exact executable continuity only',performance='not measured',
                scope='Prior source-exact numerical/ownership auditors plus complete current native words/registers; pressure-aware virtual scheduling explicitly bound to independent A100 correctness/resource/paired-kernel gates',
                validated_backend_report=str(admission_path.relative_to(ROOT)),validated_backend_report_sha256=sha(admission_path.read_bytes()),
                backend_hashes=backend,changed_backend_inputs=['openassembler/backend/spill.py','openassembler/backend/pressure.py'],
                added_production_inputs=['openassembler/backend/pressure.py'],
                baseline_hashes={'openassembler/backend/spill.py':sha((HERE/'baseline-spill.py').read_bytes())},
                previous_validation=str(validation_path.relative_to(ROOT)),previous_validation_sha256=sha(validation_path.read_bytes()),
                previous_application_chain=str(chain_path.relative_to(ROOT)),previous_application_chain_sha256=sha(chain_path.read_bytes()),
                changed_production_inputs=sorted(changed),
                numerical_proofs=[str(path.relative_to(ROOT))for path in proofs],retained_hashes=files,
                native_comparisons=records,native_comparison_count=len(records),auditor_sha256=sha(Path(__file__).read_bytes()))
    (HERE/'application-continuity.json').write_text(json.dumps(result,indent=2)+'\n')
    print(result['passed'],len(records))

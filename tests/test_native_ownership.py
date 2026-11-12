"""Public native compilation/replay must not load hidden vendor GPU programs."""
from __future__ import annotations
import base64
import json
from pathlib import Path
import pytest
from tests.native_fixture import save_native_manifest

from openassembler import project
from openassembler.runner import RunError,RunOptions,build_plan


def repository(tmp_path):
    root=tmp_path/'repo';root.mkdir()
    (root/'model.py').write_text('def create_model(): pass\n')
    (root/'openassembler.toml').write_text('[model]\nentrypoint="model.py"\ninput_shapes=[[17,8]]\n')
    return root


def test_legacy_worker_process_rejects_before_runtime_discovery(monkeypatch):
    from openassembler.torchfront import worker
    monkeypatch.setattr(worker,'runtime_info',lambda:pytest.fail('legacy CUDA runtime entered'))
    with pytest.raises(RuntimeError,match='disabled'):
        worker.process({'mode':'compile'})


def test_legacy_direct_execute_cannot_bypass_disabled_worker(monkeypatch,tmp_path):
    from openassembler.torchfront import worker
    monkeypatch.setattr(worker,'runtime_info',lambda:pytest.fail('legacy CUDA runtime entered'))
    with pytest.raises(RuntimeError,match='disabled'):
        worker.execute_program(tmp_path,{},tmp_path/'state',compiled=True,report_path=tmp_path/'report')


def test_explicit_model_output_cannot_select_another_repository(tmp_path):
    from openassembler.modelfront.frontend import ModelError,resolve_model_output
    source=repository(tmp_path);other=tmp_path/'other-output';other.mkdir()
    (other/'model.json').write_text(json.dumps({'source_root':'/another/repository'}))
    with pytest.raises(ModelError,match='another source'):
        resolve_model_output(source,other)


def test_worker_artifacts_validate_before_any_publication(monkeypatch,tmp_path):
    from openassembler.modelfront import frontend,runtime
    from openassembler.torchfront.archive import pack
    source=repository(tmp_path);out=tmp_path/'output';out.mkdir()
    existing=out/'user.txt';existing.write_text('preserve')
    fake=pack({'plan.json':b'{}','native-ledger.json':b'{}'})
    monkeypatch.setattr(frontend,'_request',lambda *a,**k:{'report':{'passed':True},'artifacts':base64.b64encode(fake).decode()})
    def reject(*a,**k):raise ValueError('native ownership rejected')
    monkeypatch.setattr(runtime,'validate_bundle',reject)
    with pytest.raises((ValueError,frontend.ModelError),match='native ownership rejected'):
        frontend.compile_model_repository(source,out,backend='local')
    assert existing.read_text()=='preserve'
    assert not (out/'plan.json').exists()
    assert not (out/'native-ledger.json').exists()


def manifest(tmp_path,*,owner='openassembler',extra_owner=None):
    source=tmp_path/'kernel.ptx';source.write_text('retained source')
    (tmp_path/'ours.cubin').write_bytes(b'owned native test fixture')
    # Deliberately omit the reference file: public replay must not require it.
    kernels=[{'name':'k','params':[['u64','out'],['u32','n']], 'compiled_by':owner}]
    if extra_owner:
        kernels.append({'name':'unselected_vendor','params':[],'compiled_by':extra_owner})
    save_native_manifest(tmp_path,tmp_path,'sm_80','opt',[{'source':str(source),'cubin':'ours.cubin',
                         'reference':'.build/reference/never-read.cubin','kernels':kernels}])
    return source


@pytest.mark.parametrize('explicit_spec',[False,True])
def test_public_runner_never_reads_reference_files(monkeypatch,tmp_path,explicit_spec):
    manifest(tmp_path)
    original=Path.read_bytes
    def guarded(path):
        if '.build' in path.parts or 'reference' in path.name:
            pytest.fail(f'public runner read reference artifact {path}')
        return original(path)
    monkeypatch.setattr(Path,'read_bytes',guarded)
    spec=None
    if explicit_spec:
        spec=tmp_path/'launch.json'
        spec.write_text(json.dumps({'buffers':{'out':{'dtype':'u32','count':16,'init':'zeros'}},
                                   'steps':[{'kernel':'k','grid':1,'block':16,
                                             'args':[{'buf':'out'},{'u32':16}]}], 'compare':['out']}))
    plan,_=build_plan(RunOptions(out_dir=tmp_path,n=16,reps=0,spec=spec))
    assert len(plan.jobs)==1 and plan.reference_comparison is False
    assert all(case.ref is None for case in plan.cases)
    assert all(set(job['modules'])=={'ours'} or list(job['modules'])==['ours.cubin'] for job in plan.jobs)


@pytest.mark.parametrize('owner,extra',[('ptxas-fallback',None),('torch-inductor-triton',None),
                                        ('openassembler','ptxas-fallback')])
def test_public_runner_rejects_entire_module_with_vendor_kernel(tmp_path,owner,extra):
    manifest(tmp_path,owner=owner,extra_owner=extra)
    with pytest.raises(RunError,match='vendor/fallback'):
        build_plan(RunOptions(out_dir=tmp_path,kernel='k',n=16,reps=0))


@pytest.mark.parametrize('change',['missing','binary','text','registers','params','unknown_function'])
def test_public_runner_enforces_native_bytes_inventory_and_abi(tmp_path,change):
    manifest(tmp_path)
    path=tmp_path/'manifest.json';saved=json.loads(path.read_text());entry=saved['entries'][0]
    if change=='missing':entry.pop('native_ownership')
    elif change=='binary':
        binary=tmp_path/'ours.cubin';raw=bytearray(binary.read_bytes());raw[-1]^=1;binary.write_bytes(raw)
    elif change=='text':entry['native_ownership']['kernels']['k']['text_sha256']='0'*64
    elif change=='registers':entry['native_ownership']['kernels']['k']['registers']=250
    elif change=='params':entry['kernels'][0]['params'][0][0]='f32'
    else:entry['native_ownership']['kernels']['unowned_vendor']={'text_sha256':'0'*64}
    path.write_text(json.dumps(saved))
    with pytest.raises(RunError,match='ownership|integrity|inventory'):
        build_plan(RunOptions(out_dir=tmp_path,n=16,reps=0))


@pytest.mark.oracle
def test_real_builder_records_all_native_words_and_rejects_replaced_cubin(tmp_path):
    from openassembler import oracle,build
    from openassembler.native_ownership import validate
    if not oracle.available('ptxas'):pytest.skip('needs CUDA toolkit')
    source=tmp_path/'source.ptx';source.write_text('''.version 8.7
.target sm_80
.address_size 64
.visible .entry k(.param .u64 out) {
 .reg .b32 %r<3>; .reg .b64 %d;
 ld.param.u64 %d,[out]; mov.u32 %r0,7; st.global.u32 [%d],%r0; ret;
}
''')
    out=tmp_path/'build';assert build.compile_paths(source,out,build.Options(no_sass=True)).rc==0
    entry=project.load_manifest(out)['entries'][0];binary=(out/entry['cubin']).read_bytes()
    assert validate(entry,binary)==binary
    assert set(entry['native_ownership']['kernels'])=={'k'}
    # A same-source ptxas binary is executable but does not have OA byte custody.
    (out/entry['cubin']).write_bytes((out/entry['reference']).read_bytes())
    with pytest.raises(RunError,match='integrity'):
        build_plan(RunOptions(out_dir=out,n=16,reps=0))


def test_module_with_extra_executable_text_is_rejected_even_with_updated_file_hash(tmp_path):
    from tests.native_fixture import attach_native_fixture
    import hashlib
    manifest(tmp_path)
    path=tmp_path/'manifest.json';saved=json.loads(path.read_text());entry=saved['entries'][0]
    copy={**entry,'kernels':entry['kernels']+[{'name':'hidden_vendor','params':[],'compiled_by':'openassembler'}]}
    attach_native_fixture(tmp_path,copy)
    entry['native_ownership']['cubin_sha256']=hashlib.sha256((tmp_path/entry['cubin']).read_bytes()).hexdigest()
    path.write_text(json.dumps(saved))
    with pytest.raises(RunError,match='inventory'):
        build_plan(RunOptions(out_dir=tmp_path,n=16,reps=0))

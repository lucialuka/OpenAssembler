"""Static package helper semantics, source diagnostics, dependency/ownership/transport."""
import copy,json,struct,tempfile
from dataclasses import asdict
from pathlib import Path
import pytest
from openassembler import build,oracle
from openassembler.pyfront import compile_file,compile_kernel,find_kernel,NotSupported
from openassembler.pyfront.modules import SourceGraph,contract_from_ptx,validate_sources
from openassembler.ptx.parse import parse_module
from openassembler.ptx.interp import Interp,Launch as PLaunch
from openassembler.emu.machine import Memory,Machine,Launch
from openassembler.emu.program import load_listing
from tests.backend_harness import listing

CASES=json.loads((Path(__file__).parent/'data/module_helper_fixtures.json').read_text())
BASE=CASES['configurations'][0]

def write(root,case=BASE):
    for name,text in case['files'].items():
        path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(text)
    return root/case['entry']

@pytest.mark.parametrize('case',CASES['configurations'],ids=lambda c:c['name'])
@pytest.mark.parametrize('name',['signed','unsigned','floating'])
@pytest.mark.parametrize('level',['safe','opt'])
def test_original_package_sources_complete_guarded_bits_and_single_entry(case,name,level,tmp_path):
    source=write(tmp_path,case);options={'source_root':tmp_path}if case['explicit']else{}
    text,names=compile_file(source,**options);graph=contract_from_ptx(text)
    assert names==['signed','unsigned','floating']and graph['entry']==case['entry']
    assert{r['path']for r in graph['files']}==set(case['files'])
    validate_sources(source,graph,tmp_path if case['explicit']else None)
    selected=find_kernel(source,name,**options);single=compile_kernel(selected);assert single==compile_kernel(selected)
    module=parse_module(text);code,labels,words,_,registers=build.generate(module.kernels[name],'sm_80',level)
    _,_,single_words,_,single_regs=build.generate(parse_module(single).kernels[name],'sm_80',level)
    assert words==single_words and registers==single_regs
    data=CASES['values'][name];a,b,wanted,initial=(bytes.fromhex(data[key])for key in('a','b','expected','initial'))
    ap,bp,out=0x100010000,0x100020000,0x100000000;guard=b'\xa5'*64
    for engine in('ptx','normal','poisoned'):
        memory=Memory();memory.write_bytes(ap-64,guard+a+guard);memory.write_bytes(bp-64,guard+b+guard);memory.write_bytes(out-64,guard+initial+guard)
        if engine=='ptx':Interp(module,name,PLaunch(block=(128,1,1),params={'a':ap,'b':bp,'out':out,'n':91}),memory).run()
        else:Machine(load_listing(listing(code,labels)),Launch(block=(128,1,1),params=struct.pack('<QQQI',ap,bp,out,91),poison=0xdeadbeef if engine=='poisoned'else None),memory).run()
        assert bytes(memory.read(out-64+i,1)for i in range(len(wanted)+128))==guard+wanted+guard
        for address,data in((ap,a),(bp,b)):assert bytes(memory.read(address-64+i,1)for i in range(len(data)+128))==guard+data+guard

@pytest.mark.parametrize('case',CASES['negative_controls'],ids=lambda c:c['name'])
def test_unsupported_import_type_effect_binding_and_call_domains_reject(case,tmp_path):
    source=write(tmp_path);path=tmp_path/case['file'];text=path.read_text();assert case['old']in text;path.write_text(text.replace(case['old'],case['new'],1))
    if case['name']=='cross_recursive':path.write_text('from .transforms import signed_step as back\n'+path.read_text())
    with pytest.raises(NotSupported,match=case['reason'])as error:compile_file(source)
    assert error.value.path and error.value.lineno
    if case['name']in('capture','memory','untyped','bad_annotation_alias','string_annotation'):
        assert Path(error.value.path).resolve()==path.resolve()


def test_imported_syntax_error_reports_original_helper_line(tmp_path):
    source=write(tmp_path);helper=tmp_path/'arithmetic/base.py';helper.write_text(helper.read_text().replace('return x * y','return (x *',1))
    with pytest.raises(NotSupported)as error:compile_file(source)
    assert Path(error.value.path)==helper and error.value.lineno==6


def test_refreshed_kernel_reference_and_dependency_mutation_are_visible(tmp_path):
    source=write(tmp_path);reference=find_kernel(source,'signed');before=compile_kernel(reference);helper=tmp_path/'arithmetic/base.py'
    helper.write_text(helper.read_text().replace('return x * y','return x + y',1));after=compile_kernel(reference)
    assert before!=after
    with pytest.raises(ValueError,match='dependency changed'):validate_sources(source,contract_from_ptx(before))


@pytest.mark.parametrize('kind',['file_symlink','directory_symlink','ambiguous','outside_root'])
def test_resolver_rejects_ambiguous_or_escaping_sources(kind,tmp_path):
    source=write(tmp_path);helper=tmp_path/'arithmetic/base.py'
    if kind=='file_symlink':
        target=tmp_path/'elsewhere.py';helper.rename(target);helper.symlink_to(target)
    elif kind=='directory_symlink':
        target=tmp_path/'elsewhere';(tmp_path/'arithmetic').rename(target);(tmp_path/'arithmetic').symlink_to(target,target_is_directory=True)
    elif kind=='ambiguous':(tmp_path/'arithmetic/base').mkdir()
    else:
        with pytest.raises(ValueError,match='inside'):compile_file(source,source_root=tmp_path/'arithmetic')
        return
    with pytest.raises(NotSupported):compile_file(source)


def test_unrelated_host_modules_and_package_top_level_are_not_executed(tmp_path):
    source=write(tmp_path);marker=tmp_path/'executed'
    helper=tmp_path/'arithmetic/base.py';helper.write_text('from pathlib import Path\nPath('+repr(str(marker))+').write_text("bad")\n'+helper.read_text())
    (tmp_path/'not_used.py').write_text('this is not valid Python !!!')
    compile_file(source);assert not marker.exists()


def test_reexport_cycles_and_package_attribute_mutations_reject(tmp_path):
    source=write(tmp_path);init=tmp_path/'arithmetic/__init__.py';init.write_text('from .aliases import signed_step\n')
    (tmp_path/'arithmetic/aliases.py').write_text('from . import signed_step\n')
    source.write_text(source.read_text().replace('from arithmetic.transforms import signed_step as repeated_product','from arithmetic import signed_step as repeated_product'))
    with pytest.raises(NotSupported,match='cyclic'):compile_file(source)
    source=write(tmp_path);init.write_text('import arithmetic.transforms as transforms\ntransforms.signed_step = 17\n')
    with pytest.raises(NotSupported,match='mutated'):compile_file(source)


def test_source_graph_file_and_byte_limits_fail_with_explicit_diagnostics(tmp_path):
    source=write(tmp_path);tree=parse_tree(source)
    from openassembler.pyfront import compile as c
    graph=SourceGraph(source,tree,c._Names(tree));graph.total_bytes=8*1024*1024
    with pytest.raises(NotSupported,match='8 MiB'):graph._load('arithmetic.base',tree.body[0])
    tree=parse_tree(source);graph=SourceGraph(source,tree,c._Names(tree));graph.files.update({tmp_path/f'ghost{i}.py':b''for i in range(64)})
    with pytest.raises(NotSupported,match='64 files'):graph._load('arithmetic.base',tree.body[0])


def parse_tree(path):
    import ast
    return ast.parse(path.read_text())


@pytest.mark.parametrize('mutate',[
    lambda g:g['files'][0].update(path='../escape.py'),
    lambda g:g['files'][0].update(sha256='bad'),
    lambda g:g.update(entry='missing.py'),
    lambda g:next(iter(g['helpers'].values())).update(kernels={'signed':0}),
])
def test_invalid_ptx_graph_metadata_rejects_before_ownership(tmp_path,mutate):
    source=write(tmp_path);text,_=compile_file(source);graph=contract_from_ptx(text);mutate(graph)
    from openassembler.pyfront.modules import PREFIX
    with pytest.raises(ValueError):contract_from_ptx(PREFIX+json.dumps(graph)+'\n')


@pytest.mark.oracle
def test_normal_builder_binds_graph_and_rejects_manifest_mutation(tmp_path):
    if not oracle.available('ptxas'):pytest.skip('needs CUDA toolkit')
    source=write(tmp_path);output=tmp_path/'out';assert build.compile_paths(source,output,build.Options(no_sass=True)).rc==0
    entry,=json.loads((output/'manifest.json').read_text())['entries'];binary=(output/entry['cubin']).read_bytes()
    from openassembler.native_ownership import validate
    validate(entry,binary);entry['kernels'][0]['python_source_graph']['files'][0]['sha256']='0'*64
    with pytest.raises(ValueError,match='inventory'):validate(entry,binary)


@pytest.mark.oracle
def test_changed_helper_during_backend_cannot_publish_a_new_artifact(tmp_path,monkeypatch):
    if not oracle.available('ptxas'):pytest.skip('needs CUDA toolkit')
    source=write(tmp_path);output=tmp_path/'out';original=build.generate;changed=False
    def generate(*args,**kwargs):
        nonlocal changed
        result=original(*args,**kwargs)
        if not changed:
            changed=True;helper=tmp_path/'arithmetic/base.py';helper.write_text(helper.read_text()+'\n# concurrent edit\n')
        return result
    monkeypatch.setattr(build,'generate',generate)
    result=build.compile_paths(source,output,build.Options(no_sass=True));assert result.rc==1 and not list(output.rglob('*.cubin'))
    assert'dependency changed'in str(result.errors)


@pytest.mark.oracle
def test_real_serialized_worker_keeps_ignored_helper_and_entry_package_paths(tmp_path,monkeypatch):
    if not oracle.available('ptxas'):pytest.skip('needs CUDA toolkit')
    import base64
    from openassembler.remote_build import _sources,process
    from openassembler.torchfront.archive import pack,unpack
    case=next(c for c in CASES['configurations']if c['name']=='package_entry');source=write(tmp_path,case)
    (tmp_path/'.gitignore').write_text('arithmetic/base.py\n');options=build.Options(no_sass=True)
    root,files,includes=_sources(source,tmp_path/'out',options);assert'source/arithmetic/base.py'in files
    oracle.SCRATCH.mkdir(parents=True,exist_ok=True);monkeypatch.setattr(tempfile,'tempdir',str(oracle.SCRATCH))
    result=process(dict(source=base64.b64encode(pack(files)).decode(),path='source/'+source.relative_to(root).as_posix(),options={**asdict(options),'include_dirs':includes},name=None))
    assert not result.get('error')and result['rc']==0,result
    output=tmp_path/'returned';output.mkdir();unpack(base64.b64decode(result['artifacts']),output)
    entry,=json.loads((output/'manifest.json').read_text())['entries'];assert entry['source']=='application/entry.py'
    from openassembler.native_ownership import validate
    validate(entry,(output/entry['cubin']).read_bytes())


def test_remote_directory_with_explicit_root_preserves_requested_target_scope(tmp_path,capsys):
    from openassembler.remote_build import compile_native
    case=next(c for c in CASES['configurations']if c['name']=='explicit_root');source=write(tmp_path,case)
    (tmp_path/'unrelated.py').write_text('from openassembler.pyfront import *\n@kernel\ndef unrelated(a:Array(i32)):\n    a[0]=17\n')
    status,_=compile_native(source.parent,tmp_path/'out',build.Options(python_source_root=tmp_path),backend='modal',dry_run=True)
    assert status==0 and 'native build: 1 source(s)'in capsys.readouterr().out

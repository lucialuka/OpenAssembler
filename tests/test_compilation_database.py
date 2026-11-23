"""Build command settings are source contracts, not host-shell instructions."""
import json
from pathlib import Path
import tempfile
import pytest
from openassembler.compilation_database import discover,load
from openassembler.project import ProjectError


def database(tmp_path,arguments=None,command=None):
    root=tmp_path.resolve();(root/'build').mkdir(exist_ok=True);(root/'include').mkdir(exist_ok=True)
    (root/'kernel.cu').write_text('OA_GLOBAL void selected() {}\n')
    row={'directory':str(root/'build'),'file':'../kernel.cu'}
    if arguments is not None:row['arguments']=arguments
    elif command is not None:row['command']=command
    else:row['arguments']=['nvcc','-DOA_ENABLE=1','-I','../include','-std=c++17','-arch=sm_80','-c','../kernel.cu','-o','kernel.o']
    path=root/'compile_commands.json';path.write_text(json.dumps([row]));return path,row


def test_flags_are_per_source_and_paths_follow_original_cwd(tmp_path):
    path,row=database(tmp_path);commands=load(path);command=commands[(tmp_path/'kernel.cu').resolve()]
    assert command.flags==('-DOA_ENABLE=1','-I','../include','-std=c++17')
    assert command.directory==(tmp_path/'build').resolve()
    assert command.record()['original_arguments']==row['arguments']
    assert discover(tmp_path)[1]==commands


def test_response_file_is_expanded_and_hashed_without_execution(tmp_path):
    path,_=database(tmp_path,arguments=['ccache','nvcc','@options.rsp','-c','../kernel.cu'])
    rsp=tmp_path/'build/options.rsp';rsp.write_text('-DVALUE=7 -I "../include" --use_fast_math')
    command=next(iter(load(path).values()))
    assert '-DVALUE=7' in command.flags and '--use_fast_math' in command.flags
    assert str(rsp.resolve()) in command.response_files
    rsp.write_text('@options.rsp')
    with pytest.raises(ProjectError,match='cyclic'):load(path)


@pytest.mark.parametrize('arguments,reason',[
 (['nvcc','-arch=sm_90','-c','../kernel.cu'],'SM80'),
 (['nvcc','-gencode','arch=compute_80,code=[sm_80,sm_90]','../kernel.cu'],'SM80'),
 (['nvcc','-dlink','../kernel.cu'],'unsupported'),
 (['nvcc','-c','elsewhere.cu'],'positional'),
 (['nvcc','-o'],'needs a value'),
 (['nvcc','-c'],'omit'),
 (['clang++','-x','cuda','../kernel.cu'],'requires supported nvcc'),
])
def test_unsupported_or_incomplete_commands_reject(tmp_path,arguments,reason):
    path,_=database(tmp_path,arguments=arguments)
    with pytest.raises(ProjectError,match=reason):load(path)


def test_plain_shell_form_and_ambiguous_build_variants(tmp_path):
    path,row=database(tmp_path,command='nvcc -DVALUE=3 -c ../kernel.cu -o out.o')
    assert next(iter(load(path).values())).flags==('-DVALUE=3',)
    row['command']='nvcc ../kernel.cu && touch owned'
    path.write_text(json.dumps([row]))
    with pytest.raises(ProjectError,match='plain invocation'):load(path)
    assert not (tmp_path/'owned').exists()
    path,row=database(tmp_path)
    second={**row,'arguments':[x.replace('ENABLE=1','ENABLE=2') for x in row['arguments']]}
    path.write_text(json.dumps([row,second]))
    with pytest.raises(ProjectError,match='ambiguous'):load(path)


def test_database_discovers_generated_ignored_translation_unit(tmp_path):
    import subprocess
    from openassembler.repository import inspect
    from openassembler.diagnostics import inspect_source
    path,row=database(tmp_path)
    (tmp_path/'.gitignore').write_text('build/\n')
    generated=tmp_path/'build/generated.cu';generated.write_text('OA_GLOBAL void configured() {}')
    row.update(file='generated.cu',arguments=['nvcc','-DOA_GLOBAL=__global__','-c','generated.cu'])
    path.write_text(json.dumps([row]))
    subprocess.run(['git','init','-q',str(tmp_path)],check=True)
    plan=inspect(tmp_path)
    assert any(c.path==generated.resolve() and c.language=='cuda' for c in plan.native)
    report=inspect_source(tmp_path)
    assert report['ready_for_capture_or_compilation']
    assert any(row['path']=='build/generated.cu' for row in report['kernel_candidates'])


def test_remote_bundle_relocates_per_unit_paths_and_keeps_origins(tmp_path):
    from openassembler.remote_build import _sources
    from openassembler.build import Options
    path,row=database(tmp_path)
    (tmp_path/'include/qualifier.h').write_text('#define OA_GLOBAL __global__\n')
    (tmp_path/'build/config.h').write_text('#define SCALE 3\n')
    row['arguments'][1:1]=['-include','config.h']
    path.write_text(json.dumps([row]))
    root,payload,_=_sources(tmp_path.resolve(),tmp_path/'output',Options())
    assert 'source/build/config.h' in payload
    assert 'source/include/qualifier.h' in payload
    assert 'source/.oa-build-origins.json' in payload
    with tempfile.TemporaryDirectory() as temporary:
        destination=Path(temporary).resolve()
        for name,raw in payload.items():
            target=destination/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
        command=next(iter(load(destination/'source/compile_commands.json').values()))
        assert command.directory==destination/'source/build'
        assert command.source==destination/'source/kernel.cu'
        assert 'config.h' in command.flags
        assert command.resolve_path('config.h')==destination/'source/build/config.h'
        assert command.record()['submitted_origin']['original_arguments']==row['arguments']


def test_real_nvcc_command_preserves_generated_source_and_macro_settings():
    from openassembler import build,oracle
    from openassembler.native_ownership import validate
    from openassembler.compilation_database import dependencies
    if not oracle.available('nvcc') or not oracle.available('ptxas'):pytest.skip('requires real CUDA tools')
    oracle.SCRATCH.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='compile-db-',dir=oracle.SCRATCH) as temporary:
        root=Path(temporary).resolve();(root/'build').mkdir();(root/'include').mkdir()
        (root/'include/qualifier.h').write_text('#define OA_GLOBAL __global__\n')
        (root/'build/config.h').write_text('#define BASE 5\n')
        source=root/'build/generated.cu'
        source.write_text('#include "qualifier.h"\nextern "C" OA_GLOBAL void configured(float *out,const float *x,int n) {\n'
                          'int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) out[i]=x[i]*SCALE+BASE; }\n')
        row={'directory':str(root/'build'),'file':'generated.cu',
             'arguments':['nvcc','-I','../include','-include','config.h','-DSCALE=3','-arch=sm_80','-c','generated.cu','-o','generated.o']}
        (root/'compile_commands.json').write_text(json.dumps([row]))
        command=next(iter(load(root/'compile_commands.json').values()))
        graph=dependencies(command)
        assert {str(source),str(root/'build/config.h'),str(root/'include/qualifier.h')}<=graph.keys()
        result=build.compile_paths(root,root/'output',build.Options(verify=True,no_sass=True))
        assert result.rc==0,[(s.error,[k.error for k in s.failed]) for s in result.built]
        entry=json.loads((root/'output/manifest.json').read_text())['entries'][0]
        contract=entry['build_contract']
        assert contract['dependencies']==graph
        assert contract['original_arguments']==row['arguments']
        assert 'release 12.8' in contract['toolchain']
        assert entry['source']=='build/generated.cu'
        validate(entry,(root/'output'/entry['cubin']).read_bytes())
        # The exact normalized command independently feeds nvcc; compare PTX
        # instruction bodies, not only a discovery or success assertion.
        reference=oracle.SCRATCH/'compile_db_control.ptx'
        try:
            original=list(row['arguments'][1:])
            original[original.index('-c')]='--ptx'
            original[original.index('-o')+1]=str(reference.resolve())
            oracle.run_tool('nvcc',*original,cwd=command.directory)
            from openassembler.ptx.parse import parse_module
            ours=parse_module((root/'output'/entry['ptx']).read_text()).kernels['configured']
            expected=parse_module(reference.read_text()).kernels['configured']
            assert [(i.op,i.mods,i.args) for i in ours.body]==[(i.op,i.mods,i.args) for i in expected.body]
        finally:reference.unlink(missing_ok=True)


def test_remote_includes_ignored_quoted_sibling_without_explicit_include(tmp_path):
    import subprocess
    from openassembler.remote_build import _sources
    from openassembler.build import Options
    path,row=database(tmp_path)
    source=tmp_path/'build/generated.cu';source.write_text('#include "generated.h"\nOA_GLOBAL void k(){}\n')
    (tmp_path/'build/generated.h').write_text('#define OA_GLOBAL __global__\n')
    (tmp_path/'.gitignore').write_text('build/\n')
    row.update(file='generated.cu',arguments=['nvcc','-c','generated.cu'])
    path.write_text(json.dumps([row]))
    subprocess.run(['git','init','-q',str(tmp_path)],check=True)
    _,payload,_=_sources(tmp_path.resolve(),tmp_path/'output',Options())
    assert payload['source/build/generated.h']==(tmp_path/'build/generated.h').read_bytes()


def test_nvcc_options_file_preserves_each_header_search_path(tmp_path):
    path,row=database(tmp_path,arguments=['nvcc','--options-file','settings.rsp','-c','../kernel.cu'])
    (tmp_path/'build/settings.rsp').write_text('-I../include,../build -DSCALE=3')
    command=next(iter(load(path).values()))
    assert command.flags==('-I','../include','-I','../build','-DSCALE=3')


def test_command_form_never_silently_skips_shell_variable_expansion(tmp_path):
    path,_=database(tmp_path,command='nvcc -I$CUDA_INCLUDE_PATH -DVALUE=$VALUE -c ../kernel.cu')
    with pytest.raises(ProjectError,match='plain invocation'):load(path)
    path,_=database(tmp_path,arguments=['nvcc','-DVALUE=$VALUE','-c','../kernel.cu'])
    assert '-DVALUE=$VALUE' in next(iter(load(path).values())).flags


def test_original_source_argument_spelling_is_retained_for_file_macros(tmp_path):
    path,row=database(tmp_path)
    command=next(iter(load(path).values()))
    assert str(command.source_argument)=='../kernel.cu'
    assert command.record()['source_argument']=='../kernel.cu'


def test_real_command_preserves_relative_file_macro_value():
    from openassembler import build,oracle
    if not oracle.available('nvcc') or not oracle.available('ptxas'):pytest.skip('requires CUDA tools')
    oracle.SCRATCH.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='compile-db-file-',dir=oracle.SCRATCH) as temporary:
        root=Path(temporary).resolve();(root/'build').mkdir()
        source=root/'kernel.cu'
        source.write_text('extern "C" __global__ void file_length(unsigned *out) {out[0]=sizeof(__FILE__);}\n')
        row={'directory':str(root/'build'),'file':'../kernel.cu','arguments':['nvcc','-c','../kernel.cu','-o','length.o']}
        (root/'compile_commands.json').write_text(json.dumps([row]))
        result=build.compile_paths(root,root/'output',build.Options(no_sass=True))
        assert result.rc==0
        from openassembler.ptx.parse import parse_module,ImmI
        entry=json.loads((root/'output/manifest.json').read_text())['entries'][0]
        kernel=parse_module((root/'output'/entry['ptx']).read_text()).kernels['file_length']
        assert any(isinstance(arg,ImmI) and arg.value==len('../kernel.cu')+1 for ins in kernel.body for arg in ins.args)


def test_relocation_maps_original_absolute_file_and_include_spellings(tmp_path):
    from openassembler.remote_build import _sources
    from openassembler.build import Options
    path,row=database(tmp_path)
    source=(tmp_path/'kernel.cu').resolve()
    row['file']=str(source)
    row['arguments']=['nvcc','-I',str((tmp_path/'include').resolve()),'-c',str(source)]
    path.write_text(json.dumps([row]))
    _,payload,_=_sources(tmp_path.resolve(),tmp_path/'output',Options())
    with tempfile.TemporaryDirectory() as temporary:
        root=Path(temporary).resolve()
        for name,data in payload.items():
            target=root/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
        command=next(iter(load(root/'source/compile_commands.json').values()))
        assert f'-Xcompiler=-ffile-prefix-map=../kernel.cu={source}' in command.flags
        assert f'-Xcompiler=-ffile-prefix-map=../include={(tmp_path/"include").resolve()}' in command.flags


def test_real_relocated_file_macros_equal_original_command():
    from openassembler import build,oracle
    from openassembler.remote_build import _sources
    if not oracle.available('nvcc') or not oracle.available('ptxas'):pytest.skip('requires CUDA tools')
    with tempfile.TemporaryDirectory(prefix='oa-file-origin-',dir=oracle.SCRATCH) as original, tempfile.TemporaryDirectory(prefix='oa-file-stage-',dir=oracle.SCRATCH) as relocated:
        origin=Path(original).resolve();stage=Path(relocated).resolve();path,row=database(origin)
        (origin/'include/path.h').write_text('static constexpr unsigned header_length=sizeof(__FILE__);\n')
        (origin/'kernel.cu').write_text('#include "path.h"\nextern "C" __global__ void path(unsigned*out){out[0]=header_length+sizeof(__FILE__);}\n')
        row['arguments']=['nvcc','-I',str(origin/'include'),'-c',str(origin/'kernel.cu')]
        path.write_text(json.dumps([row]))
        a=build.compile_paths(origin,origin/'output',build.Options(no_sass=True));assert a.rc==0
        _,payload,_=_sources(origin,origin/'output',build.Options())
        for name,data in payload.items():
            target=stage/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
        b=build.compile_paths(stage/'source',stage/'output',build.Options(no_sass=True));assert b.rc==0
        from openassembler.ptx.parse import parse_module
        def body(directory):
            entry=json.loads((directory/'manifest.json').read_text())['entries'][0]
            k=parse_module((directory/entry['ptx']).read_text()).kernels['path']
            return [(i.op,i.mods,i.args) for i in k.body]
        assert body(origin/'output')==body(stage/'output')


def test_source_argument_retains_dot_and_repeated_slash_spelling(tmp_path):
    path,row=database(tmp_path,arguments=['nvcc','-c','./../kernel.cu'])
    assert next(iter(load(path).values())).source_argument=='./../kernel.cu'
    path,row=database(tmp_path,arguments=['nvcc','-c','..//kernel.cu'])
    assert next(iter(load(path).values())).source_argument=='..//kernel.cu'


def test_overlapping_include_map_keeps_original_source_filename():
    from openassembler import build,oracle
    from openassembler.remote_build import _sources
    if not oracle.available('nvcc') or not oracle.available('ptxas'):pytest.skip('requires CUDA tools')
    with tempfile.TemporaryDirectory(prefix='oa-map-origin-',dir=oracle.SCRATCH) as a, tempfile.TemporaryDirectory(prefix='oa-map-stage-',dir=oracle.SCRATCH) as b:
        root=Path(a).resolve();stage=Path(b).resolve();(root/'build').mkdir();(root/'include').mkdir()
        source=root/'include/kernel.cu';source.write_text('extern "C" __global__ void path(unsigned*out){out[0]=sizeof(__FILE__);}\n')
        row={'directory':str(root/'build'),'file':'../include/kernel.cu',
            'arguments':['nvcc','-I',str(root/'include'),'-c','../include/kernel.cu']}
        (root/'compile_commands.json').write_text(json.dumps([row]))
        result=build.compile_paths(root,root/'output',build.Options(no_sass=True));assert result.rc==0
        _,payload,_=_sources(root,root/'output',build.Options())
        for name,data in payload.items():
            target=stage/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
        result=build.compile_paths(stage/'source',stage/'output',build.Options(no_sass=True));assert result.rc==0
        from openassembler.ptx.parse import parse_module,ImmI
        for out in (root/'output',stage/'output'):
            entry=json.loads((out/'manifest.json').read_text())['entries'][0]
            k=parse_module((out/entry['ptx']).read_text()).kernels['path']
            assert any(isinstance(arg,ImmI) and arg.value==len('../include/kernel.cu')+1 for ins in k.body for arg in ins.args)


def test_implicit_quoted_closure_does_not_package_unrelated_ignored_state(tmp_path):
    import subprocess
    from openassembler.remote_build import _sources
    from openassembler.build import Options
    path,row=database(tmp_path)
    (tmp_path/'.gitignore').write_text('build/\n.venv/\n')
    (tmp_path/'.venv').mkdir();(tmp_path/'.venv/unrelated.bin').write_bytes(b'not a dependency')
    (tmp_path/'build/unrelated.bin').write_bytes(b'not a dependency')
    (tmp_path/'kernel.cu').write_text('#include "build/nested.h"\nOA_GLOBAL void k(){}\n')
    (tmp_path/'build/nested.h').write_text('#include "../include/qualifier.h"\n')
    (tmp_path/'include/qualifier.h').write_text('#define OA_GLOBAL __global__\n')
    subprocess.run(['git','init','-q',str(tmp_path)],check=True)
    _,payload,_=_sources(tmp_path.resolve(),tmp_path/'output',Options())
    assert 'source/build/nested.h' in payload
    assert 'source/include/qualifier.h' in payload
    assert not any(name.endswith('unrelated.bin') for name in payload)


def test_duplicate_variants_with_different_source_spelling_reject(tmp_path):
    path,row=database(tmp_path)
    other={**row,'arguments':['nvcc','-DOA_ENABLE=1','-I','../include','-std=c++17','-arch=sm_80','-c',str((tmp_path/'kernel.cu').resolve()),'-o','kernel.o']}
    path.write_text(json.dumps([row,other]))
    with pytest.raises(ProjectError,match='ambiguous'):load(path)


def test_unchanged_relative_spelling_preserves_existing_user_prefix_map(tmp_path):
    from openassembler.remote_build import _sources
    from openassembler.build import Options
    path,row=database(tmp_path)
    row['arguments']=['nvcc','-Xcompiler=-ffile-prefix-map=../kernel.cu=SHORT','-c','../kernel.cu']
    path.write_text(json.dumps([row]))
    _,payload,_=_sources(tmp_path.resolve(),tmp_path/'output',Options())
    with tempfile.TemporaryDirectory() as temporary:
        root=Path(temporary).resolve()
        for name,data in payload.items():
            target=root/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
        command=next(iter(load(root/'source/compile_commands.json').values()))
        assert command.flags==('-Xcompiler=-ffile-prefix-map=../kernel.cu=SHORT',)

"""Rebind measured dispatch only after frozen-source and retained-binary continuity.

Never alters historical measurements, binaries, generated PTX hashes, policy
selection, or production identity functions. No GPU execution is performed.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from contextlib import contextmanager
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import types
import zipfile

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from openassembler import build, gemm_dispatch as gd, rowwise_dispatch as rd
from openassembler.backend.tensor import TensorConfig
from openassembler.container.cubin import regcount, text_words
from openassembler.gemm import source_for
from openassembler.kernels.gemm import GemmPlan, generate as gemm_source
from openassembler.kernels.reductions import ReductionPlan, generate as rowwise_source
from openassembler.ptx.parse import parse_module

HERE = Path(__file__).resolve().parent


def sha(data):
    return hashlib.sha256(data).hexdigest()


def git_file(commit, path):
    return subprocess.check_output(['git', 'show', f'{commit}:{path}'], cwd=ROOT)


def identity(files, commit=None, sections=None):
    digest = hashlib.sha256()
    for name, path in sorted(files.items()):
        data = git_file(commit, name) if commit else path.read_bytes()
        digest.update(name.encode() + b'\0' + data + b'\0')
    for name, data in sorted((sections or {}).items()):
        digest.update(name.encode() + b'\0' + data + b'\0')
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', help='Exact frozen Git commit; defaults to HEAD')
    parser.add_argument('--update', action='store_true')
    args = parser.parse_args()
    commit = subprocess.check_output(['git', 'rev-parse', args.baseline or 'HEAD'], cwd=ROOT, text=True).strip()
    tables = {kind: json.loads(module.TABLE.read_text()) for kind, module in [('gemm', gd), ('rowwise', rd)]}
    inputs = {'gemm': gd.compiler_inputs(), 'rowwise': rd.identity_inputs()}
    sections = rd._runtime_source_sections()
    # Scoped CLI/runner sections are reused only after proving those full files
    # equal their frozen versions; unrelated sections cannot mask any change here.
    for name in ('openassembler/cli.py', 'openassembler/runner.py'):
        assert (ROOT / name).read_bytes() == git_file(commit, name), name
    old_identity = {kind: identity(files, commit, sections if kind == 'rowwise' else None)
                    for kind, files in inputs.items()}
    current_identity = {'gemm': gd.compiler_identity(), 'rowwise': rd.compiler_identity()}
    for kind in tables:
        assert old_identity[kind] == json.loads(git_file(commit, str((gd.TABLE if kind == 'gemm' else rd.TABLE).relative_to(ROOT))))['compiler_identity']
        prior_audit = tables[kind].get('source_continuity', {}).get('audit')
        assert prior_audit, 'missing prior compiler binding'
        prior = json.loads((ROOT / prior_audit).read_text())
        assert tables[kind]['compiler_identity'] == prior['current_compiler_identity'][kind]
    changed = {kind: [name for name, path in files.items() if path.read_bytes() != git_file(commit, name)]
               for kind, files in inputs.items()}
    assert all(set(names) <= {'openassembler/build.py','openassembler/backend/ra.py','openassembler/backend/contracts.py'} for names in changed.values()), changed
    old_build = git_file(commit, 'openassembler/build.py')
    import openassembler.backend as backend_package
    @contextmanager
    def frozen_backend():
        names = ('contracts', 'ra', 'tensor')
        saved = {name:(sys.modules.get('openassembler.backend.'+name),getattr(backend_package,name,None)) for name in names}
        try:
            for name in names:
                full = 'openassembler.backend.'+name
                module = types.ModuleType(full)
                module.__file__ = str(ROOT/'openassembler/backend'/(name+'.py'))
                module.__package__ = 'openassembler.backend'
                sys.modules[full] = module
                setattr(backend_package,name,module)
                raw = git_file(commit,'openassembler/backend/'+name+'.py')
                exec(compile(raw,commit+':'+name+'.py','exec'),module.__dict__)
            yield
        finally:
            for name,(module,attribute) in saved.items():
                full = 'openassembler.backend.'+name
                if module is None:sys.modules.pop(full,None)
                else:sys.modules[full] = module
                if attribute is None:
                    if hasattr(backend_package,name):delattr(backend_package,name)
                else:setattr(backend_package,name,attribute)

    current_build = (ROOT / 'openassembler/build.py').read_bytes()
    def function_ast(data, name):
        return ast.dump(next(n for n in ast.parse(data).body if isinstance(n, ast.FunctionDef) and n.name == name), include_attributes=False)
    assert function_ast(old_build, 'generate') == function_ast(current_build, 'generate')
    frozen = types.ModuleType('openassembler._gap_frozen_build')
    frozen.__file__ = str(ROOT / 'openassembler/build.py')
    frozen.__package__ = 'openassembler'
    sys.modules[frozen.__name__] = frozen
    exec(compile(old_build, f'{commit}:openassembler/build.py', 'exec'), frozen.__dict__)
    proof = {'schema': 1, 'created_utc': datetime.now(timezone.utc).isoformat(),
             'python': platform.python_version(), 'baseline_commit': commit,
             'scope': 'Frozen HEAD and current generated PTX/native words/controls/registers equal retained selection and heldout executables for all 44 installed GEMM and five rowwise policies; no new hardware or timing claim',
             'old_compiler_identity': old_identity, 'current_compiler_identity': current_identity,
             'changed_identity_inputs': changed,
             'compiler_inputs_sha256': {name: sha(path.read_bytes()) for files in inputs.values() for name, path in files.items()},
             'scoped_runtime_sections_sha256': {name: sha(data) for name, data in sections.items()},
             'frozen_build_sha256': sha(old_build), 'current_build_sha256': sha(current_build),
             'generate_function_ast_identical': True,
             'frozen_backend_sources_loaded': ['openassembler/backend/ra.py','openassembler/backend/contracts.py','openassembler/backend/tensor.py'],
             'tables_before_sha256': {kind: sha((gd.TABLE if kind == 'gemm' else rd.TABLE).read_bytes()) for kind in tables},
             'checks': [], 'retained_files': {}, 'hardware': 'not_run', 'performance': 'not_measured'}
    def checked(relative, expected):
        data = (ROOT / relative).read_bytes()
        assert sha(data) == expected, relative
        proof['retained_files'][str(relative)] = {'sha256': expected, 'bytes': len(data)}
        return data
    def encode_check(source, kernel, config):
        with frozen_backend():
            frozen_config = backend_package.tensor.TensorConfig(**asdict(config)) if config is not None else None
            old = frozen.generate(parse_module(source).kernels[kernel], 'sm_80', 'opt', tensor_config=frozen_config)
        new = build.generate(parse_module(source).kernels[kernel], 'sm_80', 'opt', tensor_config=config)
        assert old == new, (kernel, 'frozen/current code, labels, words, exits or register mismatch')
        return new[2], new[4]
    for entry in tables['gemm']['entries']:
        for evidence in entry['evidence']:
            raw = json.loads(checked(evidence['raw_result'], evidence['raw_result_sha256']))
            candidate = raw['candidates'][entry['policy']]
            plan = GemmPlan(**candidate['plan'])
            assert gd.contract_key(plan) == entry['contract']
            assert sha(gemm_source(plan).encode()) == entry['ptx_body_sha256'] == candidate['ptx_body_sha256']
            relative = Path(evidence['raw_result']).parent / raw['artifact_archive']['path']
            checked(relative, raw['artifact_archive']['sha256'])
            with zipfile.ZipFile(ROOT / relative) as archive:
                source = archive.read(f"{entry['policy']}/gemm.ptx").decode()
                binary = archive.read(f"{entry['policy']}/native.cubin")
            assert source == source_for(plan)
            assert sha(binary) == candidate['artifacts']['native']['cubin_sha256']
            words, registers = encode_check(source, 'gemm', TensorConfig(pointer_mode=plan.pointer_mode, pointer_advance=plan.pointer_advance))
            assert words == text_words(binary, 'gemm') and registers == regcount(binary, 'gemm')
            proof['checks'].append({'kind': 'gemm', 'contract': entry['contract'], 'policy': entry['policy'],
                                    'record': evidence['raw_result'], 'ptx_body_sha256': entry['ptx_body_sha256'],
                                    'native_text_sha256': sha(b''.join(w.to_bytes(16, 'little') for w in words)),
                                    'words': len(words), 'registers': registers, 'passed': True})
        print('GEMM', entry['contract']['m'], entry['contract']['n'], entry['contract']['k'], entry['contract']['dtype'], flush=True)
    for entry in tables['rowwise']['entries']:
        c = entry['contract']
        plan = ReductionPlan(rows=c['rows'], width=c['width'], op=c['op'], epsilon=c['epsilon_f32'], arch=c['arch'], warps_per_cta=entry['warps_per_cta'])
        source = rowwise_source(plan)
        assert sha(source.encode()) == entry['ptx_body_sha256']
        words, registers = encode_check(source, plan.op, None)
        for evidence in entry['evidence']:
            raw = json.loads(checked(evidence['record'], evidence['record_sha256']))
            name = next(n for n, candidate in raw['candidates'].items() if candidate['certificate']['warps_per_cta'] == entry['warps_per_cta'])
            checked(evidence['artifact_archive'], evidence['artifact_archive_sha256'])
            with zipfile.ZipFile(ROOT / evidence['artifact_archive']) as archive:
                assert archive.read(f'{name}/{plan.op}.ptx').decode() == source
                binary = archive.read(f'{name}/oa.cubin')
            assert words == text_words(binary, plan.op) and registers == regcount(binary, plan.op)
            proof['checks'].append({'kind': 'rowwise', 'contract': c, 'policy': name,
                                    'record': evidence['record'], 'ptx_body_sha256': entry['ptx_body_sha256'],
                                    'native_text_sha256': sha(b''.join(w.to_bytes(16, 'little') for w in words)),
                                    'words': len(words), 'registers': registers, 'passed': True})
    assert current_identity == {'gemm': gd.compiler_identity(), 'rowwise': rd.compiler_identity()}, 'compiler changed during audit'
    assert len(proof['checks']) == 98
    proof['passed'] = True
    proof['counts'] = {'gemm_contracts': 44, 'rowwise_contracts': 5, 'retained_session_checks': 98}
    target = HERE / ('audit-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '.json')
    data = (json.dumps(proof, indent=2) + '\n').encode()
    target.write_bytes(data)
    if args.update:
        for kind, module in [('gemm', gd), ('rowwise', rd)]:
            table = tables[kind]
            table['source_continuity_history'] = table.get('source_continuity_history', []) + [table['source_continuity']]
            table['compiler_identity'] = current_identity[kind]
            table['source_continuity'] = {'audit': str(target.relative_to(ROOT)), 'sha256': sha(data),
                                          'previous_compiler_identity': tables[kind]['compiler_identity'],
                                          'current_compiler_identity': current_identity[kind],
                                          'native_text_and_registers_unchanged': True, 'scope': proof['scope']}
            for entry in table['entries']:
                if 'compiler_identity' in entry:
                    entry['compiler_identity'] = current_identity[kind]
            module.TABLE.write_text(json.dumps(table, indent=2) + '\n')
    print(str(target), proof['counts'], flush=True)


if __name__ == '__main__':
    main()

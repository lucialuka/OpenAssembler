"""Rebind installed policies after exact allocator and retained-binary checks."""
import argparse
import ast
from contextlib import nullcontext
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import subprocess
from unittest.mock import patch
import zipfile

from openassembler import build, gemm_dispatch as gd, rowwise_dispatch as rd
from openassembler.backend import ra
from openassembler.backend.tensor import TensorConfig
from openassembler.container.cubin import regcount, text_words
from openassembler.gemm import source_for
from openassembler.kernels.gemm import GemmPlan, generate as gemm_source
from openassembler.kernels.reductions import ReductionPlan, generate as rowwise_source
from openassembler.ptx.parse import parse_module


def digest(data):
    return sha256(data).hexdigest()


def prior_identity(module, paths, old_ra):
    result = sha256()
    for name, path in sorted(paths.items()):
        data = old_ra if name == 'openassembler/backend/ra.py' else path.read_bytes()
        result.update(name.encode() + b'\0' + data + b'\0')
    if module is rd:
        for name, data in sorted(rd._runtime_source_sections().items()):
            result.update(name.encode() + b'\0' + data + b'\0')
    return result.hexdigest()


def main(output, update):
    tables = {'gemm': json.loads(gd.TABLE.read_text()), 'rowwise': json.loads(rd.TABLE.read_text())}
    old_ra = subprocess.check_output(['git', 'show', 'HEAD:openassembler/backend/ra.py'])
    new_ra = Path(ra.__file__).read_bytes()

    def without_intervals(source):
        text = source.decode()
        node = next(node for node in ast.parse(text).body
                    if isinstance(node, ast.FunctionDef) and node.name == '_intervals')
        lines = text.splitlines(keepends=True)
        return ''.join(lines[:node.lineno-1] + lines[node.end_lineno:])

    assert without_intervals(old_ra) == without_intervals(new_ra)
    frozen_function = next(node for node in ast.parse(old_ra).body
                           if isinstance(node, ast.FunctionDef) and node.name == '_intervals')
    namespace = vars(ra).copy()
    exec(ast.unparse(frozen_function), namespace)
    original_intervals = namespace['_intervals']
    inputs = {'gemm': gd.compiler_inputs(), 'rowwise': rd.identity_inputs()}
    previous = {key: prior_identity(module, inputs[key], old_ra)
                for key, module in [('gemm', gd), ('rowwise', rd)]}
    current = {'gemm': gd.compiler_identity(), 'rowwise': rd.compiler_identity()}
    assert all(table['compiler_identity'] in (previous[key], current[key])
               for key, table in tables.items()), previous
    proof = {'created_utc': datetime.now(timezone.utc).isoformat(), 'passed': False,
             'previous_compiler_identity': previous, 'current_compiler_identity': current,
             'scope': 'Exact code, controls, labels, exits, registers and retained native text under the interval scan change',
             'sources': {name: digest(path.read_bytes()) for paths in inputs.values() for name, path in paths.items()},
             'baseline_ra_sha256': digest(old_ra), 'checks': []}
    checked = {}

    def compare(source, entry, config, binary):
        key = digest(source.encode() + repr(config).encode())
        if key not in checked:
            kernel = parse_module(source).kernels[entry]
            baseline = None
            for context in (patch.object(ra, '_intervals', original_intervals), nullcontext()):
                with context:
                    native = build.generate(kernel, 'sm_80', 'opt', tensor_config=config)
                if baseline is None:
                    baseline = native
                else:
                    assert native == baseline, entry
            checked[key] = native
        code, labels, words, exits, registers = checked[key]
        assert words == text_words(binary, entry) and registers == regcount(binary, entry)
        return {'native_text_sha256': digest(b''.join(w.to_bytes(16, 'little') for w in words)),
                'words': len(words), 'registers': registers}

    def read_checked(path, expected):
        data = Path(path).read_bytes()
        assert digest(data) == expected, path
        return data

    for row in tables['gemm']['entries']:
        for evidence in row['evidence']:
            raw = json.loads(read_checked(evidence['raw_result'], evidence['raw_result_sha256']))
            candidate = raw['candidates'][row['policy']]
            plan = GemmPlan(**candidate['plan'])
            assert gd.contract_key(plan) == row['contract']
            assert digest(gemm_source(plan).encode()) == row['ptx_body_sha256']
            archive_path = Path(evidence['raw_result']).parent / raw['artifact_archive']['path']
            read_checked(archive_path, raw['artifact_archive']['sha256'])
            with zipfile.ZipFile(archive_path) as archive:
                source = archive.read(f"{row['policy']}/gemm.ptx").decode()
                binary = archive.read(f"{row['policy']}/native.cubin")
            assert source == source_for(plan)
            assert digest(binary) == candidate['artifacts']['native']['cubin_sha256']
            check = compare(source, 'gemm', TensorConfig(pointer_mode=plan.pointer_mode,
                                                        pointer_advance=plan.pointer_advance), binary)
            proof['checks'].append({'kind': 'gemm', 'contract': row['contract'], 'evidence': evidence, **check})
    for row in tables['rowwise']['entries']:
        contract = row['contract']
        plan = ReductionPlan(rows=contract['rows'], width=contract['width'], op=contract['op'],
                             epsilon=contract['epsilon_f32'], warps_per_cta=row['warps_per_cta'])
        source = rowwise_source(plan)
        assert digest(source.encode()) == row['ptx_body_sha256']
        for evidence in row['evidence']:
            raw = json.loads(read_checked(evidence['record'], evidence['record_sha256']))
            name = next(name for name, candidate in raw['candidates'].items()
                        if candidate['certificate']['warps_per_cta'] == row['warps_per_cta'])
            read_checked(evidence['artifact_archive'], evidence['artifact_archive_sha256'])
            with zipfile.ZipFile(evidence['artifact_archive']) as archive:
                assert archive.read(f'{name}/{plan.op}.ptx').decode() == source
                binary = archive.read(f'{name}/oa.cubin')
            check = compare(source, plan.op, None, binary)
            proof['checks'].append({'kind': 'rowwise', 'contract': contract, 'evidence': evidence, **check})
    assert len(proof['checks']) == 98
    assert current == {'gemm': gd.compiler_identity(), 'rowwise': rd.compiler_identity()}
    proof['passed'] = True
    proof['script_sha256'] = digest(Path(__file__).read_bytes())
    output.write_text(json.dumps(proof, indent=2) + '\n')
    if update:
        for key, module in [('gemm', gd), ('rowwise', rd)]:
            table = tables[key]
            if table['compiler_identity'] != current[key]:
                table['source_continuity_history'].append(table['source_continuity'])
            table['compiler_identity'] = current[key]
            table['source_continuity'] = {'audit': str(output), 'sha256': digest(output.read_bytes()),
                                         'previous_compiler_identity': previous[key],
                                         'current_compiler_identity': current[key],
                                         'native_text_and_registers_unchanged': True, 'scope': proof['scope']}
            for row in table['entries']:
                if 'compiler_identity' in row:
                    row['compiler_identity'] = current[key]
            module.TABLE.write_text(json.dumps(table, indent=2) + '\n')
    print('Passed 98 retained session checks across 49 installed policies', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--update', action='store_true')
    args = parser.parse_args()
    main(args.output, args.update)

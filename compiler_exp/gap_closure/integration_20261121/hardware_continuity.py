"""Read-only source/native-text continuity for retained model hardware records.

No GPU, toolkit, mutation of production, or replacement of historical evidence.
Current opt generation must match every archived scalar instruction word,
including control bits, declared register count and exit offsets. Source drift
is named explicitly; semantic/runtime sources require exact hardware identity.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from openassembler import build
from openassembler.backend.tensor import needs_tensor
from openassembler.container import nvinfo
from openassembler.container.cubin import regcount, text_words, words_to_bytes
from openassembler.container.elf import Elf, EM_CUDA
from openassembler.isa.word import control
from openassembler.modelfront.kernels import canonical, prepare
from openassembler.ptx.parse import parse_module
from openassembler.transport import runtime_files

# Exact source equality is required here, independently of executable equality.
# The backend itself may drift only with exact native comparisons below; its
# changed filenames and old/new hashes are all retained rather than waived.
REQUIRED_MODEL_SOURCES = (
    'openassembler/modelfront/graph.py',
    'openassembler/modelfront/kernels.py',
    'openassembler/modelfront/kernels_nn.py',
    'openassembler/modelfront/kernels_semantics.py',
    'openassembler/modelfront/reference.py',
    'openassembler/modelfront/optimization.py',
    'openassembler/modelfront/runtime.py',
    'openassembler/modelfront/worker.py',
    'openassembler/modelfront/profiling.py',
    'openassembler/modelfront/frontend.py',
    'openassembler/modelfront/optimizer.py',
    'openassembler/modelfront/replay.py',
)
ALLOWED_EXACT_NATIVE_DRIFT = (
    'openassembler/backend/ra.py',
    'openassembler/kernels/sm80_gemm_dispatch.json',
    'openassembler/kernels/sm80_rowwise_dispatch.json',
)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def source_snapshot():
    return {name: sha(data) for name, data in sorted(runtime_files(native=True).items())}


def checked(condition, message):
    if not condition:
        raise ValueError(message)


def latest(folder):
    for path in sorted((ROOT/'compiler_exp/gap_closure'/folder).glob('hardware-*.json'), reverse=True):
        report = json.loads(path.read_text())
        if report.get('returncode') == 0 and report.get('report', {}).get('passed') is True:
            return path
    raise ValueError(f'no completed successful hardware record in {folder}')


def archive_member(prefix, relative):
    path = Path(relative)
    checked(not path.is_absolute() and '..' not in path.parts, f'unsafe archive path {relative!r}')
    return prefix + path.as_posix()


def run(report_paths):
    checked(bool(report_paths), 'at least one hardware report is required')
    started = time.perf_counter()
    current = source_snapshot()
    auditor_sha = sha(Path(__file__).read_bytes())
    proof = {
        'schema': 1, 'created_utc': datetime.now(timezone.utc).isoformat(),
        'python': platform.python_version(), 'passed': False,
        'scope': 'current opt scalar generation compared bit-for-bit with archived OA executable sections, embedded controls, registers and exits; exact model/runtime source identity checked independently',
        'hardware': 'not-run: reuses the named earlier hardware gates only after native equality',
        'performance': 'not-measured: no new timing or profiler evidence',
        'excluded_claims': ['new hardware execution', 'all PTX semantics', 'new target support',
                            'new ELF construction/toolkit validation', 'new whole-workload speedup'],
        'current_compiler_inputs_sha256': current,
        'auditor_source_sha256': auditor_sha,
        'required_model_sources': list(REQUIRED_MODEL_SOURCES),
        'explicitly_reviewed_native_drift_files': list(ALLOWED_EXACT_NATIVE_DRIFT),
        'reports': [], 'bundles': [], 'failures': [],
    }
    generated = {}
    parsed = {}
    preparations = {}
    cache_hits = 0
    for report_path in report_paths:
        report_path = Path(report_path).resolve()
        data = report_path.read_bytes()
        result = json.loads(data)
        checked(result.get('returncode') == 0 and result.get('report', {}).get('passed') is True,
                f'{report_path}: no successful retained hardware gate')
        archive = report_path.parent/result['archive']['path']
        checked(archive.resolve().is_relative_to(report_path.parent), 'escaping hardware archive')
        archived_bytes = archive.read_bytes()
        checked(sha(archived_bytes) == result['archive']['sha256'], 'hardware archive hash differs')
        old = result['compiler_hashes']
        drift = {name: {'hardware': old.get(name), 'current': current.get(name)}
                 for name in sorted(set(old) | set(current)) if old.get(name) != current.get(name)}
        required = {name: {'hardware': old.get(name), 'current': current.get(name),
                           'passed': old.get(name) == current.get(name) and name in old and name in current}
                    for name in REQUIRED_MODEL_SOURCES}
        record = {'report': str(report_path.relative_to(ROOT)), 'report_sha256': sha(data),
                  'archive': str(archive.relative_to(ROOT)), 'archive_sha256': sha(archived_bytes),
                  'hardware_environment': {key: result['report'].get(key) for key in
                                           ('device','python','torch','cuda_runtime')},
                  'compiler_source_drift': drift, 'required_model_source_checks': required,
                  'bundles': 0, 'kernel_occurrences': 0}
        proof['reports'].append(record)
        for name in sorted(set(drift) - set(ALLOWED_EXACT_NATIVE_DRIFT)):
            proof['failures'].append({'report': record['report'], 'source': name,
                                      'reason': 'unreviewed compiler/runtime/data source drift'})
        for name, check in required.items():
            if not check['passed']:
                proof['failures'].append({'report': record['report'], 'source': name,
                                          'reason': 'required semantic/runtime source differs from hardware identity'})
        with zipfile.ZipFile(archive) as z:
            names = z.namelist()
            checked(len(names) == len(set(names)), 'archive has duplicate members')
            ledgers = sorted(name for name in names if name.endswith('/native-ledger.json'))
            checked(ledgers, 'hardware archive has no native bundles')
            hardware = result['report']
            if 'runs' in hardware and 'cli' in hardware:
                expected = {'work/'+row['name']+'/' for row in hardware['runs']}
                for command in hardware['cli']:
                    args = command['args']
                    if args[0] == 'compile':
                        checked('--out-dir' in args, 'operator CLI record lacks an explicit output directory')
                        expected.add(args[args.index('--out-dir')+1].rstrip('/')+'/')
            elif 'results' in hardware:
                expected = {'work/'+row['name']+'/artifacts/' for row in hardware['results']}
            else:
                raise ValueError('unsupported hardware report bundle-inventory contract')
            checked({name.removesuffix('native-ledger.json') for name in ledgers} == expected,
                    'archive bundle inventory differs from recorded hardware workloads')
            record['expected_bundle_inventory'] = sorted(expected)
            for ledger_path in ledgers:
                prefix = ledger_path.removesuffix('native-ledger.json')
                row = {'report': record['report'], 'bundle': prefix, 'passed': False, 'kernels': []}
                proof['bundles'].append(row)
                record['bundles'] += 1
                try:
                    ledger = json.loads(z.read(ledger_path))
                    checked(ledger.get('schema') == 1 and ledger.get('backend') == 'openassembler-native'
                            and ledger.get('arch') == 'sm_80', 'unsupported native ledger identity')
                    checked(not ledger.get('tensor_kernels') and not ledger.get('tensor_recipes'),
                            'tensor recipes need their own explicitly configured continuity gate')
                    plan = json.loads(z.read(prefix+'plan.json'))
                    plan_key = sha(canonical(plan))
                    checked(plan_key == ledger['plan_sha256'], 'archived plan identity mismatch')
                    source = z.read(prefix+'sources/native.ptx')
                    source_key = sha(source)
                    checked(source_key == ledger['source_sha256'], 'archived PTX identity mismatch')
                    # Re-derive the current kernel contracts from the archived plan.
                    if plan_key not in preparations:
                        _, steps, kernels = prepare(plan)
                        emitted = ('.version 8.7\n.target sm_80\n.address_size 64\n' +
                                   '\n'.join(kernels[name].ptx for name in sorted(kernels))).encode()
                        preparations[plan_key] = steps, kernels, emitted
                    steps, contracts, emitted = preparations[plan_key]
                    checked(emitted == source and steps == ledger['steps'],
                            'current plan emitter/PTX or launch steps differ from archived hardware program')
                    binary = z.read(archive_member(prefix, ledger['cubin']))
                    checked(sha(binary) == ledger['cubin_sha256'], 'archived cubin identity mismatch')
                    elf = Elf.parse(binary)
                    checked(elf.machine == EM_CUDA and elf.flags & 255 == 80, 'archived cubin is not SM80 CUDA')
                    sections = {s.name[6:]: s for s in elf.sections if s.name.startswith('.text.') and s.data}
                    if source_key not in parsed:
                        parsed[source_key] = parse_module(source.decode())
                    module = parsed[source_key]
                    checked(set(module.kernels) == set(ledger['kernels']) == set(sections) == set(contracts),
                            'source/ledger/cubin/current-plan kernel inventories differ')
                    for name, kernel in module.kernels.items():
                        entry = ledger['kernels'][name]
                        checked(not needs_tensor(kernel), f'{name}: tensor instruction in scalar audit')
                        checked(entry['compiled_by'] == 'openassembler' and
                                entry['signature'] == contracts[name].signature and
                                entry['launch'] == contracts[name].launch, f'{name}: kernel/launch identity mismatch')
                        original_text = sections[name].data
                        checked(sha(original_text) == entry['text_sha256'], f'{name}: archived text identity mismatch')
                        # Full parsed Kernel identity includes all operands, labels,
                        # declarations/directives and source positions. No fields
                        # are erased merely to increase the cache hit rate.
                        key = sha(canonical({'arch': 'sm_80', 'opt_level': 'opt', 'kernel': asdict(kernel)}))
                        if key in generated:
                            cache_hits += 1
                        else:
                            code, labels, words, exits, registers = build.generate(kernel, 'sm_80', 'opt')
                            generated[key] = words, exits, registers
                        words, exits, registers = generated[key]
                        old_words = text_words(binary, name)
                        current_text = words_to_bytes(words)
                        equal = words == old_words and current_text == original_text
                        # The full 128-bit comparison already includes these
                        # fields; keep the explicit independent control digest.
                        controls = [asdict(control(word)) for word in words]
                        old_controls = [asdict(control(word)) for word in old_words]
                        archived_registers = regcount(binary, name)
                        info = nvinfo.parse(elf.section('.nv.info.'+name).data)
                        archived_exits = [nvinfo.u32s(r) for r in info if r.name == 'EXIT_INSTR_OFFSETS']
                        checked(len(archived_exits) == 1, f'{name}: missing/duplicate exit metadata')
                        item = {'entry': name, 'parsed_kernel_sha256': key, 'passed': equal and
                                controls == old_controls and registers == archived_registers == entry['registers'] and
                                exits == archived_exits[0],
                                'words': len(words), 'archived_words': len(old_words),
                                'current_native_text_sha256': sha(current_text),
                                'archived_native_text_sha256': sha(original_text),
                                'current_controls_sha256': sha(canonical(controls)),
                                'archived_controls_sha256': sha(canonical(old_controls)),
                                'registers': registers, 'archived_registers': archived_registers,
                                'exit_offsets': exits, 'archived_exit_offsets': archived_exits[0]}
                        if not equal:
                            item['first_word_differences'] = [
                                {'index': i, 'current': hex(a), 'archived': hex(b)}
                                for i, (a, b) in enumerate(zip(words, old_words)) if a != b][:8]
                        row['kernels'].append(item)
                        checked(item['passed'], f'{name}: native words, controls, register count or exits differ')
                        record['kernel_occurrences'] += 1
                    row.update({'passed': True, 'plan_sha256': plan_key, 'source_sha256': source_key,
                                'cubin_sha256': sha(binary)})
                except Exception as exc:
                    row['error'] = f'{type(exc).__name__}: {exc}'
                    proof['failures'].append({'report': record['report'], 'bundle': prefix, 'reason': row['error']})
    final = source_snapshot()
    changed_during_audit = {name: {'before': current.get(name), 'after': final.get(name)}
                            for name in sorted(set(current)|set(final)) if current.get(name) != final.get(name)}
    proof['source_changes_during_audit'] = changed_during_audit
    if changed_during_audit:
        proof['failures'].append({'reason': 'compiler source changed while continuity audit was running'})
    proof['counts'] = {'reports': len(proof['reports']), 'bundles': len(proof['bundles']),
                       'kernel_occurrences': sum(r['kernel_occurrences'] for r in proof['reports']),
                       'unique_generated_kernel_inputs': len(generated), 'kernel_generation_cache_hits': cache_hits,
                       'unique_parsed_sources': len(parsed), 'unique_current_plan_preparations': len(preparations)}
    if not proof['bundles'] or not proof['counts']['kernel_occurrences']:
        proof['failures'].append({'reason': 'continuity audit requires positive bundle and kernel counts'})
    if sha(Path(__file__).read_bytes()) != auditor_sha:
        proof['failures'].append({'reason': 'auditor source changed while continuity audit was running'})
    proof['passed'] = not proof['failures'] and all(r['passed'] for r in proof['bundles'])
    proof['audit_host_seconds'] = time.perf_counter()-started
    proof['source_drift_interpretation'] = ('Named backend/data drift is bound to the current source snapshot only '
        'when every scalar native word/control/register/exit matches and all required model/runtime source identities pass. '
        'This does not establish execution correctness of any changed source or kernel outside these archived programs.')
    return proof


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--operators-report', type=Path)
    parser.add_argument('--profiling-report', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    paths = [args.operators_report or latest('operators'), args.profiling_report or latest('profiling')]
    result = run(paths)
    target = args.output or HERE/('hardware-continuity-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'.json')
    target.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'path': str(target), 'passed': result['passed'], 'counts': result['counts'],
                      'failures': result['failures'], 'seconds': result['audit_host_seconds']}, indent=2))
    raise SystemExit(0 if result['passed'] else 1)


if __name__ == '__main__':
    main()

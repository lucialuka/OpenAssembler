"""Versioned cross-stage inventory; preserve the pre-repair contract snapshot."""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidate', action='store_true')
    args = parser.parse_args()
    if args.candidate:
        from candidate import install
        install()
    from openassembler.backend.contracts import scalar_forms
    from openassembler.isa.encode import Encoder
    specification = importlib.util.spec_from_file_location('contract_fixture_ledger', HERE.parent / 'contracts/ledger.py')
    ledger = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(ledger)
    original_fixture = ledger.fixture
    def atomic_fixture(op, mods, arity, register_backed=True):
        source,line,domain = original_fixture(op,mods,arity,register_backed)
        if op in ('atom','red'):
            form='.'.join((op,*mods))
            address='[storage]' if 'shared' in mods else '[%input]'
            target=(form+' %rout,'+address+',%r0'+(',%r1' if mods[-2]=='cas' else '')+';'
                    if op=='atom' else form+' [%input],%r0; mov.u32 %rout,0;')
            lines=source.splitlines();lines[line-1]=target;source='\n'.join(lines)+'\n'
            domain.update(memory_space='shared' if 'shared' in mods else 'global',element_bits=32,vector_width=1,access_bytes=4,
                          scope='cta' if 'shared' in mods else 'gpu',ordering=mods[0] if mods[0] in ('relaxed','acquire','release','acq_rel') else 'relaxed',address_offset=0,
                          probe='atomic/reduction operation, result kept live where applicable; numerical/hardware gates not run for this fixture')
        return source,line,domain
    def shuffle_fixture(op,mods,arity,register_backed=True):
        source,line,domain=atomic_fixture(op,mods,arity,register_backed)
        if op=='shfl':
            mode=mods[1]
            if mode=='up':
                lines=source.splitlines();lines[line-1]=lines[line-1].replace(',31,0xffffffff;',',0,0xffffffff;');source='\n'.join(lines)+'\n'
            domain.update(mode=mode,clamp=0 if mode=='up' else 31,membermask='0xffffffff',participation='all 32 lanes; not executed')
        return source,line,domain
    ledger.fixture=shuffle_fixture
    encoder = Encoder()
    forms = scalar_forms()
    rows = []
    for (op, mods), arity in sorted(forms.items()):
        row = ledger.evaluate(op, mods, arity, encoder)
        row['parameter_backed_variant'] = ledger.evaluate(op, mods, arity, encoder, register_backed=False)
        rows.append(row)
    memory = []
    for op in ('ld', 'st'):
        for space in ('global', 'shared'):
            for dtype in ('b8', 'u8', 's8', 'b16', 'u16', 's16', 'b32', 'u32', 's32', 'f32', 'b64', 'u64', 's64', 'f64'):
                for width in (1, 2, 4):
                    mods = (space,) + (('v' + str(width),) if width > 1 else ()) + (dtype,)
                    memory.append(ledger.evaluate(op, mods, 2, encoder, declared=(op, mods) in forms))
    fixtures_path = HERE / 'fixtures-shuffles-20261123.json'
    fixtures_path.write_text(json.dumps(ledger.FIXTURES, indent=2) + '\n')
    for row in rows + [row['parameter_backed_variant'] for row in rows] + memory:
        row['fixture_artifact'] = fixtures_path.relative_to(ROOT).as_posix()
    files = ['openassembler/backend/ra.py', 'openassembler/backend/contracts.py',
             'openassembler/backend/compile.py', 'openassembler/isa/encode.py',
             'compiler_exp/gap_closure/contracts/ledger.py',
             'compiler_exp/gap_closure/warp_collectives/ledger_snapshot.py']
    hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in files}
    if args.candidate:
        for name in ('ra.py', 'contracts.py', 'compile.py', 'sched.py'):
            hashes['openassembler/backend/' + name] = hashlib.sha256((HERE / ('candidate-' + name)).read_bytes()).hexdigest()
    report = {'schema': 1, 'target': 'sm_80', 'python': platform.python_version(),
              'scope': 'Versioned representative stage observations; independent numerical/toolkit/hardware gates remain separate',
              'source_hashes': hashes, 'declared_forms': len(forms), 'forms': rows, 'memory_width_matrix': memory,
              'counts': {stage: dict(Counter(row['stages'][stage]['status'] for row in rows)) for stage in rows[0]['stages']},
              'fixture_artifact': fixtures_path.relative_to(ROOT).as_posix(),
              'fixtures_sha256': hashlib.sha256(fixtures_path.read_bytes()).hexdigest(),
              'prior_snapshot': 'compiler_exp/gap_closure/contracts/ledger.json',
              'hardware_evidence_policy': 'Separate source-exact cases in atomics/ordering A100 reports; no attribution to these generated fixtures'}
    output = HERE / 'ledger-shuffles-20261123.json'
    output.write_text(json.dumps(report, indent=2) + '\n')
    assert len(rows) == len(forms)
    assert all(row['stages']['admission']['status'] == 'passed' for row in rows)
    print(output, len(forms), report['counts'])


if __name__ == '__main__':
    main()

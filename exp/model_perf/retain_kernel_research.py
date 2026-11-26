"""Retain result envelopes and archive identities for the November kernel study."""
from hashlib import sha256
import json
from pathlib import Path
import shutil
import zipfile


def main():
    here = Path(__file__).resolve().parent
    evidence = here / 'evidence'
    records = []
    for run in sorted((here / 'runs').iterdir()):
        if not run.name.startswith(('native-pack-', 'pack-reuse-', 'allocator-build-')):
            continue
        result = run / 'result.json'
        if not result.exists():
            continue
        target = evidence / (run.name + '.json')
        shutil.copyfile(result, target)
        envelope = json.loads(target.read_text())
        record = {'result': str(target.relative_to(here)), 'sha256': sha256(target.read_bytes()).hexdigest(),
                  'passed': envelope['result']['passed'], 'returncode': envelope['returncode']}
        archive = run / 'artifacts.zip'
        if archive.exists():
            record['archive'] = {'path': str(archive.relative_to(here)),
                                 'sha256': sha256(archive.read_bytes()).hexdigest(), 'bytes': archive.stat().st_size}
            if not record['passed']:
                with zipfile.ZipFile(archive) as source:
                    if 'results/result.json' in source.namelist():
                        partial = evidence / (run.name + '-partial.json')
                        partial.write_bytes(source.read('results/result.json'))
                        record['partial'] = str(partial.relative_to(here))
        records.append(record)
    scratch = Path('output/kernel-research')
    for name in ('allocator-experiment.json', 'allocator-intervals-experiment.json',
                 'allocator-integrated.json', 'allocator-all-integrated.json', 'allocator-cold.json',
                 'full-suite.log', 'final-suite.log', 'targeted.log', 'docs-check.log'):
        source = scratch / name
        if source.exists():
            target = evidence / ('kernel-study-' + name)
            shutil.copyfile(source, target)
            records.append({'result': str(target.relative_to(here)),
                            'sha256': sha256(target.read_bytes()).hexdigest()})
    output = evidence / 'kernel-study-20261126.json'
    for name in ('allocator-continuity-20261126.json', 'pack-reuse-complete-state-audit-20261126.json',
                 'kernel-study-validation-20261126.json'):
        proof = evidence / name
        if proof.exists():
            records.append({'result': str(proof.relative_to(here)), 'sha256': sha256(proof.read_bytes()).hexdigest()})
    output.write_text(json.dumps({'schema': 1, 'records': records,
                                 'scope': 'Raw envelopes include failed attempts. Archives remain under ignored runs.'}, indent=2) + '\n')
    print(output)


if __name__ == '__main__':
    main()

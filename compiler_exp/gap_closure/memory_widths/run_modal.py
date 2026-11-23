"""A100 byte-exact memory-width and predicate candidate gate; no timing claims."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import modal

HERE = Path(__file__).resolve().parent
ROOT = Path.cwd()
app = modal.App('openassembler-memory-width-foundations')
image = modal.Image.from_registry('nvidia/cuda:12.8.1-devel-ubuntu22.04', add_python='3.12')


@app.function(image=image, gpu='A100-40GB', timeout=1800, max_containers=1, retries=0)
def execute(files):
    import base64
    import ctypes as C
    from dataclasses import asdict
    import io
    import os
    import subprocess
    import sys
    import tempfile
    import zipfile
    with tempfile.TemporaryDirectory(prefix='oa-g12-') as temporary:
        root = Path(temporary)
        for name, data in files.items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        (root / 'env/cuda.sh').chmod(0o700)
        os.chdir(root)
        sys.path[:0] = [str(root), str(root / 'compiler_exp/gap_closure/memory_widths')]
        from cases import GUARD, HEADER, N, THREADS, cases
        from openassembler import build
        from openassembler.gpu.cuda_driver import Driver
        from openassembler.native_ownership import validate
        fixtures = list(cases())
        combined = HEADER + '\n'.join(case['source'][len(HEADER):].replace('probe(', case['name'] + '(') for case in fixtures)
        source = root / 'fixtures.ptx'
        source.write_text(combined)
        records, archive_bytes = [], io.BytesIO()
        driver = Driver()
        driver.open()
        try:
            device = asdict(driver.info())
            if device['cc'] != '8.0' or 'A100' not in device['name']:
                raise RuntimeError(f'A100 required: {device}')
            with zipfile.ZipFile(archive_bytes, 'w', zipfile.ZIP_DEFLATED) as archive:
                archive.writestr('fixtures.ptx', combined)
                for case in fixtures:
                    archive.writestr(case['name'] + '/input.bin', case['input'])
                    archive.writestr(case['name'] + '/expected.bin', case['expected'])
                for level in ('safe', 'mined', 'sched', 'opt'):
                    directory = root / level
                    result = build.compile_paths(source, directory, build.Options(opt_level=level))
                    if result.rc:
                        raise RuntimeError(f'candidate compile {level} failed: {result}')
                    manifest = json.loads((directory / 'manifest.json').read_text())
                    archive.writestr(level + '/manifest.json', json.dumps(manifest, indent=2))
                    entries = {kernel['name']: entry for entry in manifest['entries'] for kernel in entry['kernels']}
                    for case in fixtures:
                        entry = entries[case['name']]
                        native = (directory / entry['cubin']).read_bytes()
                        validate(entry, native)
                        archive.writestr(level + '/' + case['name'] + '/oa.sass.txt', (directory / entry['sass']).read_bytes())
                        for compiler, binary in [('oa', native), ('ptxas', (directory / entry['reference']).read_bytes())]:
                            archive.writestr(level + '/' + case['name'] + '/' + compiler + '.cubin', binary)
                            module = driver.load(binary)
                            try:
                                function = driver.function(module, case['name'])
                                size = len(case['input'])
                                src, dst = driver.alloc(size + 512), driver.alloc(size + 512)
                                try:
                                    driver.htod(src, GUARD + case['input'] + GUARD)
                                    driver.htod(dst, GUARD + b'\x3c' * size + GUARD)
                                    driver.launch(function, (1, 1, 1), (THREADS, 1, 1),
                                                  [C.c_uint64(dst + 256), C.c_uint64(src + 256), C.c_uint32(N)])
                                    actual = driver.dtoh(dst, size + 512)
                                    expected = GUARD + case['expected'] + GUARD
                                    mismatches = [i for i, (a, b) in enumerate(zip(actual, expected)) if a != b]
                                    unchanged = driver.dtoh(src, size + 512) == GUARD + case['input'] + GUARD
                                    archive.writestr(level + '/' + case['name'] + '/' + compiler + '.output.bin', actual)
                                    records.append({'name': case['name'], 'level': level, 'compiler': compiler,
                                                    'contract': case['contract'], 'threads': THREADS, 'valid_lanes': N,
                                                    'compared_bytes': len(actual), 'mismatches': len(mismatches),
                                                    'first_mismatches': mismatches[:20], 'inputs_unchanged': unchanged,
                                                    'guards': actual[:256] == actual[-256:] == GUARD,
                                                    'cubin_sha256': hashlib.sha256(binary).hexdigest(),
                                                    'output_sha256': hashlib.sha256(actual).hexdigest(),
                                                    'resources': driver.func_attrs(function)})
                                finally:
                                    driver.free(src)
                                    driver.free(dst)
                            finally:
                                driver.unload(module)
        finally:
            driver.close()
        report = {'schema': 1, 'completed_utc': datetime.now(timezone.utc).isoformat(), 'python': sys.version,
                  'ptxas': subprocess.check_output(['ptxas', '--version'], text=True), 'device': device,
                  'scope': 'Independent byte-exact memory/predicate outputs, guards and input immutability; no timing',
                  'results': records, 'passed': all(r['mismatches'] == 0 and r['guards'] and r['inputs_unchanged'] for r in records),
                  'source_hashes': {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}}
        return report, base64.b64encode(archive_bytes.getvalue()).decode()


@app.local_entrypoint()
def main():
    import base64
    import sys
    sys.path.insert(0, str(ROOT))
    from openassembler.transport import runtime_files
    files = runtime_files(native=True)
    for name in ('ra.py', 'contracts.py'):
        files['openassembler/backend/' + name] = (HERE / ('candidate-' + name)).read_bytes()
    files['compiler_exp/gap_closure/memory_widths/cases.py'] = (HERE / 'cases.py').read_bytes()
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    report, payload = execute.remote(files)
    archive = HERE / ('a100-' + stamp + '.artifacts.zip')
    archive.write_bytes(base64.b64decode(payload))
    report['artifact_archive'] = {'path': archive.name, 'sha256': hashlib.sha256(archive.read_bytes()).hexdigest()}
    report['harness_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    output = HERE / ('a100-' + stamp + '.json')
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(output, 'passed', report['passed'], 'launches', len(report['results']))
    if not report['passed']:
        raise RuntimeError('A100 memory-width candidate failed independent outputs')

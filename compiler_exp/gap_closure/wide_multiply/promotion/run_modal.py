"""Source-exact A100 low64 multiply gate with original ptxas and CUDA controls."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import modal

ROOT = Path.cwd()
HERE = Path(__file__).resolve().parent
app = modal.App('oa-typed-low64-multiply')
image = modal.Image.from_registry('nvidia/cuda:12.8.1-devel-ubuntu22.04', add_python='3.12')

@app.function(image=image, gpu='A100-40GB', timeout=1800, retries=0, max_containers=1)
def execute(payload):
    import base64, ctypes as C, io, os, struct, subprocess, sys, tempfile, zipfile
    from dataclasses import asdict
    with tempfile.TemporaryDirectory(prefix='oa-low64-multiply-') as temporary:
        root = Path(temporary)
        for name, data in payload.items():
            path = root/name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        (root/'env/cuda.sh').chmod(0o700)
        os.chdir(root)
        sys.path[:0] = [str(root), str(root/'compiler_exp/gap_closure/wide_multiply/promotion')]
        from python_fixtures import cases as expanded_cases, values, expected, SENTINEL, CHOICE
        from openassembler import build
        from openassembler.gpu.cuda_driver import Driver
        from openassembler.native_ownership import validate
        from openassembler.backend.storage import local_frame_inventory
        fixtures = list(expanded_cases())
        driver = Driver()
        driver.open()
        device = asdict(driver.info())
        assert device['cc'] == '8.0' and 'A100' in device['name']
        rows, error = [], None
        blob = io.BytesIO()
        guard, lanes = b'\xa5'*256, 384
        try:
            with zipfile.ZipFile(blob, 'w', zipfile.ZIP_DEFLATED) as archive:
                for case in fixtures:
                    name = case['name']
                    source = root/(name + ('.py' if 'python_source' in case else '.cu' if 'cuda_source' in case else '.ptx'))
                    source.write_text(case.get('python_source', case.get('cuda_source', case.get('source'))))
                    archive.writestr(name+'/input'+source.suffix, source.read_bytes())
                    binaries = []
                    for level in ('safe', 'mined', 'sched', 'opt'):
                        target = root/(name+'-'+level)
                        result = build.compile_paths(source, target, build.Options(opt_level=level))
                        assert result.rc == 0, result
                        manifest = json.loads((target/'manifest.json').read_text())
                        entry = manifest['entries'][0]
                        contract = entry['kernels'][0]
                        assert contract['name'] == case['entry']
                        prefix = name+'/'+level+'/'
                        emitted = (target/entry['ptx']).read_bytes()
                        if level == 'safe':
                            archive.writestr(name+'/source.ptx', emitted)
                            binaries.append(('ptxas', (target/entry['reference']).read_bytes()))
                        archive.writestr(prefix+'emitted.ptx', emitted)
                        archive.writestr(prefix+'manifest.json', json.dumps(manifest, indent=2))
                        binary = (target/entry['cubin']).read_bytes()
                        validate(entry, binary)
                        binaries.append(('oa-'+level, binary))
                    for compiler, binary in binaries:
                        prefix = name+'/'+compiler+'/'
                        archive.writestr(prefix+'native.cubin', binary)
                        path = root/(name+'-'+compiler+'.cubin')
                        path.write_bytes(binary)
                        archive.writestr(prefix+'native.sass', subprocess.check_output(['nvdisasm', '-hex', '-c', str(path)]))
                        module = driver.load(binary)
                        try:
                            fn = driver.function(module, case['entry'])
                            resources = driver.func_attrs(fn)
                            frames = local_frame_inventory(binary) if compiler.startswith('oa-') else {}
                            if compiler.startswith('oa-'):
                                assert frames[case['entry']] == resources['local_bytes'] == 0
                            inp, out = driver.alloc(lanes*16+512), driver.alloc(lanes*8+512)
                            try:
                                for n in (0, 1, 259, 384):
                                    for seed in (27391, 27392):
                                        data = values(seed, lanes)
                                        before = guard+struct.pack('<768Q', *data)+guard
                                        initial = guard+struct.pack('<Q', SENTINEL)*lanes+guard
                                        driver.htod(inp, before)
                                        driver.htod(out, initial)
                                        args = [C.c_uint64(out+256), C.c_uint64(inp+256), C.c_uint32(n)]
                                        if case['mode'] == 'cuda':
                                            args = [C.c_uint64(out+256), C.c_uint64(inp+256), C.c_uint64(inp+256+lanes*8), C.c_uint32(n)]
                                        else:
                                            args.append(C.c_uint64(CHOICE))
                                        driver.launch(fn, (3, 1, 1), (128, 1, 1), args)
                                        actual, unchanged = driver.dtoh(out, len(initial)), driver.dtoh(inp, len(before))
                                        wanted = guard+expected(data, case, n, lanes)+guard
                                        run = prefix+str(n)+'/'+str(seed)+'/'
                                        for key, raw in (('before-input', before), ('input', unchanged), ('output', actual)):
                                            archive.writestr(run+key+'.bin', raw)
                                        rows.append(dict(name=name, compiler=compiler, n=n, seed=seed,
                                            passed=actual == wanted and unchanged == before,
                                            resources=resources, local_frame_bytes=frames.get(case['entry']),
                                            cubin_sha256=hashlib.sha256(binary).hexdigest(),
                                            input_sha256=hashlib.sha256(unchanged).hexdigest(), output_sha256=hashlib.sha256(actual).hexdigest()))
                            finally:
                                driver.free(inp)
                                driver.free(out)
                        finally:
                            driver.unload(module)
        except Exception as exc:
            error = dict(type=type(exc).__name__, message=str(exc))
        finally:
            driver.close()
        return dict(schema=1, passed=error is None and len(rows) == len(fixtures)*40 and all(row['passed'] for row in rows),
            error=error, results=rows, device=device, python=sys.version,
            nvcc=subprocess.check_output(['nvcc', '--version'], text=True),
            source_hashes={name: hashlib.sha256(data).hexdigest() for name, data in payload.items()},
            scope='Low64 integer multiplication with independent arbitrary-precision references, original PTX ptxas controls and actual unsigned CUDA source and typed Python arrays/parameters/literals/augmented/helpers/loops/guards/casts, all four native OA levels, full guarded bytes and input immutability; no performance claim'), base64.b64encode(blob.getvalue()).decode()

@app.local_entrypoint()
def main():
    import base64, sys
    sys.path.insert(0, str(ROOT))
    from openassembler.transport import runtime_files
    payload = runtime_files(native=True)
    for name, local in {'openassembler/backend/ra.py': 'candidate-ra.py',
                        'openassembler/backend/contracts.py': 'candidate-contracts.py',
                        'compiler_exp/gap_closure/wide_multiply/promotion/fixtures.py': 'fixtures.py',
                        'compiler_exp/gap_closure/wide_multiply/promotion/cuda.cu': 'cuda.cu',
                        'compiler_exp/gap_closure/wide_multiply/promotion/python_fixtures.py': 'python_fixtures.py',
                        'openassembler/pyfront/compile.py': 'candidate-pyfront.py'}.items():
        payload[name] = (HERE/local).read_bytes()
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    report, encoded = execute.remote(payload)
    archive = HERE/('candidate-a100-'+stamp+'.zip')
    archive.write_bytes(base64.b64decode(encoded))
    report.update(archive=dict(path=archive.name, sha256=hashlib.sha256(archive.read_bytes()).hexdigest()),
                  harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    path = HERE/('candidate-a100-'+stamp+'.json')
    path.write_text(json.dumps(report, indent=2)+'\n')
    print(path, 'passed', report['passed'], 'launches', len(report['results']))
    if not report['passed']:
        raise RuntimeError('low64 multiply hardware gate failed')

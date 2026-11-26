"""Original narrow scalar entries and public explicit launch specs on A100."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import modal

ROOT = Path.cwd()
HERE = Path(__file__).resolve().parent
app = modal.App('oa-narrow-scalar-parameters')
image = modal.Image.from_registry('nvidia/cuda:12.8.1-devel-ubuntu22.04', add_python='3.12')


@app.function(image=image, gpu='A100-40GB', timeout=1800, retries=0, max_containers=1)
def execute(payload):
    import base64
    import ctypes as C
    import io
    import os
    import struct
    import subprocess
    import sys
    import tempfile
    import zipfile
    from dataclasses import asdict
    with tempfile.TemporaryDirectory(prefix='oa-narrow-parameter-') as temporary:
        root = Path(temporary)
        for name, data in payload.items():
            path = root/name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        (root/'env/cuda.sh').chmod(0o700)
        os.chdir(root)
        sys.path[:0] = [str(root), str(root/'compiler_exp/gap_closure/narrow_parameters')]
        from fixtures import module, CASES, fields, inputs, expected, initial
        from openassembler import build
        from openassembler.cli import main as cli
        from openassembler.gpu.cuda_driver import Driver, run_plan_job
        from openassembler.native_ownership import validate
        from openassembler.runner import RunOptions, build_plan
        driver = Driver()
        driver.open()
        device = asdict(driver.info())
        assert device['cc'] == '8.0' and 'A100' in device['name']
        source = root/'original.ptx'
        source.write_text(module())
        blob = io.BytesIO()
        rows, spec_rows, error = [], [], None
        guard = b'\xa5'*256
        lanes = 128
        try:
            with zipfile.ZipFile(blob, 'w', zipfile.ZIP_DEFLATED) as archive:
                archive.writestr('original.ptx', source.read_bytes())
                binaries = []
                for level in ('safe', 'mined', 'sched', 'opt'):
                    output = root/('output-'+level)
                    assert cli(['compile', str(source), '--out-dir', str(output), '--opt-level', level]) == 0
                    manifest = json.loads((output/'manifest.json').read_text())
                    entry, = manifest['entries']
                    binary = (output/entry['cubin']).read_bytes()
                    validate(entry, binary)
                    archive.writestr(level+'/manifest.json', json.dumps(manifest, indent=2))
                    archive.writestr(level+'/oa.cubin', binary)
                    archive.writestr(level+'/oa.sass', subprocess.check_output(['nvdisasm', '-hex', '-c', str(output/entry['cubin'])]))
                    binaries.append(('oa-'+level, binary))
                    if level == 'safe':
                        reference = (output/entry['reference']).read_bytes()
                        archive.writestr('ptxas.cubin', reference)
                        binaries.append(('ptxas', reference))
                    # Actual public explicit-run schema + batch driver launch bindings.
                    for case in CASES:
                        values = inputs(case, 49117, 4, 0, 37)
                        args = []
                        for dtype, name in fields(case):
                            if name == 'out': args.append({'buf': 'out', 'offset': 256})
                            else:
                                bits = int(dtype[1:]);value = values[name] & ((1 << bits)-1)
                                kind = ('i' if dtype.startswith('s') else 'u')+str(bits)
                                if dtype.startswith('s') and value & (1 << (bits-1)):value -= 1 << bits
                                args.append({kind: value})
                        initial_bytes = guard+initial(lanes)+guard
                        spec = dict(buffers={'out': dict(dtype='u32', count=len(initial_bytes)//4, init=list(struct.unpack('<'+str(len(initial_bytes)//4)+'I', initial_bytes)))},
                                    steps=[dict(kernel=case['name'], block=lanes, args=args)], compare=['out'], reps=0)
                        path = root/'launch.json'
                        path.write_text(json.dumps(spec))
                        plan = build_plan(RunOptions(out_dir=output, spec=path, reps=0))
                        job = plan.jobs[0]
                        result = run_plan_job(driver, job)
                        actual = result['out']['out']
                        wanted = guard+expected(case, values, lanes)+guard
                        assert actual == wanted, ('explicit-run', case, level)
                        prefix = 'explicit/'+level+'/'+case['name']+'/'
                        archive.writestr(prefix+'spec.json', json.dumps(spec, indent=2))
                        archive.writestr(prefix+'output.bin', actual)
                        spec_rows.append(dict(entry=case['name'], level=level, passed=True, output_sha256=hashlib.sha256(actual).hexdigest()))
                for compiler, binary in binaries:
                    file = root/(compiler+'.cubin')
                    file.write_bytes(binary)
                    loaded = driver.load(binary)
                    try:
                        for case in CASES:
                            fn = driver.function(loaded, case['name'])
                            resources = driver.func_attrs(fn)
                            ptr = driver.alloc(lanes*32+512)
                            try:
                                for seed in (49117, 49118):
                                    for sample in range(6):
                                        for n in (0, 1, 91, 128):
                                            values = inputs(case, seed, sample, ptr+256, n)
                                            driver.htod(ptr, guard+initial(lanes)+guard)
                                            cargs = [({8:C.c_uint8,16:C.c_uint16,32:C.c_uint32,64:C.c_uint64}[int(dtype[1:])])(values[name]) for dtype,name in fields(case)]
                                            driver.launch(fn, (1,1,1), (lanes,1,1), cargs)
                                            actual = driver.dtoh(ptr, lanes*32+512)
                                            wanted = guard+expected(case, values, lanes)+guard
                                            prefix = f'{compiler}/{case["name"]}/{seed}/{sample}/{n}/'
                                            archive.writestr(prefix+'output.bin', actual)
                                            rows.append(dict(compiler=compiler, entry=case['name'], seed=seed, sample=sample, n=n,
                                                             passed=actual==wanted, resources=resources, cubin_sha256=hashlib.sha256(binary).hexdigest(),
                                                             output_sha256=hashlib.sha256(actual).hexdigest()))
                            finally:
                                driver.free(ptr)
                    finally:
                        driver.unload(loaded)
        except Exception as exc:
            error = dict(type=type(exc).__name__, message=str(exc))
        finally:
            driver.close()
        return dict(schema=1, passed=error is None and len(rows)==14400 and all(row['passed'] for row in rows) and len(spec_rows)==240,
                    error=error, results=rows, explicit_launch_results=spec_rows, device=device, python=sys.version,
                    nvcc=subprocess.check_output(['nvcc','--version'],text=True),
                    source_hashes={name:hashlib.sha256(data).hexdigest() for name,data in payload.items()},
                    scope='60 original narrow scalar PTX entries, independent exact sign/zero extension and adjacent-slot/guard/loop/alias/slice effects at four OA levels plus ptxas, two seeds/six edge samples/four lengths, complete guards/tails; 240 actual explicit run-spec/batch-driver launches; A100 correctness without timing'), base64.b64encode(blob.getvalue()).decode()


@app.local_entrypoint()
def main():
    import base64
    import sys
    sys.path.insert(0, str(ROOT))
    from openassembler.transport import runtime_files
    payload = runtime_files(native=True)
    for suffix, name in json.loads((HERE/'inputs.json').read_text()).items():
        payload[name] = (HERE/('candidate-'+suffix+'.py')).read_bytes()
    payload['compiler_exp/gap_closure/narrow_parameters/fixtures.py'] = (HERE/'fixtures.py').read_bytes()
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    report, encoded = execute.remote(payload)
    archive = HERE/('candidate-a100-'+stamp+'.zip')
    archive.write_bytes(base64.b64decode(encoded))
    report.update(archive=dict(path=archive.name, sha256=hashlib.sha256(archive.read_bytes()).hexdigest()),
                  harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    path = HERE/('candidate-a100-'+stamp+'.json')
    path.write_text(json.dumps(report, indent=2)+'\n')
    print(path, 'passed', report['passed'], 'launches', len(report['results']), 'explicit', len(report['explicit_launch_results']))
    if not report['passed']: raise RuntimeError('Narrow scalar ABI gate failed')

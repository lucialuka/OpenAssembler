"""Frozen A100 overlap-policy study plus entry-ABI hardware regression.

Use: modal run compiler_exp/latency_hiding/run_modal.py --action entries|timing
No policy table is changed. Every timing candidate uses production codegen and
the established full-output/guard/immutability/poisoned-graph admission harness.
"""
from datetime import datetime, timezone
import base64
import hashlib
import io
import json
from pathlib import Path
import zipfile

import modal

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1] if len(HERE.parents) > 1 else HERE
app = modal.App('openassembler-compiler-coverage-and-overlap')
image = modal.Image.from_registry('pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel').add_local_python_source('openassembler')


@app.function(image=image, gpu='A100-40GB', timeout=960, max_containers=1, retries=0)
def measure(request, runtime):
    import os
    import subprocess
    import sys
    import tempfile
    with tempfile.TemporaryDirectory(prefix='oa-overlap-') as directory:
        for name, data in runtime.items():
            path = Path(directory) / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        (Path(directory) / 'env/cuda.sh').chmod(0o700)
        source, output = Path(directory) / 'request.json', Path(directory) / 'result.json'
        source.write_text(json.dumps(request))
        code = ('import json,sys; from openassembler.gemm_tune import benchmark; '
                'from openassembler.kernels.gemm import GemmPlan; '
                'r=json.load(open(sys.argv[1])); '
                'v=benchmark(GemmPlan(**r["plan"]),rounds=3,samples=8); '
                'json.dump(v,open(sys.argv[2],"w"),allow_nan=False)')
        subprocess.run([sys.executable, '-c', code, str(source), str(output)], cwd=directory,
                       env=dict(os.environ, PYTHONPATH=directory), check=True, timeout=900)
        result = json.loads(output.read_text())
        if 'session' in request:
            result['session'] = dict(request['session'], container_id=os.environ.get('MODAL_TASK_ID'),
                                     function_call_id=modal.current_function_call_id())
        return result


@app.function(image=image, gpu='A100-40GB', timeout=180, max_containers=1, retries=0)
def check_entries(products):
    import ctypes as C
    from dataclasses import asdict
    import struct
    from openassembler.gpu.cuda_driver import Driver
    driver = Driver()
    driver.open()
    records = []
    try:
        info = asdict(driver.info())
        if info['cc'] != '8.0' or 'A100' not in info['name']:
            raise ValueError('A100 required')
        for product in products:
            binary = base64.b64decode(product['binary'])
            assert hashlib.sha256(binary).hexdigest() == product['cubin_sha256']
            module = driver.load(binary)
            try:
                function = driver.function(module, product['kernel'])
                count = 3 * product['block']
                guard = bytes([0xa5]) * 256
                expected = struct.pack(f'<{count}I', *[i + 17 for i in range(count)])
                pointer = driver.alloc(512 + len(expected))
                try:
                    for poison in (0x00, 0x3c, 0xff):
                        driver.htod(pointer, guard + bytes([poison]) * len(expected) + guard)
                        driver.launch(function, (3, 1, 1), (product['block'], 1, 1), [C.c_uint64(pointer + 256)])
                        actual = driver.dtoh(pointer, 512 + len(expected))
                        records.append({k:v for k,v in product.items() if k != 'binary'} |
                                       {'poison': poison, 'elements': count, 'resources': driver.func_attrs(function),
                                        'passed': actual == guard + expected + guard})
                finally:
                    driver.free(pointer)
            finally:
                driver.unload(module)
        return {'scope': 'OA-native full integer outputs and guard bytes; no performance claim',
                'device': info, 'results': records, 'passed': all(r['passed'] for r in records)}
    finally:
        driver.close()


def entry_products():
    from openassembler import build
    from openassembler.native_ownership import validate
    directory = ROOT / 'compiler_exp/repository_coverage/hardware'
    directory.mkdir(parents=True, exist_ok=True)
    body = '''.reg .u64 %d, %a; .reg .u32 %t, %b, %n, %i, %v;
ld.param.u64 %d, [out]; mov.u32 %t, %tid.x; mov.u32 %b, %ctaid.x;
mov.u32 %n, %ntid.x; mad.lo.u32 %i, %b, %n, %t; add.u32 %v, %i, 17;
mul.wide.u32 %a, %i, 4; add.u64 %a, %d, %a; st.global.u32 [%a], %v; ret;'''
    products = []
    for name, prefix, directive, block in [('bare', '', '', 64), ('weak', '.weak ', '.maxntid 128', 64),
                                          ('required', '.visible ', '.reqntid 128,1,1', 128)]:
        source = directory / (name + '.ptx')
        source.write_text('.version 8.7\n.target sm_80\n.address_size 64\n' +
                          f'{prefix}.entry {name}(.param .u64 out) {directive} {{ {body} }}\n')
        output = directory / (name + '-output')
        result = build.compile_paths(source, output, build.Options(verify=True))
        if result.rc:
            raise ValueError(f'entry regression build failed: {name}')
        entry = json.loads((output / 'manifest.json').read_text())['entries'][0]
        binary = (output / entry['cubin']).read_bytes()
        validate(entry, binary)
        products.append({'kernel': name, 'block': block, 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                         'cubin_sha256': hashlib.sha256(binary).hexdigest(),
                         'binary': base64.b64encode(binary).decode()})
    return products


@app.local_entrypoint()
def main(action: str = 'timing', campaign: str = 'admission-20261115'):
    import sys
    sys.path.insert(0, str(ROOT))
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    if action == 'entries':
        result = check_entries.remote(entry_products())
        result['harness_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        path = ROOT / 'compiler_exp/repository_coverage' / ('a100-entries-' + stamp + '.json')
        path.write_text(json.dumps(result, indent=2) + '\n')
        print(path, 'passed=', result['passed'])
        if not result['passed']:
            raise ValueError('A100 entry regression failed')
        return
    if action in ('selection', 'heldout'):
        import uuid
        from dataclasses import asdict
        from openassembler.transport import runtime_files
        from openassembler.kernels.gemm import GemmPlan
        from openassembler.gemm_promotion import freeze_selection, audit_record
        directory = HERE / 'runs' / campaign
        directory.mkdir(parents=True, exist_ok=True)
        runtime = runtime_files(native=True)
        runtime_hashes = {n:hashlib.sha256(v).hexdigest() for n,v in sorted(runtime.items())}
        spec = directory / 'runtime.json'
        if action == 'selection':
            if spec.exists():
                raise ValueError('selection campaign already exists; use a new name')
            spec.write_text(json.dumps(runtime_hashes,indent=2)+'\n')
        else:
            import subprocess
            before = json.loads(spec.read_text())
            # Reconstruct the frozen documentation-check helper rather than
            # changing the runtime snapshot when only its local docs scan grows.
            helper = 'tools/check_docs.py'
            if before.get(helper) != runtime_hashes.get(helper):
                original = subprocess.check_output(['git','show','HEAD:' + helper],cwd=ROOT)
                if hashlib.sha256(original).hexdigest() != before.get(helper):
                    raise ValueError('cannot reconstruct frozen documentation helper')
                runtime[helper] = original
                runtime_hashes[helper] = hashlib.sha256(original).hexdigest()
            changed = {name for name in set(before) | set(runtime_hashes)
                       if before.get(name) != runtime_hashes.get(name)}
            # Offline continuity refreshes only table provenance/identity. These
            # explicit-policy benchmarks never use measured-policy dispatch.
            metadata = {'openassembler/kernels/sm80_gemm_dispatch.json',
                        'openassembler/kernels/sm80_rowwise_dispatch.json'}
            if changed - metadata:
                raise ValueError(f'compiler/runtime changed after selection: {sorted(changed)}')
            (directory / 'heldout-runtime.json').write_text(json.dumps(
                {'hashes':runtime_hashes, 'metadata_only_changes':sorted(changed)},indent=2)+'\n')
        session = {'id':f'{campaign}-{action}-{uuid.uuid4()}', 'split':action,
                   'modal_app_id':app.app_id, 'container_id':None}
        for shape in [(128,512,1024), (256,768,1024)]:
            key = 'x'.join(map(str,shape))
            path = directory / f'{key}-{action}.json'
            if path.exists():
                raise ValueError('refusing to overwrite an existing admission record')
            result = measure.remote({'plan':asdict(GemmPlan(*shape,'f16')), 'session':session}, runtime)
            artifacts = result.pop('artifact_bytes')
            buffer = io.BytesIO()
            with zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED) as archive:
                for name,data in artifacts.items(): archive.writestr(name,base64.b64decode(data))
            payload = buffer.getvalue()
            archive_path = path.with_suffix('.artifacts.zip')
            archive_path.write_bytes(payload)
            result['artifact_archive'] = {'path':archive_path.name,'sha256':hashlib.sha256(payload).hexdigest()}
            path.write_text(json.dumps(result,indent=2)+'\n')
            if action == 'selection':
                freeze_selection(path,directory / f'{key}-frozen.json')
            else:
                audit_record(path,split='heldout')
            print(key, action, 'audited', flush=True)
        return
    if action != 'timing':
        raise ValueError('action must be entries, timing, selection or heldout')
    from dataclasses import asdict
    from openassembler.transport import runtime_files
    from openassembler.gemm_dispatch import candidate_plans, select_plan
    from openassembler.gemm_tune import choose_policy
    from openassembler.kernels.gemm import GemmPlan
    runtime = runtime_files(native=True)
    directory = HERE / 'runs' / stamp
    directory.mkdir(parents=True)
    shapes = [(128,512,1024), (1024,1024,1024), (256,768,1024)]
    protocol = {'schema': 1, 'scope': 'resident FP16 RCR GEMM; FP32 accumulation/output; no whole-application claim',
                'rounds': 3, 'samples': 8, 'sessions': 2,
                'harness_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'runtime_sha256': {n:hashlib.sha256(v).hexdigest() for n,v in sorted(runtime.items())},
                'cases': [{'plan':asdict(GemmPlan(*shape, 'f16')), 'policies':list(candidate_plans(GemmPlan(*shape, 'f16'))),
                           'incumbent': select_plan(GemmPlan(*shape, 'f16'))[1]} for shape in shapes],
                'promotion': 'none; independent descriptive replay, existing tables unchanged'}
    (directory / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    # Freeze a first-session winner for each shape, then assess the identical
    # policy in the second session. Never replace it with a held-out winner.
    frozen = {}
    for session in range(2):
        for shape in shapes:
            key = 'x'.join(map(str,shape))
            result = measure.remote({'plan':asdict(GemmPlan(*shape,'f16'))}, runtime)
            artifacts = result.pop('artifact_bytes')
            buffer = io.BytesIO()
            with zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED) as archive:
                for name,data in artifacts.items():
                    archive.writestr(name,base64.b64decode(data))
            payload = buffer.getvalue()
            archive_path = directory / f'{key}-session{session}.artifacts.zip'
            archive_path.write_bytes(payload)
            result['artifact_archive'] = {'path':archive_path.name,'sha256':hashlib.sha256(payload).hexdigest()}
            if session == 0:
                frozen[key] = choose_policy(result)
                (directory / 'frozen-selection.json').write_text(json.dumps(frozen,indent=2)+'\n')
            result['frozen_policy'] = frozen[key]
            (directory / f'{key}-session{session}.json').write_text(json.dumps(result,indent=2)+'\n')
            print(key, 'session',session,'frozen',frozen[key],flush=True)
    print(directory)

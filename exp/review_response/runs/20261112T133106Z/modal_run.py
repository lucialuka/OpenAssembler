"""Stage frozen candidate/compiler sources and isolate each A100 model worker."""
from pathlib import Path
import json
import modal

app = modal.App('openassembler-review-response')
image = modal.Image.from_registry('nvidia/cuda:12.8.1-devel-ubuntu22.04', add_python='3.12').pip_install(
    'torch==2.7.1', index_url='https://download.pytorch.org/whl/cu128')


@app.function(image=image, gpu='A100-40GB', timeout=1800, max_containers=1, retries=0)
def measure(files):
    import os
    import subprocess
    import sys
    import tempfile
    import hashlib
    import io
    import zipfile
    with tempfile.TemporaryDirectory(prefix='oa-review-response-') as directory:
        root = Path(directory)
        for name, raw in files.items():
            path = root/name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
        (root/'env/cuda.sh').chmod(0o700)
        results = []
        for case in ['native_model', 'native_attention_model', 'native_tensor_model']:
            completed = subprocess.run([sys.executable, '-m', 'exp.review_response.campaign', case],
                cwd=root, env=dict(os.environ, PYTHONPATH=directory), capture_output=True, text=True, timeout=500)
            (root/'results').mkdir(exist_ok=True)
            (root/'results'/f'{case}.log').write_text(completed.stdout+completed.stderr)
            if completed.returncode:
                results.append({'case': case, 'passed': False, 'error': (completed.stdout+completed.stderr)[-6000:]})
            else:
                results.append(json.loads((root/'results'/case/'result.json').read_text()))
        output = io.BytesIO()
        with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as archive:
            for path in (root/'results').rglob('*'):
                if path.is_file():
                    archive.write(path, path.relative_to(root).as_posix())
        return {'cases': results, 'source_sha256': {n: hashlib.sha256(v).hexdigest() for n,v in files.items()},
                'artifacts': output.getvalue()}


@app.local_entrypoint()
def main():
    import hashlib
    from datetime import datetime, timezone
    from openassembler.transport import runtime_files
    files = runtime_files(native=True)
    root = Path(__file__).resolve().parents[2]
    for name in ['campaign.py', 'replay.py', 'audit.py']:
        files['exp/review_response/'+name] = (Path(__file__).parent/name).read_bytes()
    for case in ['native_model', 'native_attention_model', 'native_tensor_model']:
        files[f'cases/{case}/model.py'] = (root/'examples'/case/'model.py').read_bytes()
    output = Path(__file__).parent/'runs'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    output.mkdir(parents=True)
    for name in ['campaign.py', 'replay.py', 'audit.py', 'modal_run.py']:
        (output/name).write_bytes((Path(__file__).parent/name).read_bytes())
    result = measure.remote(files)
    (output/'artifacts.zip').write_bytes(result.pop('artifacts'))
    (output/'result.json').write_text(json.dumps(result, indent=2, allow_nan=False)+chr(10))
    print(json.dumps({'path': str(output), 'cases': [{ 'case': c['case'], 'passed': c['passed'],
        'host_speedup': [r['host_speedup'] for r in c.get('rounds', [])], 'error': c.get('error')} for c in result['cases']]}))
    assert all(c['passed'] for c in result['cases'])

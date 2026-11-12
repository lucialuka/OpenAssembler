"""A100 loss and build ablations, independent from training-replay workers."""
from pathlib import Path
import json
import modal

app=modal.App('openassembler-review-ablations')
image=modal.Image.from_registry('nvidia/cuda:12.8.1-devel-ubuntu22.04',add_python='3.12').pip_install(
    'torch==2.7.1',index_url='https://download.pytorch.org/whl/cu128')


@app.function(image=image,gpu='A100-40GB',timeout=1800,max_containers=1,retries=0)
def run(files):
    import hashlib,io,os,subprocess,sys,tempfile,zipfile
    with tempfile.TemporaryDirectory() as directory:
        root=Path(directory)
        for n,b in files.items():
            p=root/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b)
        (root/'env/cuda.sh').chmod(0o700)
        result={}
        for campaign in ['loss_campaign','build_campaign']:
            p=subprocess.run([sys.executable,'-m','exp.review_response.'+campaign],cwd=root,
                env=dict(os.environ,PYTHONPATH=directory),text=True,capture_output=True,timeout=800)
            (root/'results').mkdir(exist_ok=True)
            (root/'results'/f'{campaign}.log').write_text(p.stdout+p.stderr)
            path=root/'results'/('loss' if campaign=='loss_campaign' else 'build')/'result.json'
            result[campaign]=json.loads(path.read_text()) if path.exists() else {'passed':False,'error':(p.stdout+p.stderr)[-6000:]}
        buffer=io.BytesIO()
        with zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED) as z:
            for p in (root/'results').rglob('*'):
                if p.is_file():z.write(p,p.relative_to(root))
        return {'results':result,'source_sha256':{n:hashlib.sha256(b).hexdigest() for n,b in files.items()},'artifacts':buffer.getvalue()}


@app.local_entrypoint()
def main():
    from datetime import datetime,timezone
    from openassembler.transport import runtime_files
    files=runtime_files(native=True);here=Path(__file__).parent;root=here.resolve().parents[1]
    for n in ['loss_campaign.py','ordered_loss.py','audit.py','build_campaign.py']:
        files['exp/review_response/'+n]=(here/n).read_bytes()
    files['exp/review_response/input/baseline-kernels.py']=(here/'input/baseline-kernels.py').read_bytes()
    files['cases/native_tensor_model/model.py']=(root/'examples/native_tensor_model/model.py').read_bytes()
    dest=here/'runs'/('ablation-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'));dest.mkdir()
    for n in ['loss_campaign.py','ordered_loss.py','audit.py','build_campaign.py','ablation_run.py']:
        (dest/n).write_bytes((here/n).read_bytes())
    result=run.remote(files)
    (dest/'artifacts.zip').write_bytes(result.pop('artifacts'))
    (dest/'result.json').write_text(json.dumps(result,indent=2)+chr(10))
    print(json.dumps({'path':str(dest),'results':{n:{'passed':r['passed'],'ratios':[x['speedup'] for x in r.get('rounds',[])]} for n,r in result['results'].items()}}))
    assert all(r['passed'] for r in result['results'].values())

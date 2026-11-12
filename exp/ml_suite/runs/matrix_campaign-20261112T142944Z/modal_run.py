"""Run frozen experiments in a fresh subprocess on an A100."""
from pathlib import Path
import json
import modal
app=modal.App('openassembler-ml-suite')
image=modal.Image.from_registry('nvidia/cuda:12.8.1-devel-ubuntu22.04',add_python='3.12').pip_install(
    'torch==2.7.1',index_url='https://download.pytorch.org/whl/cu128').pip_install('numpy','pytest==8.4.2')


@app.function(image=image,gpu='A100-40GB',timeout=1800,max_containers=1,retries=0)
def execute(files,campaign):
    import hashlib,io,os,subprocess,sys,tempfile,zipfile
    with tempfile.TemporaryDirectory(prefix='oa-ml-suite-') as directory:
        root=Path(directory)
        for name,raw in files.items():
            p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
        (root/'env/cuda.sh').chmod(0o700)
        proc=subprocess.run([sys.executable,'-m','exp.ml_suite.'+campaign],cwd=root,
            env=dict(os.environ,PYTHONPATH=directory),capture_output=True,text=True,timeout=1650)
        (root/'results').mkdir(exist_ok=True)
        (root/'results/worker.log').write_text(proc.stdout+proc.stderr)
        result_path=root/'results'/campaign.replace('_campaign','')/'result.json'
        result=json.loads(result_path.read_text()) if result_path.exists() else {'passed':False,'error':(proc.stdout+proc.stderr)[-10000:]}
        output=io.BytesIO()
        with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED) as archive:
            for p in (root/'results').rglob('*'):
                if p.is_file():archive.write(p,p.relative_to(root))
        return {'result':result,'source_sha256':{n:hashlib.sha256(b).hexdigest() for n,b in files.items()},
                'artifacts':output.getvalue(),'returncode':proc.returncode}


@app.local_entrypoint()
def main(campaign: str='matrix_campaign'):
    from datetime import datetime,timezone
    from openassembler.transport import runtime_files
    here=Path(__file__).parent;root=here.resolve().parents[1];files=runtime_files(native=True)
    for p in here.glob('*.py'):files['exp/ml_suite/'+p.name]=p.read_bytes()
    for name in ['native_model','native_attention_model','native_tensor_model']:
        files['examples/'+name+'/model.py']=(root/'examples'/name/'model.py').read_bytes()
    dest=here/'runs'/(campaign+'-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'));dest.mkdir()
    for p in here.glob('*.py'):(dest/p.name).write_bytes(p.read_bytes())
    result=execute.remote(files,campaign)
    (dest/'artifacts.zip').write_bytes(result.pop('artifacts'))
    (dest/'result.json').write_text(json.dumps(result,indent=2)+chr(10))
    print(json.dumps({'path':str(dest),'passed':result['result']['passed'],'error':result['result'].get('error')}))
    assert result['result']['passed']

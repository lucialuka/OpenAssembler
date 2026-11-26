"""Frozen OA-only A100 pack/layout/training experiments."""
from pathlib import Path
import json
import modal
app=modal.App('oa-model-perf')
image=modal.Image.from_registry('nvidia/cuda:12.8.1-devel-ubuntu22.04',add_python='3.12').pip_install('torch==2.7.1',index_url='https://download.pytorch.org/whl/cu128').pip_install('numpy','pytest==8.4.2')
@app.function(image=image,gpu='A100-40GB',timeout=1800,retries=0,max_containers=1)
def execute(files,phase):
    import tempfile,subprocess,sys,os,io,zipfile,hashlib
    with tempfile.TemporaryDirectory(prefix='oa-model-perf-') as d:
        root=Path(d)
        for n,b in files.items():
            p=root/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b)
        (root/'env/cuda.sh').chmod(0o700)
        proc=subprocess.run([sys.executable,'-m','exp.model_perf.campaign',phase],cwd=d,env=dict(os.environ,PYTHONPATH=d,OA_PERF_REDUCTIONS='1' if phase in ('reductions','confirm') else ''),capture_output=True,text=True,timeout=1650)
        (root/'results').mkdir(exist_ok=True);(root/'results/worker.log').write_text(proc.stdout+proc.stderr)
        result_path=root/'results/result.json'
        result=json.loads(result_path.read_text()) if result_path.exists() else {'passed':False,'error':(proc.stdout+proc.stderr)[-12000:]}
        if proc.returncode:result={'passed':False,'error':(proc.stdout+proc.stderr)[-12000:]}
        output=io.BytesIO()
        with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED) as z:
            for p in (root/'results').rglob('*'):
                if p.is_file():z.write(p,p.relative_to(root))
        return {'result':result,'artifacts':output.getvalue(),'returncode':proc.returncode,'source_sha256':{n:hashlib.sha256(b).hexdigest() for n,b in files.items()}}
@app.local_entrypoint()
def main(phase:str='probe'):
    from datetime import datetime,timezone
    from openassembler.transport import runtime_files
    files=runtime_files(native=True);files['examples/native_attention_model/model.py']=Path('examples/native_attention_model/model.py').read_bytes();here=Path(__file__).resolve().parent
    files['exp/weaknesses/bf16_verification.py']=Path('exp/weaknesses/bf16_verification.py').read_bytes()
    files['pyproject.toml']=Path('pyproject.toml').read_bytes()
    for p in Path('tests').glob('*.py'):files['tests/'+p.name]=p.read_bytes()
    if phase=='allocator-build':
        for p in Path('tests/data').rglob('*'):
            if p.is_file() and p.suffix in ('.json','.ptx','.py','.cu','.h'):
                files[p.as_posix()]=p.read_bytes()
    for p in Path('exp/ml_suite').glob('*.py'):files['exp/ml_suite/'+p.name]=p.read_bytes()
    for name in ['native_model','native_tensor_model','native_attention_model']:
        files['examples/'+name+'/model.py']=Path('examples',name,'model.py').read_bytes()
    for p in here.glob('*.py'):files['exp/model_perf/'+p.name]=p.read_bytes()
    dest=here/'runs'/(phase+'-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'));dest.mkdir(parents=True)
    for p in here.glob('*.py'):(dest/p.name).write_bytes(p.read_bytes())
    result=execute.remote(files,phase);(dest/'artifacts.zip').write_bytes(result.pop('artifacts'))
    (dest/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'path':str(dest),'passed':result['result']['passed'],'error':result['result'].get('error')}));assert result['result']['passed']

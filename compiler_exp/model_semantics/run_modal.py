"""Frozen candidate/production A100 gate with raw failures and artifacts retained."""
from datetime import datetime,timezone
import base64
import hashlib
import json
from pathlib import Path
import modal
ROOT=Path.cwd();HERE=Path(__file__).resolve().parent
app=modal.App('oa-model-semantics')
image=modal.Image.from_registry('pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel').apt_install('git').pip_install('pytest==8.4.2')

@app.function(image=image,gpu='A100-40GB',timeout=1800,max_containers=1,retries=0)
def execute(payload,tests,action):
    import io,os,subprocess,sys,tempfile,zipfile
    with tempfile.TemporaryDirectory(prefix='oa-semantics-') as tmp:
        root=Path(tmp)
        for name,data in {**payload,**tests}.items():
            path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
        (root/'env/cuda.sh').chmod(0o700)
        command=[sys.executable,'-c',"import json, pathlib;from gate import run;r,a=run(pathlib.Path('work'));pathlib.Path('report.json').write_text(json.dumps(r));pathlib.Path('artifacts.zip').write_bytes(a)"]
        if action=='cli':
            command=[sys.executable,'-c',"import json,pathlib;from cli_gate import run;r=run(pathlib.Path.cwd());pathlib.Path('report.json').write_text(json.dumps(r))"]
        if action=='tests':
            command=[sys.executable,'-m','pytest','-q',*[n for n in tests if n.startswith('tests/test_')]]
        (root/'work').mkdir()
        done=subprocess.run(command,cwd=root,env=dict(os.environ,PYTHONPATH=str(root)),capture_output=True,text=True,timeout=1700)
        result={'returncode':done.returncode,'stdout':done.stdout,'stderr':done.stderr,'report':json.loads((root/'report.json').read_text()) if (root/'report.json').exists() else None}
        if action=='tests':
            result['report']={'passed':done.returncode==0,'torch':__import__('torch').__version__}
        if (root/'artifacts.zip').exists():artifact=(root/'artifacts.zip').read_bytes()
        else:
            stream=io.BytesIO()
            with zipfile.ZipFile(stream,'w',zipfile.ZIP_DEFLATED) as z:
                for p in (root/'work').rglob('*'):
                    if p.is_file() and p.suffix in ('.json','.ptx','.cubin','.pt'):z.write(p,p.relative_to(root))
                for p in (root/'cli-output').rglob('*'):
                    if p.is_file():z.write(p,p.relative_to(root))
            artifact=stream.getvalue()
        return result,base64.b64encode(artifact).decode()

@app.local_entrypoint()
def main(production:bool=False, action:str='hardware'):
    import sys
    sys.path.insert(0,str(ROOT))
    from openassembler.transport import runtime_files
    payload=runtime_files(native=True)
    if not production:
        base=HERE/'overlay'
        for p in (base/'openassembler').rglob('*.py'):payload[p.relative_to(base).as_posix()]=p.read_bytes()
    tests={n:(HERE/n).read_bytes() for n in ('gate.py','decode_gate.py','test_semantics.py','optimizer_bench.py','cli_gate.py','loss_gate.py')}
    tests.update({'repo/'+p.name:p.read_bytes() for p in (HERE/'repo').iterdir() if p.is_file()})
    if action=='tests':
        names=['tests/__init__.py','tests/backend_harness.py']+['tests/test_model_'+n+'.py' for n in ('graph','frontend','runtime','optimization','tensor','batched_tensor','kernels','training_kernels','semantics')]
        tests.update({n:(ROOT/n).read_bytes() for n in names})
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ');prefix='production' if production else 'candidate'
    result,blob=execute.remote(payload,tests,action);raw=base64.b64decode(blob)
    artifact=HERE/(prefix+'-'+action+'-'+stamp+'.zip');artifact.write_bytes(raw)
    result.update(compiler_hashes={n:hashlib.sha256(v).hexdigest() for n,v in payload.items()},
                  harness_hashes={n:hashlib.sha256(v).hexdigest() for n,v in tests.items()},
                  archive={'path':artifact.name,'sha256':hashlib.sha256(raw).hexdigest()})
    path=HERE/(prefix+'-'+action+'-'+stamp+'.json');path.write_text(json.dumps(result,indent=2)+'\n')
    print(path,result['returncode']);print(result['stdout']);print(result['stderr'])
    if result['returncode'] or not result.get('report',{}).get('passed'):raise ValueError('semantics gate failed')

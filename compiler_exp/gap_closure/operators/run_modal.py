"""Run the current production G21 implementation in a supported A100 environment."""
from datetime import datetime,timezone
import base64
import hashlib
import json
from pathlib import Path
import modal

ROOT=Path.cwd()
HERE=Path(__file__).resolve().parent
app=modal.App('oa-gap-model-operators')
image=modal.Image.from_registry('nvidia/cuda:12.8.1-devel-ubuntu22.04',add_python='3.12').apt_install('git').pip_install('torch==2.7.1','pytest==8.4.2')

@app.function(image=image,gpu='A100-40GB',timeout=1800,retries=0,max_containers=1)
def execute(payload,harness):
    import io,os,subprocess,sys,tempfile,zipfile
    with tempfile.TemporaryDirectory(prefix='oa-g25-') as tmp:
        root=Path(tmp)
        for name,data in {**payload,**harness}.items():
            p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
        (root/'env/cuda.sh').chmod(0o700)
        command=[sys.executable,'-c',"from hardware_gate import run;run('work')"]
        done=subprocess.run(command,cwd=root,env=dict(os.environ,PYTHONPATH=str(root)),capture_output=True,text=True,timeout=1700)
        result={'returncode':done.returncode,'stdout':done.stdout,'stderr':done.stderr,
                'report':json.loads((root/'work/report.json').read_text()) if (root/'work/report.json').exists() else None}
        stream=io.BytesIO()
        with zipfile.ZipFile(stream,'w',zipfile.ZIP_DEFLATED) as z:
            for p in root.rglob('*'):
                if p.is_file() and (p.suffix in ('.json','.ptx','.cubin','.pt') or p.name=='hardware_gate.py'):
                    z.write(p,p.relative_to(root))
        return result,base64.b64encode(stream.getvalue()).decode()

@app.local_entrypoint()
def main():
    import sys
    sys.path.insert(0,str(ROOT))
    from openassembler.transport import runtime_files
    payload=runtime_files(native=True);harness={'hardware_gate.py':(HERE/'hardware_gate.py').read_bytes()}
    harness.update({'repo/'+p.name:p.read_bytes() for p in (HERE/'repo').iterdir() if p.is_file()})
    result,blob=execute.remote(payload,harness)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    archive=HERE/('hardware-'+stamp+'.zip');archive.write_bytes(base64.b64decode(blob))
    result.update(compiler_hashes={n:hashlib.sha256(v).hexdigest() for n,v in payload.items()},
                  harness_sha256={n:hashlib.sha256(v).hexdigest() for n,v in harness.items()},
                  archive={'path':archive.name,'sha256':hashlib.sha256(archive.read_bytes()).hexdigest()})
    report=HERE/('hardware-'+stamp+'.json');report.write_text(json.dumps(result,indent=2)+'\n')
    print(report);print(result['stdout']);print(result['stderr'])
    if result['returncode'] or not result.get('report',{}).get('passed'):raise RuntimeError('G21 hardware gate failed')

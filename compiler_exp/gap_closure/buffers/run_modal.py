"""Run pure structured device helpers through ordinary compile on A100."""
from datetime import datetime,timezone
import base64
import hashlib
import json
from pathlib import Path
import modal

ROOT=Path.cwd();HERE=Path(__file__).resolve().parent
app=modal.App('oa-checked-buffers')
image=modal.Image.from_registry('nvidia/cuda:12.8.1-devel-ubuntu22.04',add_python='3.12')

@app.function(image=image,gpu='A100-40GB',timeout=600,retries=0,max_containers=1)
def execute(files):
    import io,os,subprocess,sys,tempfile,zipfile
    with tempfile.TemporaryDirectory(prefix='oa-helper-control-') as d:
        root=Path(d)
        for name,data in files.items():
            path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
        (root/'env/cuda.sh').chmod(0o700)
        command=[sys.executable,'-c',"from compiler_exp.gap_closure.buffers.hardware_gate import run;run('work')"]
        done=subprocess.run(command,cwd=root,env=dict(os.environ,PYTHONPATH=str(root)),capture_output=True,text=True,timeout=550)
        result={'returncode':done.returncode,'stdout':done.stdout,'stderr':done.stderr,
                'report':json.loads((root/'work/report.json').read_text()) if (root/'work/report.json').exists() else None}
        stream=io.BytesIO()
        with zipfile.ZipFile(stream,'w',zipfile.ZIP_DEFLATED) as z:
            for path in root.rglob('*'):
                if path.is_file() and (path.suffix in ('.json','.ptx','.cubin','.py') or path.name.endswith('.sass')):z.write(path,path.relative_to(root))
        return result,base64.b64encode(stream.getvalue()).decode()

@app.local_entrypoint()
def main():
    import sys
    sys.path.insert(0,str(ROOT))
    from openassembler.transport import runtime_files
    files=runtime_files(native=True)
    for name in ('fixture.py','hardware_gate.py','probe.py'):
        relative=Path('compiler_exp/gap_closure/buffers')/name
        files[str(relative)]=(ROOT/relative).read_bytes()
    result,blob=execute.remote(files)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    archive=HERE/('hardware-'+stamp+'.zip');archive.write_bytes(base64.b64decode(blob))
    result.update(source_hashes={n:hashlib.sha256(v).hexdigest() for n,v in files.items()},
                  harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  archive={'path':archive.name,'sha256':hashlib.sha256(archive.read_bytes()).hexdigest()})
    target=HERE/('hardware-'+stamp+'.json');target.write_text(json.dumps(result,indent=2)+'\n')
    print(target);print(result['stdout']);print(result['stderr'])
    if result['returncode'] or not result.get('report',{}).get('passed'):raise RuntimeError('Wide integer frontend hardware gate failed')

"""Stage an original build database through the ordinary native transport worker."""
from datetime import datetime,timezone
import base64
import hashlib
import json
from pathlib import Path
import tempfile
import modal
ROOT=Path.cwd();HERE=Path(__file__).resolve().parent
app=modal.App('oa-build-command-graph')
image=modal.Image.from_registry('nvidia/cuda:12.8.1-devel-ubuntu22.04',add_python='3.12').apt_install('git')

@app.function(image=image,gpu='A100-40GB',timeout=900,retries=0,max_containers=1)
def execute(runtime,payload,harness):
    import io,os,sys,subprocess,zipfile
    with tempfile.TemporaryDirectory(prefix='oa-build-graph-') as temporary:
        root=Path(temporary)
        for name,raw in {**runtime,'hardware_gate.py':harness}.items():
            path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
        (root/'payload.json').write_text(json.dumps({n:base64.b64encode(v).decode() for n,v in payload.items()}))
        (root/'env/cuda.sh').chmod(0o700)
        code="import json,base64; from pathlib import Path; from hardware_gate import run; p={n:base64.b64decode(v) for n,v in json.loads(Path('payload.json').read_text()).items()};run(p,'work')"
        completed=subprocess.run([sys.executable,'-c',code],cwd=root,env=dict(os.environ,PYTHONPATH=str(root)),capture_output=True,text=True,timeout=850)
        result={'returncode':completed.returncode,'stdout':completed.stdout,'stderr':completed.stderr,
            'report':json.loads((root/'work/report.json').read_text()) if (root/'work/report.json').is_file() else None}
        blob=io.BytesIO()
        with zipfile.ZipFile(blob,'w',zipfile.ZIP_DEFLATED) as archive:
            for path in root.rglob('*'):
                if path.is_file() and ('work' in path.parts or path.name in ('payload.json','hardware_gate.py')):
                    archive.write(path,path.relative_to(root))
        return result,base64.b64encode(blob.getvalue()).decode()

@app.local_entrypoint()
def main():
    import sys
    sys.path.insert(0,str(ROOT))
    from openassembler.build import Options
    from openassembler.remote_build import _sources
    from openassembler.transport import runtime_files
    with tempfile.TemporaryDirectory(prefix='oa-build-origin-',dir=ROOT/'corpus/.tmp') as temporary:
        root=Path(temporary).resolve();(root/'build').mkdir();(root/'include').mkdir()
        (root/'.gitignore').write_text('build/\n')
        (root/'include/qualifier.h').write_text('#define OA_GLOBAL __global__\nstatic constexpr unsigned HEADER_NAME_SIZE=sizeof(__FILE__);\n')
        rows=[]
        for name,scale,bias in [('three',3,5),('negative',-2,7)]:
            (root/'build'/('config_'+name+'.h')).write_text(f'#define BASE {bias}\n')
            (root/'build'/('options_'+name+'.rsp')).write_text(f'-I ../include -include config_{name}.h -DSCALE={scale}')
            source=root/'build'/('generated_'+name+'.cu')
            source.write_text('#include "qualifier.h"\nextern "C" OA_GLOBAL void configured_'+name+'(float*out,const float*x,int n) { int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) out[i]=x[i]*SCALE+BASE+HEADER_NAME_SIZE+sizeof(__FILE__); }\n')
            rows.append({'directory':str(root/'build'),'file':source.name,
                'arguments':['nvcc','--options-file','options_'+name+'.rsp','-arch=sm_80','-c',source.name,'-o',name+'.o']})
        (root/'compile_commands.json').write_text(json.dumps(rows))
        _,payload,_=_sources(root,root/'output',Options())
        runtime=runtime_files(native=True)
        compiler_hashes={n:hashlib.sha256(v).hexdigest() for n,v in runtime.items()}
        result,blob=execute.remote(runtime,payload,(HERE/'hardware_gate.py').read_bytes())
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ');archive=HERE/('a100-'+stamp+'.artifacts.zip');archive.write_bytes(base64.b64decode(blob))
    result['archive']={'path':archive.name,'sha256':hashlib.sha256(archive.read_bytes()).hexdigest()}
    result['compiler_hashes']=compiler_hashes
    result['harness_sha256']=hashlib.sha256((HERE/'hardware_gate.py').read_bytes()).hexdigest()
    path=HERE/('a100-'+stamp+'.json');path.write_text(json.dumps(result,indent=2)+'\n')
    print(path);print(result['stdout']);print(result['stderr'])
    if result['returncode'] or not result.get('report',{}).get('passed'):raise RuntimeError('build database A100 gate failed')

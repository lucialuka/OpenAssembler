"""Retain repeated real cold/hit attribution and binary identities, without GPU timing."""
import hashlib
import json
from pathlib import Path
import platform
import statistics
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from openassembler.modelfront.kernels import build_bundle
from openassembler.modelfront.runtime import validate_bundle


def plan():
    specs={name:{'shape':[7],'dtype':'f32'} for name in ('x','y','seed','dx')}
    return {'schema':1,'inputs':[{'name':'x',**specs['x']}],'parameters':[], 'buffers':specs,
            'steps':[{'kernel':'relu','inputs':['x'],'output':'y','attrs':{}}],'outputs':['y'],
            'backward':{'steps':[{'kernel':'relu_backward','inputs':['seed','x'],'output':'dx','attrs':{}}],
                        'seed_grads':{'y':'seed'},'grad_inputs':{'x':'dx'},'grad_parameters':{}}}


def run():
    rows=[]
    for trial in range(3):
        with tempfile.TemporaryDirectory(prefix='oa-phase-experiment-') as temporary:
            base=Path(temporary)
            for kind in ('cold','hit'):
                output=base/kind
                build_bundle(plan(),{},output,cache_dir=base/'cache',sass_listings=False)
                record=json.loads((output/'model-cache-result.json').read_text())
                _,ledger,binary=validate_bundle(output)
                rows.append({'trial':trial,'kind':kind,'record':record,
                             'native_cubin_sha256':hashlib.sha256(binary).hexdigest()})
    hashes={row['native_cubin_sha256'] for row in rows}
    passed=(len(hashes)==1 and all(row['record']['hit']==(row['kind']=='hit') and
               bool(row['record']['source_builds'])==(row['kind']=='cold') for row in rows))
    report={'schema':1,'passed':passed,'plan':plan(),'python':sys.version,'platform':platform.platform(),
            'scope':'three fresh cache directories followed by validated hits; worker wall attribution only; no execution, device, transfer or speedup claim',
            'samples':rows,'compiler_sha256':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
                ('openassembler/build.py','openassembler/modelfront/model_cache.py','openassembler/modelfront/worker.py')}}
    output=ROOT/'compiler_exp/gap_closure/measurement/cache-results.json'
    with output.open('x') as stream:json.dump(report,stream,indent=2);stream.write('\n')
    print(json.dumps({'passed':passed,'samples':len(rows),'unique_cubins':len(hashes)}))
    if not passed:raise SystemExit(1)


if __name__=='__main__':run()

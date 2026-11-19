"""Audit supplied coverage evidence without relabeling CPU work as GPU work."""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path


def audit(source):
    raw=Path(source).read_bytes();value=json.loads(raw);cases=value['cases']
    errors=[];timings={};rows=[]
    for case in cases:
        accepted=case['oa_status']=='passed'
        if accepted!=case['expected_supported']:errors.append(case['name']+': unexpected admission')
        phase_samples=case.get('timings',{})
        if accepted:
            for phase in ('capture_fx_and_autodiff_ms','storage_planning_ms','prepare_emit_ptx_ms'):
                samples=phase_samples.get(phase,[])
                if len(samples)!=7 or any(type(x) not in (int,float) or not math.isfinite(x) or x<0 for x in samples):
                    errors.append(case['name']+': invalid samples '+phase)
                timings[phase]=timings.get(phase,0)+len(samples)
        elif phase_samples:errors.append(case['name']+': rejected row has timing data')
        rows.append({'name':case['name'],'expected_supported':case['expected_supported'],
                     'oa_status':case['oa_status'],'torch_eager_cpu':case['torch_eager_cpu'],
                     'math':case['math'],'packing':case['packing'],'inputs':case['inputs']})
    return {'schema':1,'source_sha256':hashlib.sha256(raw).hexdigest(),'environment':value['environment'],
            'scope':'Audit of supplied frozen CPU capture/autodiff/planning/PTX-preparation records; not a fresh implementation or hardware benchmark',
            'cases':len(cases),'counts':dict(Counter(c['oa_status'] for c in cases)),
            'unexpected_admission_errors':errors,'raw_cpu_samples':timings,'rows':rows,
            'limits':['Input JSON does not include the generator/test source or model factory identities',
                      'Input JSON alone does not establish complete forward/dX/dW/SGD execution checks',
                      'Seven repetitions include cold first samples; no hidden exclusion or pooled GPU timing',
                      'Unsupported rows remain outside accepted timing cohorts']}


if __name__=='__main__':
    here=Path(__file__).parent;result=audit(here/'input-coverage.json')
    (here/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(result['counts'],result['raw_cpu_samples'],'errors',result['unexpected_admission_errors'])
    if result['unexpected_admission_errors']:raise SystemExit(1)

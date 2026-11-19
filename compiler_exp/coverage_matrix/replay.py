"""New source-defined boundary corpus derived from supplied shape metadata.

The original factory source was not supplied, so this is explicitly a new
reproducible corpus, not a reproduction of its plan hashes or state checks.
"""
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import torch
from openassembler.modelfront.graph import capture_model
from openassembler.modelfront.optimization import optimize_plan
from openassembler.modelfront.kernels import prepare


class Operation(torch.nn.Module):
    def __init__(self,kind):super().__init__();self.kind=kind
    def forward(self,x):
        if self.kind=='concat':return torch.cat([x,x],dim=-1)
        if self.kind=='permute':return x.permute(0,2,1).contiguous()
        if self.kind=='sigmoid':return torch.sigmoid(x)
        if self.kind=='sum':return x.sum(dim=-1)
        if self.kind=='slice':return x[:,:-1]
        if self.kind=='divide':return x/2
        return torch.relu(x)


class BatchedProduct(torch.nn.Module):
    def forward(self,a,b):return torch.bmm(a,b)


def make(case):
    name=case['name'];shape=case['inputs'][0]['shape'];torch.manual_seed(19119)
    inputs=[torch.empty_strided(meta['shape'],meta['stride'],dtype=getattr(torch,meta['dtype'].split('.')[-1])).normal_()
            for meta in case['inputs']]
    if name=='reject_nan':inputs[0].view(-1)[0]=float('nan')
    if name.startswith('linear_dual_'):
        model=torch.nn.Linear(shape[-1],case['eager_output_shape'][-1])
    elif name.startswith('linear_'):
        model=torch.nn.Sequential(torch.nn.Linear(shape[-1],7),torch.nn.ReLU(),torch.nn.Linear(7,3))
    elif name.startswith('softmax_'):model=torch.nn.Softmax(dim=-1)
    elif name.startswith('layernorm_'):model=torch.nn.LayerNorm(shape[-1])
    elif name.startswith('gelu_'):model=torch.nn.GELU(approximate='tanh')
    elif name=='reject_exact_gelu':model=torch.nn.GELU()
    elif name=='reject_dropout':model=torch.nn.Dropout(.25)
    elif name.startswith('bmm_'):model=BatchedProduct()
    else:
        kind=name.split('_')[0]
        if name.startswith('reject_'):kind=name.removeprefix('reject_')
        model=Operation(kind)
    return model,inputs


if __name__=='__main__':
    torch.set_num_threads(1)
    here=Path(__file__).parent;raw=(here/'input-coverage.json').read_bytes();cases=json.loads(raw)['cases'];rows=[]
    for case in cases:
        model,inputs=make(case);row={'name':case['name'],'expected_supported':case['expected_supported'],'samples':[]}
        try:
            for repetition in range(7):
                start=time.perf_counter();plan=capture_model(model,inputs,math_mode=case['math'])
                plan['tensor_packing']=case['packing'];capture_ms=1000*(time.perf_counter()-start)
                snapshot=copy.deepcopy(plan);combinations=[];storage_ms=emit_ms=0
                for views,reuse in ((False,False),(False,True),(True,False),(True,True)):
                    start=time.perf_counter();optimized=optimize_plan(plan,storage_views=views,reuse_storage=reuse)
                    storage_ms+=1000*(time.perf_counter()-start)
                    start=time.perf_counter();specs,steps,kernels=prepare(optimized)
                    emit_ms+=1000*(time.perf_counter()-start)
                    assert plan==snapshot,'storage/preparation mutated input plan'
                    combinations.append({'views':views,'reuse':reuse,'steps':len(steps),'kernels':len(kernels)})
                row['samples'].append({'capture_autodiff_ms':capture_ms,'all_four_storage_combinations_ms':storage_ms,
                                       'all_four_prepare_combinations_ms':emit_ms})
                row['storage_combinations']=combinations
            row['status']='accepted'
        except Exception as exc:
            row.update(status='rejected',error_type=type(exc).__name__,error=str(exc))
            row['samples_before_rejection']=row.pop('samples')
        row['expected_decision']=(row['status']=='accepted')==case['expected_supported'];rows.append(row)
    report={'schema':1,'scope':'New source-defined CPU capture/autodiff/storage/PTX-preparation corpus; no GPU execution or state-equivalence claim',
            'torch':torch.__version__,'python':sys.version,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'input_sha256':hashlib.sha256(raw).hexdigest(),'cases':rows,
            'passed':all(r['expected_decision'] for r in rows)}
    target=here/('replay-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'.json')
    target.write_text(json.dumps(report,indent=2)+'\n')
    print(target,'accepted',sum(r['status']=='accepted' for r in rows),'rejected',sum(r['status']=='rejected' for r in rows),'passed',report['passed'])
    if not report['passed']:raise SystemExit(1)

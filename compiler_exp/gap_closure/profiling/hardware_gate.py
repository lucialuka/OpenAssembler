"""Validate ordinary worker profiling without inserting extra optimizer updates."""
import base64
import hashlib
import json
from pathlib import Path
import sys
import torch

from openassembler.modelfront.frontend import load_config
from openassembler.modelfront.worker import process
from openassembler.torchfront.archive import pack,unpack

SOURCE='''import torch
class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.first=torch.nn.Linear(3,5)
        self.second=torch.nn.Linear(5,2)
    def forward(self,x):
        return self.second(torch.relu(self.first(x)))
def create_model():
    return Net()
'''


def run(destination):
    root=Path(destination);root.mkdir(parents=True,exist_ok=True)
    reports=[]
    for name,mode,execution,repetitions in [('inference','inference','direct',3),('training','training','direct',3),
            ('graph','training','cuda_graph',3),('single','inference','direct',1),('unverified','inference','direct',3)]:
        source=root/name/'source';source.mkdir(parents=True)
        (source/'model.py').write_text(SOURCE)
        (source/'openassembler.toml').write_text('[model]\nentrypoint="model.py"\ninput_shapes=[[7,3]]\n'
            f'mode="{mode}"\nexecution="{execution}"\nsteps=4\nprofile_iterations={repetitions}\nsass_listings=false\n')
        config=load_config(source)
        payload=pack({p.name:p.read_bytes() for p in source.iterdir()})
        response=process({'action':'compile','config':config,'source':base64.b64encode(payload).decode(),
            'source_digest':hashlib.sha256(payload).hexdigest(),'verify':name!='unverified'})
        if 'error' in response:raise ValueError(response['error'])
        report=response['report'];report_path=root/name/'report.json';report_path.write_text(json.dumps(report,indent=2)+'\n')
        unpack(base64.b64decode(response['artifacts']),root/name/'artifacts')
        if name=='unverified':
            run_response=process({'action':'run','config':config,'artifacts':response['artifacts'],'verify':False})
            if 'error' in run_response:raise ValueError(run_response['error'])
            report=run_response['report'];(root/name/'unverified-run.json').write_text(json.dumps(report,indent=2)+'\n')
        assert report['passed'] and report['iterations']==4
        sampled=report['profiled_iterations']
        assert sampled==([] if name=='unverified' else list(range(4-repetitions,4)))
        assert len(report['host_step_ms'])==len(report['cuda_stream_elapsed_ms'])==4
        assert report['profiler_free_iterations']==[i for i in range(4) if i not in sampled]
        assert len(report['profiler_free_host_step_ms'])==4-len(sampled)
        profiles=report['kernel_profiles']
        assert profiles['profiled_iterations']==sampled
        assert sum(row['launch_count'] for row in profiles['kernels'])==sum(a['observed_compute_launches'] for a in report['profiler_native_audits'])
        for row in profiles['kernels']:
            assert row['profiled_invocations']==repetitions
            assert row['repeated_invocations']==(repetitions>1)
            resources=row['resources']
            assert resources['driver']['binary_version']==80 and resources['driver']['num_regs']>0
            assert resources['driver']['local_bytes']>=0 and resources['module_cubin_bytes']>=resources['kernel_text_bytes']>0
            assert resources['module_sass_bytes'] is None
        if name!='unverified':
            assert report['checks'] and all(check['passed'] for check in report['checks'])
            assert all(g['passed'] if isinstance(g,dict) and 'passed' in g else bool(g) for g in report['canaries'])
        reports.append({'name':name,'passed':True,'checks':len(report['checks']),
            'profiled_iterations':sampled,'unique_contracts':len(profiles['kernels']),
            'audited_launches':profiles['compute_launches'],'execution':execution})
    summary={'schema':1,'passed':all(row['passed'] for row in reports),'results':reports,
        'python':sys.version,'torch':torch.__version__,'cuda_runtime':torch.version.cuda,
        'device':torch.cuda.get_device_name(0),'scope':'correctness/ownership/accounting of repeated profiler observations through ordinary worker; no performance comparator or profiler-free per-kernel benchmark'}
    (root/'report.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))

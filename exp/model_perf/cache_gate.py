"""Real binary-only miss/hit without nvdisasm, corruption rejection unchanged."""
from pathlib import Path
from unittest.mock import patch
import json,time
import torch
from openassembler import oracle
from openassembler.modelfront.model_cache import toolchain_identity,cached_build_bundle
from openassembler.modelfront.graph import capture_model,extract_weights
from openassembler.modelfront.runtime import validate_bundle,load_weights
from exp.ml_suite.tasks import create


def main():
    torch.manual_seed(712);torch.set_num_threads(1)
    m=create('dense_regression_fp16');p=capture_model(m,[torch.randn(128,128)*.25],math_mode='f16_rne_f32_accum');w=extract_weights(m,p)
    root=Path('results/cache');store=root/'store';calls=[];original=oracle.run_tool
    def tool(name,*args,**kw):
        calls.append(name)
        if name=='nvdisasm':raise FileNotFoundError('nvdisasm absent negative gate')
        return original(name,*args,**kw)
    toolchain_identity.cache_clear();records=[]
    with patch.object(oracle,'run_tool',tool):
        for name in ['miss','hit']:
            t=time.perf_counter();result=cached_build_bundle(p,w,root/name,cache_dir=store,sass_listings=False);elapsed=(time.perf_counter()-t)*1000
            assert result['hit']==(name=='hit');validate_bundle(root/name)
            records.append({'name':name,'ms':elapsed,'hit':result['hit']})
        changed={n:v+.03125 for n,v in w.items()}
        with patch('openassembler.modelfront.kernels._build_bundle_uncached',side_effect=AssertionError('cache hit compiled code')):
            result=cached_build_bundle(p,changed,root/'changed',cache_dir=store,sass_listings=False)
        assert result['hit'];actual=load_weights(root/'changed',p,torch,device='cpu');assert all(torch.equal(actual[n],changed[n]) for n in changed)
        binary=next(store.rglob('native/cubin/*.cubin'));data=bytearray(binary.read_bytes());data[-1]^=1;binary.write_bytes(data)
        result=cached_build_bundle(p,w,root/'repaired',cache_dir=store,sass_listings=False);assert not result['hit'];validate_bundle(root/'repaired')
    assert 'nvdisasm' not in calls and not list(store.rglob('reference.cubin'))
    Path('results/result.json').write_text(json.dumps({'passed':True,'records':records,'tool_calls':calls,'changed_weights':True,'corrupt_hit_rebuilt':True,'no_nvdisasm':True,'no_donors':True},indent=2))

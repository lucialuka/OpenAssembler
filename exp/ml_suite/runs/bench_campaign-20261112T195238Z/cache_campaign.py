"""Exact warm-build cache pairs, changed weights and rejected corrupted hits."""
import json
from pathlib import Path
import runpy
import time
import torch
from openassembler.modelfront.graph import capture_model,extract_weights
from openassembler.modelfront.optimization import optimize_plan
from openassembler.modelfront.kernels import _build_bundle_uncached
from openassembler.modelfront.runtime import validate_bundle,load_weights
from openassembler.modelfront.model_cache import cached_build_bundle,key_for


def main():
    torch.manual_seed(124)
    root=Path('results/cache');root.mkdir(parents=True,exist_ok=True);records=[]
    for case,shape,math in [('native_model',(17,8),'fp32'),('native_tensor_model',(128,128),'f16_rne_f32_accum')]:
        model=runpy.run_path('examples/'+case+'/model.py')['create_model']()
        plan=optimize_plan(capture_model(model,[torch.randn(shape)*.25],math_mode=math))
        weights=extract_weights(model,plan);folder=root/case;cache=folder/'store'
        cold=cached_build_bundle(plan,weights,folder/'populate',cache_dir=cache,sass_listings=False)
        assert not cold['hit']
        rounds=[]
        for index in range(3):
            values={}
            for name in (['cold','hit'] if index%2==0 else ['hit','cold']):
                out=folder/f'{index}-{name}';before=time.perf_counter()
                if name=='cold':_build_bundle_uncached(plan,weights,out,sass_listings=False)
                else:assert cached_build_bundle(plan,weights,out,cache_dir=cache,sass_listings=False)['hit']
                values[name]=(time.perf_counter()-before)*1000
                validate_bundle(out)
            a=(folder/f'{index}-cold/native-ledger.json').read_bytes()
            b=(folder/f'{index}-hit/native-ledger.json').read_bytes()
            assert a==b,'cached executable inventory changed'
            rounds.append({'ms':values,'speedup':values['cold']/values['hit']})
        changed={n:value+.03125 for n,value in weights.items()}
        assert cached_build_bundle(plan,changed,folder/'new-weights',cache_dir=cache,sass_listings=False)['hit']
        actual=load_weights(folder/'new-weights',plan,torch,device='cpu')
        assert all(torch.equal(actual[n],changed[n]) for n in actual)
        key,_=key_for(plan,sass_listings=False)
        record=json.loads((cache/key/'cache-record.json').read_text())
        binary=next(n for n in record['files'] if n.startswith('native/cubin/') and n.endswith('.cubin'))
        p=cache/key/binary;data=bytearray(p.read_bytes());data[-1]^=1;p.write_bytes(data)
        repair=cached_build_bundle(plan,weights,folder/'repaired',cache_dir=cache,sass_listings=False)
        assert not repair['hit'] and repair['rebuild_reason']
        assert (folder/'repaired/native-ledger.json').read_bytes()==(folder/'populate/native-ledger.json').read_bytes()
        assert not list(cache.rglob('reference.cubin')) and not list(cache.rglob('.build'))
        assert not list((folder/'repaired').rglob('reference.cubin')) and not list((folder/'repaired').rglob('.build'))
        records.append({'case':case,'rounds':rounds,'changed_weights_hit':True,'corrupted_hit_rebuilt':True,'no_reference_cubins':True,
                        'key':key,'artifact_validation':True})
    (root/'result.json').write_text(json.dumps({'passed':True,'records':records,
        'scope':'fresh-output native bundle build versus validated persistent cache; excludes CPU model capture and transport; cached weights forbidden'},indent=2))


if __name__=='__main__':main()

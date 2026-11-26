"""Full attention/cache transitions, failed builders, guards and native provenance."""
import hashlib
import json
from pathlib import Path
import platform
import torch

from openassembler.modelfront.decode import DecodeSession
from openassembler.modelfront.graph import capture_model, extract_weights
from openassembler.modelfront.kernels import build_bundle
from openassembler.modelfront.optimization import optimize_plan
from openassembler.modelfront.profiling import audit_trace
from openassembler.modelfront.runtime import Runtime


class Prefill(torch.nn.Module):
    def __init__(self):
        super().__init__();self.scale=torch.nn.Parameter(torch.tensor([0.75]))
    def forward(self,x):
        q=x*self.scale
        return torch.nn.functional.scaled_dot_product_attention(q,q,q,is_causal=True),q.clone(),q.clone()


class Decoder(Prefill):
    def forward(self,x,past_k,past_v):
        q=x*self.scale;k=torch.cat((past_k,q),dim=-2);v=torch.cat((past_v,q),dim=-2)
        return torch.nn.functional.scaled_dot_product_attention(q,k,v),k,v


def run(directory):
    root=Path(directory);root.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1);torch.manual_seed(39052)
    torch.backends.cuda.matmul.allow_tf32=False
    comparisons=[];raw={};records=[];requests=[];all_runtimes=[];failures=[]
    def check(actual,expected,label):
        actual=actual.detach().cpu();expected=expected.detach().cpu()
        close=torch.isclose(actual,expected,rtol=1e-4,atol=1e-5)
        comparisons.append(dict(name=label,elements=actual.numel(),passed=bool(close.all()),
                                max_abs=float((actual-expected).abs().max())))
        raw[label]={'actual':actual,'expected':expected}
        assert comparisons[-1]['passed'],comparisons[-1]
    def make(model,args,name,views,reuse):
        plan=optimize_plan(capture_model(model,args),storage_views=views,reuse_storage=reuse)
        target=root/name;build_bundle(plan,extract_weights(model,plan),target)
        runtime=Runtime.load(target);all_runtimes.append(runtime)
        return runtime
    def audited(runtime,callback,label):
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as profiler:
            result=callback();torch.cuda.synchronize()
        trace=root/(label+'.trace.json');profiler.export_chrome_trace(str(trace))
        runtime=runtime() if callable(runtime) else runtime
        audit=audit_trace(json.loads(trace.read_text()),runtime.launch_ledger)
        assert audit['passed'],audit
        records.append(dict(name=label,audit=audit,guards=runtime.check_canaries(),launches=list(runtime.launch_ledger)))
        return result
    try:
        for views,reuse in ((False,False),(False,True),(True,False),(True,True)):
            suffix=f'{int(views)}{int(reuse)}'
            model=Prefill();prefix=torch.randn(2,2,3,5)*.25
            prefill=make(model,(prefix,),f'prefill-{suffix}',views,reuse)
            built={};calls=[]
            def factory(length,contract):
                requests.append(json.loads(json.dumps(contract)));calls.append(length)
                expected_shapes=[item['shape']for item in contract['input_specs']+contract['cache_specs']]
                args=tuple(torch.zeros(shape)for shape in expected_shapes)
                decoder=Decoder();decoder.scale.data.fill_(9)
                runtime=make(decoder,args,f'decode-{suffix}-{length}',views,reuse)
                built[length]=runtime
                return runtime
            session=DecodeSession(prefill,{},specialization=dict(schema=1,factory=factory,max_context=7,
                max_build_attempts=4,artifact_key='attention-cache-'+suffix+'-v1'))
            try:
                for repetition in range(2):
                    source=prefix+(repetition*.03125)
                    prediction=audited(prefill,lambda:session.prefill({'x':source.cuda()}),f'{suffix}-{repetition}-prefill')
                    expected=model(source)
                    check(prediction,expected[0],f'{suffix}/{repetition}/prefill/prediction')
                    for i,cache in enumerate(session.cache):check(cache,expected[i+1],f'{suffix}/{repetition}/prefill/cache{i}')
                    full=source
                    for length in range(3,7):
                        token=torch.randn(2,2,1,5)*.25;full=torch.cat((full,token),dim=-2)
                        token_gpu=token.cuda()
                        old_cache=session.cache
                        immutable=[t.detach().cpu().clone()for t in (*old_cache,token_gpu)]
                        if length in built:built[length].poison_workspace()
                        prediction=audited(lambda:built[length],lambda:session.step({'x':token_gpu}),f'{suffix}-{repetition}-{length}')
                        assert all(torch.equal(t.detach().cpu(),original)for t,original in zip((*old_cache,token_gpu),immutable))
                        q=full*model.scale
                        expected_prediction=torch.nn.functional.scaled_dot_product_attention(q,q,q,is_causal=True)[:,:,-1:,:]
                        check(prediction,expected_prediction,f'{suffix}/{repetition}/{length}/prediction')
                        for i,cache in enumerate(session.cache):check(cache,q,f'{suffix}/{repetition}/{length}/cache{i}')
                        assert built[length].weights['parameter:scale'] is prefill.weights['parameter:scale']
                    cache=session.cache;before=len(calls)
                    try:session.step({'x':token.cuda()})
                    except ValueError as exc:failures.append(dict(kind='context',message=str(exc),passed=True))
                    else:raise AssertionError('context limit admitted')
                    assert session.cache is cache and len(calls)==before
                    session.reset()
                assert calls==[3,4,5,6]
                session.prefill({'x':prefix.cuda()});session.cache[0].add_(1)
                before=len(calls)
                try:session.step({'x':torch.zeros(2,2,1,5,device='cuda')})
                except ValueError as exc:failures.append(dict(kind='cache mutation',message=str(exc),passed=True))
                else:raise AssertionError('mutated cache admitted')
                assert len(calls)==before
            finally:session.close()
            assert all(runtime._closed for runtime in [prefill,*built.values()])
        # Builder and post-build schema failures must retain all cache bytes.
        for kind in ('factory','schema','budget'):
            prefix=torch.randn(2,2,3,5)*.25
            prefill=make(Prefill(),(prefix,),f'failure-prefill-{kind}',False,False)
            def broken(length,contract):
                if kind=='factory':raise RuntimeError('deliberate builder failure')
                args=(torch.zeros(2,2,1,5),torch.zeros(2,2,length,5),torch.zeros(2,2,length,5))
                rt=make(Decoder(),args,f'failure-decode-{kind}',False,False)
                if kind=='schema':rt.plan['math']='f16_rne_f32_accum'
                return rt
            session=DecodeSession(prefill,{},specialization=dict(schema=1,factory=broken,max_context=7,max_build_attempts=1,artifact_key='failure-'+kind))
            try:
                session.prefill({'x':prefix.cuda()})
                if kind=='budget':session.step({'x':torch.zeros(2,2,1,5,device='cuda')})
                cache=session.cache;before=[t.clone()for t in cache];buckets=dict(session.buckets)
                try:session.step({'x':torch.zeros(2,2,1,5,device='cuda')})
                except (ValueError,RuntimeError) as exc:failures.append(dict(kind=kind,message=str(exc),passed=True))
                else:raise AssertionError('failed-builder control admitted')
                assert session.cache is cache and session.buckets==buckets
                for i,tensor in enumerate(cache):check(tensor,before[i],f'failure/{kind}/cache{i}')
            finally:session.close()
        torch.save(raw,root/'full-tensors.pt')
        report=dict(schema=1,gap='G39',passed=all(r['passed']for r in comparisons),python=platform.python_version(),
                    torch=torch.__version__,device=str(torch.cuda.get_device_properties(0)),runtime_cuda=torch.version.cuda,
                    comparisons=comparisons,requests=requests,runs=records,failure_controls=failures,
                    source_hashes={'openassembler/modelfront/decode.py':hashlib.sha256(Path(__import__('openassembler.modelfront.decode',fromlist=['']).__file__).read_bytes()).hexdigest()},
                    contract='FP32 causal prefill and one-token attention, positive exact lengths 3..6, four view/reuse policies, explicit bounded host factory; no timing/serving/paging claim')
        (root/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    finally:
        for runtime in all_runtimes:
            if not runtime._closed:runtime.close()

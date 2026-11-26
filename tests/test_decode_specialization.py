"""Explicit specialization limits, metadata custody and failure state ownership."""
import copy

import pytest

torch = pytest.importorskip('torch')
from openassembler.modelfront.decode import DecodeSession
from openassembler.modelfront.runtime import Runtime


class ReferenceRuntime(Runtime):
    """CPU control double; actual native ownership is checked in the A100 gate."""
    def __init__(self,length,prefill=False):
        self.length=length;self.is_prefill=prefill;self._forward_epoch=0;self._closed=False
        self.close_count=0;self.forward_count=0;self.failure=None
        self.plan={'arch':'sm_80','math':'fp32',
                   'parameters':[{'name':'parameter:w','target':'w','shape':[1],'dtype':'f32'}],
                   'outputs':['prediction','keys','values'],
                   'inputs':[{'name':'x','shape':[1,length if prefill else 1,2],'dtype':'f32'}]}
        if not prefill:self.plan['inputs'] += [{'name':n,'shape':[1,length,2],'dtype':'f32'} for n in ('past_k','past_v')]
        n=length if prefill else length+1
        self.specs={'prediction':{'shape':[1,length if prefill else 1,2],'dtype':'f32'},
                    'keys':{'shape':[1,n,2],'dtype':'f32'},'values':{'shape':[1,n,2],'dtype':'f32'}}
        self.weights={'parameter:w':torch.ones(1)}

    def _check(self,tensor,spec,name):
        dtype={'f32':torch.float32,'i64':torch.int64,'bool':torch.bool}[spec.get('dtype','f32')]
        if not isinstance(tensor,torch.Tensor) or list(tensor.shape)!=spec['shape'] or tensor.dtype!=dtype or not tensor.is_contiguous():
            raise ValueError('invalid '+name+' tensor metadata')
        return tensor.device

    def forward(self,inputs):
        if self.failure:raise self.failure
        self.forward_count+=1;self._forward_epoch+=1
        x=inputs['x']
        k=x.clone() if self.is_prefill else torch.cat((inputs['past_k'],x),dim=-2)
        v=x.clone() if self.is_prefill else torch.cat((inputs['past_v'],x),dim=-2)
        return {'prediction':x.clone(),'keys':k,'values':v}

    def close(self):
        self._closed=True;self.close_count+=1


def options(factory,**updates):
    return dict(schema=1,factory=factory,max_context=7,max_build_attempts=4,artifact_key='test-model-v1',**updates)


def begin(factory,*,buckets=None,**updates):
    prefill=ReferenceRuntime(2,True)
    configuration=options(factory);configuration.update(updates)
    session=DecodeSession(prefill,buckets or {},specialization=configuration)
    session.prefill({'x':torch.arange(4,dtype=torch.float32).reshape(1,2,2)})
    return session,prefill


def test_unseen_lengths_build_once_then_reuse_after_reset():
    calls=[];runtimes=[]
    def factory(length,contract):
        calls.append(copy.deepcopy(contract));runtime=ReferenceRuntime(length);runtimes.append(runtime)
        contract['parameter_specs'].clear()
        return runtime
    session,prefill=begin(factory)
    for repetition in range(2):
        full=session.cache[0].clone()
        for length in range(2,6):
            token=torch.full((1,1,2),float(length+repetition))
            assert torch.equal(session.step({'x':token}),token)
            full=torch.cat((full,token),dim=-2)
            assert all(torch.equal(cache,full) for cache in session.cache)
        session.reset()
        if repetition==0:session.prefill({'x':torch.arange(4,dtype=torch.float32).reshape(1,2,2)})
    assert [c['past_length']for c in calls]==[2,3,4,5]
    assert all(c['parameter_specs'] and c['artifact_key']=='test-model-v1'for c in calls)
    assert all(rt.weights['parameter:w'] is prefill.weights['parameter:w']for rt in runtimes)
    assert session._lazy.attempts==4
    session.close();session.close()
    assert prefill.close_count==1 and all(r.close_count==1 for r in runtimes)
    with pytest.raises(ValueError,match='closed'):session.prefill({'x':torch.zeros(1,2,2)})


@pytest.mark.parametrize('failure', ['factory','math','parameters','cache','append','prediction','runtime','forward','already_owned','used','closed','input'])
def test_failed_build_does_not_adopt_cache_or_variant_and_consumes_attempt(failure):
    returned=[];session=None
    def factory(length,contract):
        if failure=='factory':raise RuntimeError('builder failed')
        if failure=='already_owned':return session.prefill_runtime
        if failure=='runtime':return object()
        rt=ReferenceRuntime(length);returned.append(rt)
        if failure=='math':rt.plan['math']='f16_rne_f32_accum'
        if failure=='parameters':rt.plan['parameters'][0]['shape']=[2]
        if failure=='cache':rt.plan['inputs'][1]['shape']=[1,length,3]
        if failure=='append':rt.specs['keys']['shape'][-2]+=1
        if failure=='prediction':rt.specs['prediction']['dtype']='i64'
        if failure=='forward':rt.failure=RuntimeError('launch failed')
        if failure=='used':rt._forward_epoch=1
        if failure=='closed':rt._closed=True
        if failure=='input':rt.plan['inputs'][0]['shape']=[1,2,2]
        return rt
    session,prefill=begin(factory,max_build_attempts=1)
    old_cache=session.cache;old=[t.clone()for t in old_cache];old_producer=session._producer
    with pytest.raises((ValueError,RuntimeError)):session.step({'x':torch.ones(1,1,2)})
    assert session.cache is old_cache and session._producer==old_producer and session.buckets=={}
    assert all(torch.equal(a,b)for a,b in zip(old_cache,old))
    assert session._lazy.attempts==1 and session._lazy.schema is None
    assert not prefill._closed
    assert all(r._closed for r in returned)
    with pytest.raises(ValueError,match='budget'):session.step({'x':torch.ones(1,1,2)})
    session.close()


@pytest.mark.parametrize('guard',['context','binding','dtype','cache_mutation','weights','producer','input_schema'])
def test_live_state_guards_run_before_factory(guard):
    calls=[]
    def factory(length,contract):calls.append(length);return ReferenceRuntime(length)
    session,prefill=begin(factory,max_context=3 if guard=='context'else 7)
    token=torch.ones(1,1,2)
    if guard in ('context','input_schema'):session.step({'x':token})
    inputs={'x':token}
    if guard=='binding':inputs['past_k']=session.cache[0]
    if guard=='dtype':inputs['x']=token.to(torch.float64)
    if guard=='cache_mutation':session.cache[0].add_(1)
    if guard=='weights':prefill.weights['parameter:w'].add_(1)
    if guard=='producer':prefill._forward_epoch+=1
    if guard=='input_schema':inputs['x']=torch.ones(1,2,2)
    before=list(calls);cache=session.cache
    with pytest.raises(ValueError):session.step(inputs)
    assert calls==before and session.cache is cache
    session.close()


@pytest.mark.parametrize('field,value',[('schema',True),('max_context',0),('max_build_attempts',True),('artifact_key',''),('factory',None)])
def test_invalid_specialization_policy_rejects(field,value):
    config=options(lambda length,contract:ReferenceRuntime(length));config[field]=value
    with pytest.raises(ValueError):DecodeSession(ReferenceRuntime(2,True),{},specialization=config)


def test_existing_bucket_rejects_cross_layout_before_binding_weights():
    prefill=ReferenceRuntime(2,True);bucket=ReferenceRuntime(2)
    before=bucket.weights
    bucket.plan['math']='f16_rne_f32_accum'
    with pytest.raises(ValueError,match='math'):
        DecodeSession(prefill,{2:bucket},specialization=options(lambda length,contract:ReferenceRuntime(length)))
    assert bucket.weights is before


def test_changed_prediction_shape_rejects_new_variant_and_keeps_first_variant():
    def factory(length,contract):
        rt=ReferenceRuntime(length)
        if length==3:rt.specs['prediction']['shape']=[1,1,3]
        return rt
    session,_=begin(factory)
    session.step({'x':torch.ones(1,1,2)})
    cache=session.cache;variants=dict(session.buckets)
    with pytest.raises(ValueError,match='prediction schema'):
        session.step({'x':torch.ones(1,1,2)})
    assert session.cache is cache and session.buckets==variants
    session.close()


def test_prefill_context_limit_rejects_before_forward_and_budget_survives_reset():
    prefill=ReferenceRuntime(3,True)
    with pytest.raises(ValueError,match='maximum context'):
        DecodeSession(prefill,{},specialization=dict(options(lambda length,contract:ReferenceRuntime(length)),max_context=2))
    assert prefill.forward_count==0
    session,_=begin(lambda length,contract:ReferenceRuntime(length),max_build_attempts=1)
    session.step({'x':torch.ones(1,1,2)})
    session.reset();session.prefill({'x':torch.ones(1,2,2)})
    session.step({'x':torch.ones(1,1,2)})
    with pytest.raises(ValueError,match='budget'):session.step({'x':torch.ones(1,1,2)})
    session.close()

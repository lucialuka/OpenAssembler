"""Owned cache transitions over explicitly precompiled exact-length OA programs.

Each program returns (prediction, keys, values). Keys/values are FP32, contiguous,
rank3/4 and grow on axis -2. No hidden compilation, padding or vendor compute.
"""
class DecodeSession:
    def __init__(self,prefill,buckets,*,cache_inputs=('past_k','past_v')):
        self.prefill_runtime=prefill
        self.buckets=dict(buckets)
        self.cache_inputs=tuple(cache_inputs)
        self.cache=None
        self._versions=None
        self._producer=None
        self._weight_versions=None
        if len(self.cache_inputs)!=2 or len(set(self.cache_inputs))!=2 or not self.buckets:
            raise ValueError('decode needs two cache inputs and positive-length buckets')
        canonical={p['target']:p for p in prefill.plan['parameters']}
        for length,runtime in self.buckets.items():
            specs={s['name']:s for s in runtime.plan['inputs']}
            if type(length) is not int or length<=0 or len(runtime.plan['outputs'])!=3:
                raise ValueError('decode buckets need positive exact lengths and three outputs')
            for name in self.cache_inputs:
                spec=specs.get(name,{})
                if spec.get('dtype')!='f32' or len(spec.get('shape',[])) not in (3,4) or spec['shape'][-2]!=length:
                    raise ValueError('decode bucket cache input shape mismatch')
            for name,output in zip(self.cache_inputs,runtime.plan['outputs'][1:]):
                expected=list(specs[name]['shape']);expected[-2]+=1
                if runtime.specs[output]['shape']!=expected:raise ValueError('decode must append exactly one cache token')
            targets={p['target']:p for p in runtime.plan['parameters']}
            if set(targets)!=set(canonical) or any(targets[t]['shape']!=canonical[t]['shape'] for t in canonical):
                raise ValueError('decode bucket parameter schema mismatch')
            # Explicit single owner for model state across all compiled variants.
            runtime.weights={targets[t]['name']:prefill.weights[canonical[t]['name']] for t in canonical}
        if len(prefill.plan['outputs'])!=3:raise ValueError('prefill must return prediction, keys and values')

    def _adopt(self,runtime,outputs):
        cache=tuple(outputs[n] for n in runtime.plan['outputs'][1:])
        if cache[0].shape[-2]!=cache[1].shape[-2]:raise ValueError('cache lengths disagree')
        self.cache=cache
        self._versions=tuple((t.data_ptr(),t._version) for t in cache)
        self._producer=(runtime,runtime._forward_epoch)
        self._weight_versions={n:(t.data_ptr(),t._version) for n,t in self.prefill_runtime.weights.items()}
        return outputs[runtime.plan['outputs'][0]]

    def prefill(self,inputs):
        if self.cache is not None:raise ValueError('reset the session before a new prefill')
        outputs=self.prefill_runtime.forward(inputs)
        return self._adopt(self.prefill_runtime,outputs)

    def step(self,inputs):
        if self.cache is None:raise ValueError('prefill is required before decode')
        if self._producer[0]._forward_epoch != self._producer[1]:
            raise ValueError('cache producer was run outside the decode session')
        if self._weight_versions != {n:(t.data_ptr(),t._version) for n,t in self.prefill_runtime.weights.items()}:
            raise ValueError('model weights changed after cache creation; reset and prefill again')
        if any((t.data_ptr(),t._version)!=v for t,v in zip(self.cache,self._versions)):
            raise ValueError('session cache was mutated by the caller')
        length=self.cache[0].shape[-2]
        if length not in self.buckets:raise ValueError(f'no precompiled OA decode bucket for past length {length}')
        runtime=self.buckets[length]
        if set(inputs)&set(self.cache_inputs):raise ValueError('session owns cache input bindings')
        values=dict(inputs);values.update(zip(self.cache_inputs,self.cache))
        outputs=runtime.forward(values)
        return self._adopt(runtime,outputs)

    def reset(self):
        self.cache=None;self._versions=None;self._producer=None;self._weight_versions=None

    def close(self):
        for runtime in [self.prefill_runtime,*self.buckets.values()]:runtime.close()
        self.reset()

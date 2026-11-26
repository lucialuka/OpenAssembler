"""Owned cache transitions over explicitly precompiled exact-length OA programs.

Each program returns (prediction, keys, values). Keys/values are FP32, contiguous,
rank3/4 and grow on axis -2. No hidden compilation, padding or vendor compute.
"""
import copy


class DecodeSession:
    def __init__(self,prefill,buckets,*,cache_inputs=('past_k','past_v'),specialization=None):
        self.prefill_runtime=prefill
        self.buckets=dict(buckets)
        self.cache_inputs=tuple(cache_inputs)
        self.cache=None
        self._versions=None
        self._producer=None
        self._weight_versions=None
        self._specializer=None
        self._build_attempts=0
        if len(self.cache_inputs)!=2 or len(set(self.cache_inputs))!=2:
            raise ValueError('decode needs two cache inputs')
        if not self.buckets and specialization is None:
            raise ValueError('decode needs positive-length buckets unless specialization is explicit')
        if specialization is not None:
            if not isinstance(specialization,dict) or set(specialization) != {
                    'schema','factory','max_context','max_build_attempts','artifact_key'}:
                raise ValueError('specialization requires schema, factory, max_context, max_build_attempts and artifact_key')
            if specialization['schema'] != 1 or not callable(specialization['factory']):
                raise ValueError('decode specialization schema/factory is invalid')
            if type(specialization['max_context']) is not int or specialization['max_context'] <= 0:
                raise ValueError('decode specialization max_context must be a positive integer')
            if type(specialization['max_build_attempts']) is not int or specialization['max_build_attempts'] <= 0:
                raise ValueError('decode specialization max_build_attempts must be a positive integer')
            if type(specialization['artifact_key']) is not str or not specialization['artifact_key'].strip():
                raise ValueError('decode specialization artifact_key must be a stable nonempty string')
            self._specializer={**specialization}
        canonical={p['target']:p for p in prefill.plan['parameters']}
        for length,runtime in self.buckets.items():
            self._validate_bucket(length,runtime)
            targets={p['target']:p for p in runtime.plan['parameters']}
            if set(targets)!=set(canonical) or any(targets[t]['shape']!=canonical[t]['shape'] for t in canonical):
                raise ValueError('decode bucket parameter schema mismatch')
            # Explicit single owner for model state across all compiled variants.
            runtime.weights={targets[t]['name']:prefill.weights[canonical[t]['name']] for t in canonical}
        if len(prefill.plan['outputs'])!=3:raise ValueError('prefill must return prediction, keys and values')

    def _validate_bucket(self,length,runtime):
        if type(length) is not int or length<=0 or len(runtime.plan['outputs'])!=3:
            raise ValueError('decode buckets need positive exact lengths and three outputs')
        specs={s['name']:s for s in runtime.plan['inputs']}
        for name in self.cache_inputs:
            spec=specs.get(name,{})
            if spec.get('dtype')!='f32' or len(spec.get('shape',[])) not in (3,4) or spec['shape'][-2]!=length:
                raise ValueError('decode bucket cache input shape mismatch')
        for name,output in zip(self.cache_inputs,runtime.plan['outputs'][1:]):
            expected=list(specs[name]['shape']);expected[-2]+=1
            if runtime.specs[output]['shape']!=expected:raise ValueError('decode must append exactly one cache token')

    def _specialize(self,length):
        """Build one exact next-length variant after all live-state guards pass."""
        if self._specializer is None:
            raise ValueError(f'no precompiled OA decode bucket for past length {length}')
        if type(length) is not int or length <= 0 or length + 1 > self._specializer['max_context']:
            raise ValueError(f'past length {length} exceeds specialization maximum context')
        if self._build_attempts >= self._specializer['max_build_attempts']:
            raise ValueError('decode specialization build-attempt budget exhausted')
        contract={
            'schema':1,
            'artifact_key':self._specializer['artifact_key'],
            'past_length':length,
            'next_length':length+1,
            'max_context':self._specializer['max_context'],
            'cache_inputs':list(self.cache_inputs),
            'outputs':list(self.prefill_runtime.plan['outputs']),
            'parameters':copy.deepcopy(self.prefill_runtime.plan['parameters']),
        }
        self._build_attempts += 1
        runtime=self._specializer['factory'](length,copy.deepcopy(contract))
        self._validate_bucket(length,runtime)
        canonical={p['target']:p for p in self.prefill_runtime.plan['parameters']}
        targets={p['target']:p for p in runtime.plan['parameters']}
        if set(targets)!=set(canonical) or any(targets[t]['shape']!=canonical[t]['shape'] for t in canonical):
            raise ValueError('decode specialization parameter schema mismatch')
        runtime.weights={targets[t]['name']:self.prefill_runtime.weights[canonical[t]['name']] for t in canonical}
        return runtime

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
        runtime=self.buckets.get(length)
        specialized=False
        if runtime is None:
            runtime=self._specialize(length)
            specialized=True
        if set(inputs)&set(self.cache_inputs):raise ValueError('session owns cache input bindings')
        values=dict(inputs);values.update(zip(self.cache_inputs,self.cache))
        try:
            outputs=runtime.forward(values)
            prediction=self._adopt(runtime,outputs)
        except Exception:
            if specialized:
                runtime.close()
            raise
        if specialized:
            self.buckets[length]=runtime
        return prediction

    def reset(self):
        self.cache=None;self._versions=None;self._producer=None;self._weight_versions=None

    def close(self):
        seen=set()
        for runtime in [self.prefill_runtime,*self.buckets.values()]:
            if id(runtime) not in seen:
                runtime.close();seen.add(id(runtime))
        self.reset()

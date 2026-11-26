"""Owned exact-length OA decoding with optional bounded host specialization.

Each program returns (prediction, keys, values). Keys/values are FP32, contiguous,
rank3/4 and grow on axis -2. Compilation requires an explicit factory policy;
the default uses precompiled variants. No padding or vendor compute fallback.
"""
class DecodeSession:
    def __init__(self,prefill,buckets,*,cache_inputs=('past_k','past_v'),specialization=None):
        self.prefill_runtime=prefill
        self.buckets=dict(buckets)
        self.cache_inputs=tuple(cache_inputs)
        self.cache=None
        self._versions=None
        self._producer=None
        self._weight_versions=None
        self._closed=False
        self._lazy=_LazyVariants(self,specialization) if specialization is not None else None
        if len(self.cache_inputs)!=2 or len(set(self.cache_inputs))!=2 or (not self.buckets and self._lazy is None):
            raise ValueError('decode needs two cache inputs and positive-length buckets')
        if self._lazy is not None:self._lazy.validate_initial()
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
        if self._lazy is not None:self._lazy.validate_outputs(runtime,outputs)
        cache=tuple(outputs[n] for n in runtime.plan['outputs'][1:])
        if cache[0].shape[-2]!=cache[1].shape[-2]:raise ValueError('cache lengths disagree')
        self.cache=cache
        self._versions=tuple((t.data_ptr(),t._version) for t in cache)
        self._producer=(runtime,runtime._forward_epoch)
        self._weight_versions={n:(t.data_ptr(),t._version) for n,t in self.prefill_runtime.weights.items()}
        return outputs[runtime.plan['outputs'][0]]

    def prefill(self,inputs):
        if self._closed:raise ValueError('decode session is closed')
        if self._lazy is not None:self._lazy.validate_prefill()
        if self.cache is not None:raise ValueError('reset the session before a new prefill')
        outputs=self.prefill_runtime.forward(inputs)
        return self._adopt(self.prefill_runtime,outputs)

    def step(self,inputs):
        if self._closed:raise ValueError('decode session is closed')
        if self.cache is None:raise ValueError('prefill is required before decode')
        if self._producer[0]._forward_epoch != self._producer[1]:
            raise ValueError('cache producer was run outside the decode session')
        if self._weight_versions != {n:(t.data_ptr(),t._version) for n,t in self.prefill_runtime.weights.items()}:
            raise ValueError('model weights changed after cache creation; reset and prefill again')
        if any((t.data_ptr(),t._version)!=v for t,v in zip(self.cache,self._versions)):
            raise ValueError('session cache was mutated by the caller')
        if self._lazy is not None:return self._lazy.step(inputs)
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
        if self._closed:return
        seen=set()
        for runtime in [self.prefill_runtime,*self.buckets.values()]:
            if id(runtime) not in seen:runtime.close();seen.add(id(runtime))
        self.reset();self._closed=True


class _LazyVariants:
    """Explicit host specialization with exact metadata and bounded attempts."""
    def __init__(self,session,options):
        import copy
        fields={'schema','factory','max_context','max_build_attempts','artifact_key'}
        if not isinstance(options,dict) or set(options)!=fields:
            raise ValueError('specialization requires schema, factory, max_context, max_build_attempts and artifact_key')
        if type(options['schema']) is not int or options['schema']!=1 or not callable(options['factory']):
            raise ValueError('decode specialization schema/factory is invalid')
        for name in ('max_context','max_build_attempts'):
            if type(options[name]) is not int or options[name]<=0:
                raise ValueError('decode specialization '+name+' must be a positive integer')
        if type(options['artifact_key']) is not str or not options['artifact_key'].strip():
            raise ValueError('decode specialization artifact_key must be a stable nonempty string')
        self.session=session
        self.options=dict(options)
        self.attempts=0
        self.schema=None
        self.prediction_spec=None
        self.parameters=copy.deepcopy(session.prefill_runtime.plan['parameters'])
        self.arch=session.prefill_runtime.plan.get('arch')
        self.math=session.prefill_runtime.plan.get('math')
        if self.arch!='sm_80' or self.math not in ('fp32','f16_rne_f32_accum'):
            raise ValueError('decode specialization requires a declared native SM80 math policy')

    @staticmethod
    def spec(spec):
        shape=spec.get('shape')
        dtype=spec.get('dtype','f32')
        if not isinstance(shape,list) or any(type(x) is not int or x<=0 for x in shape) or dtype not in ('f32','i64','bool'):
            raise ValueError('decode specialization requires exact positive tensor shapes and declared dtypes')
        return {'shape':list(shape),'dtype':dtype}

    def parameter_schema(self,parameters):
        result={}
        for item in parameters:
            target=item.get('target')
            if not isinstance(target,str) or target in result:
                raise ValueError('decode specialization parameter targets must be unique')
            result[target]=self.spec(item)
        return result

    def validate_prefill(self):
        runtime=self.session.prefill_runtime
        outputs=runtime.plan['outputs']
        if len(outputs)!=3 or len(set(outputs))!=3:
            raise ValueError('prefill needs three distinct outputs')
        for name in outputs[1:]:
            spec=self.spec(runtime.specs[name])
            if spec['dtype']!='f32' or len(spec['shape']) not in (3,4) or spec['shape'][-2]>self.options['max_context']:
                raise ValueError('prefill cache must be positive contiguous FP32 within maximum context')

    def validate_initial(self):
        self.validate_prefill()
        self.parameter_schema(self.parameters)
        for length,runtime in self.session.buckets.items():
            self.validate_runtime(length,runtime)

    def validate_outputs(self,runtime,outputs):
        for name in runtime.plan['outputs']:
            runtime._check(outputs[name],runtime.specs[name],name)

    def validate_runtime(self,length,runtime,contract=None):
        from .runtime import Runtime
        if not isinstance(runtime,Runtime):
            raise ValueError('decode factory must return an OA native Runtime')
        if type(length) is not int or length<=0 or length+1>self.options['max_context']:
            raise ValueError('decode past length exceeds maximum context')
        if runtime.plan.get('arch')!=self.arch or runtime.plan.get('math')!=self.math:
            raise ValueError('decode specialization architecture/math policy mismatch')
        if self.parameter_schema(runtime.plan['parameters'])!=self.parameter_schema(self.parameters):
            raise ValueError('decode specialization parameter schema mismatch')
        if getattr(runtime,'_closed',False):
            raise ValueError('decode specialization runtime is closed')
        outputs=runtime.plan['outputs']
        if len(outputs)!=3 or len(set(outputs))!=3:
            raise ValueError('decode needs three distinct outputs')
        specs={}
        for item in runtime.plan['inputs']:
            if item['name'] in specs:raise ValueError('decode input names must be unique')
            specs[item['name']]=self.spec(item)
        for name,output in zip(self.session.cache_inputs,outputs[1:]):
            spec=specs.get(name,{})
            if spec.get('dtype')!='f32' or len(spec.get('shape',[])) not in (3,4) or spec['shape'][-2]!=length:
                raise ValueError('decode bucket cache input shape mismatch')
            expected={'shape':list(spec['shape']),'dtype':'f32'};expected['shape'][-2]+=1
            if self.spec(runtime.specs[output])!=expected:
                raise ValueError('decode must append exactly one FP32 cache token')
        prediction=self.spec(runtime.specs[outputs[0]])
        if self.prediction_spec is not None and prediction!=self.prediction_spec:
            raise ValueError('decode specialization prediction schema changed')
        if contract is not None:
            expected={item['name']:self.spec(item) for item in contract['input_specs']+contract['cache_specs']}
            if specs!=expected:raise ValueError('decode specialization input/cache schema mismatch')
        return prediction

    def bindings(self,inputs):
        session=self.session
        if not isinstance(inputs,dict) or any(type(name) is not str for name in inputs):
            raise ValueError('decode inputs must be a tensor-name dictionary')
        if set(inputs)&set(session.cache_inputs):raise ValueError('session owns cache input bindings')
        schema=[];devices=set()
        for name,tensor in sorted(inputs.items()):
            dtype={'torch.float32':'f32','torch.int64':'i64','torch.bool':'bool'}.get(str(getattr(tensor,'dtype',None)))
            spec=self.spec({'shape':list(getattr(tensor,'shape',[])),'dtype':dtype})
            devices.add(session.prefill_runtime._check(tensor,spec,name))
            schema.append({'name':name,**spec})
        if self.schema is not None and schema!=self.schema:
            raise ValueError('decode non-cache input schema changed')
        caches=[]
        for name,tensor in zip(session.cache_inputs,session.cache):
            spec=self.spec({'shape':list(tensor.shape),'dtype':'f32'})
            if len(spec['shape']) not in (3,4):raise ValueError('cache must have rank three or four')
            devices.add(session.prefill_runtime._check(tensor,spec,name))
            caches.append({'name':name,**spec})
        if len(devices)!=1:raise ValueError('decode inputs and cache must share one device')
        return schema,caches

    def step(self,inputs):
        import copy
        session=self.session
        length=session.cache[0].shape[-2]
        if length+1>self.options['max_context']:
            raise ValueError('decode maximum context reached')
        schema,caches=self.bindings(inputs)
        contract={'schema':1,'artifact_key':self.options['artifact_key'],'past_length':length,
                  'next_length':length+1,'max_context':self.options['max_context'],
                  'arch':self.arch,'math':self.math,'input_specs':schema,'cache_specs':caches,
                  'parameter_specs':self.parameters,'cache_inputs':list(session.cache_inputs)}
        runtime=session.buckets.get(length)
        created=runtime is None
        if created:
            if self.attempts>=self.options['max_build_attempts']:
                raise ValueError('decode specialization build-attempt budget exhausted')
            self.attempts+=1
            runtime=self.options['factory'](length,copy.deepcopy(contract))
            if any(runtime is r for r in [session.prefill_runtime,*session.buckets.values()]):
                raise ValueError('decode factory must return a fresh unowned runtime')
        try:
            if created and getattr(runtime,'_forward_epoch',None)!=0:
                raise ValueError('decode factory must return an unused runtime')
            prediction_spec=self.validate_runtime(length,runtime,contract)
            if created:
                canonical={p['target']:p for p in self.parameters}
                runtime.weights={p['name']:session.prefill_runtime.weights[canonical[p['target']]['name']]
                                 for p in runtime.plan['parameters']}
            values=dict(inputs);values.update(zip(session.cache_inputs,session.cache))
            outputs=runtime.forward(values)
            prediction=session._adopt(runtime,outputs)
        except BaseException:
            if created and callable(getattr(runtime,'close',None)):runtime.close()
            raise
        if created:session.buckets[length]=runtime
        self.schema=copy.deepcopy(schema)
        self.prediction_spec=prediction_spec
        return prediction

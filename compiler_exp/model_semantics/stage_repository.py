from pathlib import Path
P=Path(__file__).parent/'candidate/openassembler/modelfront'
def edit(name,old,new):
 p=P/name;s=p.read_text()
 if old not in s:raise ValueError((name,old[:80]))
 p.write_text(s.replace(old,new,1))
edit('frontend.py','''    if model.get('input_dtypes', ['f32'] * len(shapes)) != ['f32'] * len(shapes):
        raise ModelError('native model inputs currently require f32; implicit precision changes are rejected')''','''    dtypes=model.get('input_dtypes',['f32']*len(shapes))
    if not isinstance(dtypes,list) or len(dtypes)!=len(shapes) or any(t not in ('f32','i64','bool') for t in dtypes):
        raise ModelError('model.input_dtypes must declare f32, i64 or bool for each input')
    ranges=model.get('input_ranges',[[0,1] for _ in shapes])
    if not isinstance(ranges,list) or len(ranges)!=len(shapes):raise ModelError('input_ranges must match input count')
    for dtype,bounds in zip(dtypes,ranges):
        if dtype=='i64' and (not isinstance(bounds,list) or len(bounds)!=2 or any(type(x) is not int for x in bounds) or not 0<=bounds[0]<bounds[1]<=2**31):
            raise ModelError('i64 example ranges need nonnegative integer [lower, upper) bounds')''')
edit('frontend.py','''    if model.get('loss', 'mse') != 'mse' or model.get('optimizer', 'sgd') != 'sgd':
        raise ModelError('native model training currently supports MSE and plain SGD')''','''    if model.get('loss','mse') not in ('mse','cross_entropy'):raise ModelError('loss must be mse or cross_entropy')
    from .optimizer import optimizer_options
    try:
        policy=optimizer_options({'name':model.get('optimizer','sgd'),**{k:model[k] for k in ('beta1','beta2','epsilon','weight_decay') if k in model}})
    except ValueError as exc:raise ModelError(str(exc)) from None''')
edit('frontend.py',"input_dtypes=['f32'] * len(shapes)","input_dtypes=dtypes, input_ranges=ranges, loss=model.get('loss','mse'), optimizer_policy=policy")
edit('frontend.py',"    values['execution'] = execution","    if execution=='cuda_graph' and (any(t!='f32' for t in dtypes) or values['loss']!='mse' or policy['name']!='sgd'):\n        raise ModelError('indexed inputs, cross_entropy and Adam require direct execution')\n    values['execution'] = execution")
edit('worker.py',"""    return [torch.randn(shape, generator=generator, dtype=torch.float32) * 0.25
            for shape in config['input_shapes']]""","""    result=[]
    for shape,dtype,bounds in zip(config['input_shapes'],config.get('input_dtypes',['f32']*len(config['input_shapes'])),config.get('input_ranges',[[0,1]]*len(config['input_shapes']))):
        if dtype=='f32':value=torch.randn(shape,generator=generator,dtype=torch.float32)*0.25
        elif dtype=='i64':value=torch.randint(bounds[0],bounds[1],shape,generator=generator,dtype=torch.int64)
        else:value=torch.randint(0,2,shape,generator=generator,dtype=torch.int64).bool()
        result.append(value)
    return result""")
edit('worker.py',"    runtime = Runtime.load(output)","""    reference_optimizer=None
    if reference is not None and config['mode']=='training' and numeric_reference is None:
        from .optimizer import optimizer_options
        policy=optimizer_options(plan.get('optimizer',{}))
        cls={'sgd':torch.optim.SGD,'adam':torch.optim.Adam,'adamw':torch.optim.AdamW}[policy['name']]
        kwargs={} if policy['name']=='sgd' else dict(betas=(policy['beta1'],policy['beta2']),eps=policy['epsilon'],weight_decay=policy['weight_decay'],foreach=False)
        reference_optimizer=cls(reference.parameters(),lr=config['learning_rate'],**kwargs)
    runtime = Runtime.load(output)""")
edit('worker.py',"x.requires_grad_(config['mode'] == 'training' and verify)","x.requires_grad_(config['mode'] == 'training' and verify and x.dtype == torch.float32)")
edit('worker.py',"""                target = torch.randn(shape, generator=torch.Generator().manual_seed(
                    config['seed'] + 997 + iteration), dtype=torch.float32) * 0.25""","""                generator=torch.Generator().manual_seed(config['seed']+997+iteration)
                target = (torch.randint(0,shape[1],(shape[0],),generator=generator,dtype=torch.int64)
                          if plan.get('loss')=='cross_entropy' else torch.randn(shape,generator=generator,dtype=torch.float32)*0.25)""")
edit('worker.py',"loss = torch.nn.functional.mse_loss(expected_values[0], target)","loss = (torch.nn.functional.cross_entropy(expected_values[0],target) if plan.get('loss')=='cross_entropy' else torch.nn.functional.mse_loss(expected_values[0], target))")
edit('worker.py',"""                    with torch.no_grad():
                        for p in reference.parameters():
                            if p.grad is not None:
                                p.add_(p.grad, alpha=-config['learning_rate'])""","""                    reference_optimizer.step()
                    if plan.get('optimizer',{}).get('name','sgd')!='sgd':
                        native_state=runtime.optimizer_state_dict()
                        for parameter in plan['parameters']:
                            original=state[parameter['target']];name=parameter['name']
                            if original in reference_optimizer.state and name in native_state['moments']:
                                for key,refkey in (('m','exp_avg'),('v','exp_avg_sq')):
                                    checks.append(_check(native_state['moments'][name][key],reference_optimizer.state[original][refkey],f'{iteration}/optimizer/{name}/{key}',config))""")
edit('worker.py',"                from .optimization import optimize_plan","""                if config.get('loss','mse')!='mse':plan['loss']=config['loss']
                if config.get('optimizer_policy',{}).get('name','sgd')!='sgd':plan['optimizer']=config['optimizer_policy']
                from .optimization import optimize_plan""")
# Reference preserves independently rounded matrix operands; optimizer uses CPU torch as control.
edit('reference.py','        self.state = None','        self.state = None\n        self._optimizer = None')
edit('reference.py','''        self._check(target, self.plan["buffers"][name])''','''        ce=self.plan.get("loss")=="cross_entropy"
        self._check(target, {"shape":[prediction.shape[0]],"dtype":"i64"} if ce else self.plan["buffers"][name])''')
edit('reference.py','''            difference = prediction - target
            loss = (difference * difference).mean()
            seed = difference * (2 / difference.numel())''','''            if ce:
                loss=self.torch.nn.functional.cross_entropy(prediction,target)
                seed=self.torch.softmax(prediction,dim=-1)
                seed[self.torch.arange(prediction.shape[0]),target]-=1
                seed/=prediction.shape[0]
            else:
                difference = prediction - target
                loss = (difference * difference).mean()
                seed = difference * (2 / difference.numel())''')
edit('reference.py','''            for parameter, gradient in gradients["parameters"].items():
                self.weights[parameter] = self.weights[parameter] - lr * gradient''','''            from .optimizer import optimizer_options
            options=optimizer_options(self.plan.get("optimizer",{}))
            if options["name"]=="sgd":
                for parameter,gradient in gradients["parameters"].items():self.weights[parameter]=self.weights[parameter]-lr*gradient
            else:
                if self._optimizer is None:
                    cls=self.torch.optim.Adam if options["name"]=="adam" else self.torch.optim.AdamW
                    self._optimizer=cls(list(self.weights.values()),lr=lr,betas=(options["beta1"],options["beta2"]),eps=options["epsilon"],weight_decay=options["weight_decay"],foreach=False)
                for group in self._optimizer.param_groups:group["lr"]=lr
                for parameter,gradient in gradients["parameters"].items():self.weights[parameter].grad=gradient
                self._optimizer.step();self._optimizer.zero_grad(set_to_none=True)''')
print('repository overlay staged')

"""Explicit FP32 optimizer policy. Default policy identities remain stable."""
import math


def optimizer_options(options=None):
    options=dict(options or {})
    name=options.get('name','sgd')
    if name not in ('sgd','adam','adamw'):raise ValueError('optimizer must be sgd, adam or adamw')
    if name=='sgd':
        defaults={'momentum':0.,'dampening':0.,'weight_decay':0.,'nesterov':False,'maximize':False}
        if set(options)-{'name',*defaults}:raise ValueError('unsupported SGD optimizer options')
        values={**defaults,**{k:v for k,v in options.items() if k!='name'}}
        for k in ('nesterov','maximize'):
            if type(values[k]) is not bool:raise ValueError('optimizer flags must be Boolean')
        for k in ('momentum','dampening','weight_decay'):
            value=values[k]
            if type(value) not in (int,float) or not math.isfinite(value) or value<0:
                raise ValueError('SGD scalars must be finite and nonnegative')
        if values['nesterov'] and (values['momentum']<=0 or values['dampening']!=0):
            raise ValueError('Nesterov requires positive momentum and zero dampening')
        return {'name':name,**{k:v for k,v in values.items() if v!=defaults[k]}}
    defaults={'name':name,'beta1':.9,'beta2':.999,'epsilon':1e-8,'weight_decay':0. if name=='adam' else .01}
    if set(options)-{*defaults,'amsgrad','maximize'}:raise ValueError('unsupported optimizer options')
    result={**defaults,**{k:v for k,v in options.items() if k not in ('amsgrad','maximize')}}
    for k in ('amsgrad','maximize'):
        value=options.get(k,False)
        if type(value) is not bool:raise ValueError('optimizer flags must be Boolean')
        if value:result[k]=True
    for k in ('beta1','beta2','epsilon','weight_decay'):
        value=result[k]
        if type(value) not in (int,float) or not math.isfinite(value):raise ValueError('optimizer scalars must be finite')
    if not 0<=result['beta1']<1 or not 0<=result['beta2']<1 or result['epsilon']<=0 or result['weight_decay']<0:
        raise ValueError('invalid Adam betas, epsilon or weight decay')
    return result


def optimizer_state_keys(options=None):
    p=optimizer_options(options)
    if p['name']=='sgd':return ('momentum',) if p.get('momentum',0)>0 else ()
    return ('m','v','max_v') if p.get('amsgrad',False) else ('m','v')


def torch_optimizer_kwargs(options=None):
    """CPU control mapping only; production GPU updates never use Torch optimizers."""
    p=optimizer_options(options)
    if p['name']=='sgd':return {k:v for k,v in p.items() if k!='name'}
    return {'betas':(p['beta1'],p['beta2']),'eps':p['epsilon'],'weight_decay':p['weight_decay'],
            **{k:p[k] for k in ('amsgrad','maximize') if k in p}}

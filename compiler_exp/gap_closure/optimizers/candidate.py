"""Independent explicit optimizer recurrence candidate against CPU PyTorch."""
import copy
import hashlib
import json
import math
from pathlib import Path
import platform

POLICIES=[
 {'name':'sgd'}, {'name':'sgd','weight_decay':.03,'maximize':True},
 {'name':'sgd','momentum':.8,'dampening':.25,'weight_decay':.03},
 {'name':'sgd','momentum':.9,'nesterov':True,'maximize':True,'weight_decay':.02},
 {'name':'adam','beta1':.8,'beta2':.95,'epsilon':1e-8,'weight_decay':.03,'amsgrad':True,'maximize':True},
 {'name':'adamw','beta1':.8,'beta2':.95,'epsilon':1e-8,'weight_decay':.03,'amsgrad':True},
 {'name':'adam','beta1':0.,'beta2':0.,'epsilon':1e-4,'weight_decay':.03,'maximize':True},
 {'name':'adamw','beta1':.9,'beta2':.999,'epsilon':1e-8,'weight_decay':0.,'maximize':True},
]

def options_torch(p):
    q=dict(p);kind=q.pop('name')
    if kind!='sgd':q['betas']=(q.pop('beta1',.9),q.pop('beta2',.999));q['eps']=q.pop('epsilon',1e-8)
    return q

def recurrence(w,g,state,p,step,lr):
    g=-g if p.get('maximize',False) else g
    if p['name']=='sgd':
        g=g+p.get('weight_decay',0)*w
        if p.get('momentum',0):
            b=g.clone() if step==1 else p['momentum']*state['momentum']+(1-p.get('dampening',0))*g
            state['momentum']=b
            g=g+p['momentum']*b if p.get('nesterov',False) else b
        return w-lr*g,state
    if p['name']=='adam':g=g+p.get('weight_decay',0)*w
    b1,b2=p.get('beta1',.9),p.get('beta2',.999)
    m=state.get('m',g*0);v=state.get('v',g*0)
    m=m+(g-m)*(1-b1);v=v*b2+g*g*(1-b2)
    state.update(m=m,v=v)
    if p.get('amsgrad',False):
        state['max_v']=v.maximum(state.get('max_v',v*0));denom=state['max_v']
    else:denom=v
    denom=denom.sqrt()/math.sqrt(1-b2**step)+p.get('epsilon',1e-8)
    if p['name']=='adamw':w=w*(1-lr*p.get('weight_decay',.01))
    return w-(lr/(1-b1**step))*m/denom,state

def run():
    import torch
    torch.set_num_threads(1);g=torch.Generator().manual_seed(34052);rows=[]
    for policy in POLICIES:
        actual=torch.linspace(-.5,.5,129);expected=actual.clone().requires_grad_();state={}
        cls={'sgd':torch.optim.SGD,'adam':torch.optim.Adam,'adamw':torch.optim.AdamW}[policy['name']]
        opt=cls([expected],lr=.001,foreach=False,**options_torch(policy))
        resumed=None
        for step in range(1,25):
            grad=torch.randn(129,generator=g)*(.1 if step%3 else 1e-7)
            if step%5==0:grad.zero_()
            lr=0. if step==7 else .001+.0001*(step%4)
            opt.param_groups[0]['lr']=lr;expected.grad=grad.clone();opt.step()
            actual,state=recurrence(actual,grad,state,policy,step,lr)
            torch.testing.assert_close(actual,expected,rtol=4e-5,atol=5e-6)
            mapping={'momentum':'momentum_buffer','m':'exp_avg','v':'exp_avg_sq','max_v':'max_exp_avg_sq'}
            for key,value in state.items():torch.testing.assert_close(value,opt.state[expected][mapping[key]],rtol=4e-5,atol=5e-6)
            if step==12:resumed=(actual.clone(),copy.deepcopy(state))
            elif step>12:
                rw,rs=recurrence(*resumed[:1],grad,resumed[1],policy,step,lr)
                resumed=(rw,rs);torch.testing.assert_close(rw,actual,rtol=0,atol=0)
            rows.append({'policy':policy,'step':step,'passed':True,'state_keys':list(state)})
    return {'schema':1,'gap':'G34','passed':True,'seed':34052,'python':platform.python_version(),'torch':torch.__version__,
            'candidate_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'rows':rows,
            'hardware':'not run by this independent recurrence candidate'}
if __name__=='__main__':
    result=run();Path(__file__).with_name('candidate-results.json').write_text(json.dumps(result,indent=2)+'\n');print('passed',len(result['rows']),'complete parameter/state updates')

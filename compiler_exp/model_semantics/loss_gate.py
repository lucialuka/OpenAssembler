"""Loss stability boundaries and tiny/zero-gradient adaptive state controls."""
import torch

def run(make,check):
    class Affine(torch.nn.Module):
        def __init__(self,width):super().__init__();self.bias=torch.nn.Parameter(torch.linspace(-.2,.2,width))
        def forward(self,x):return x+self.bias
    results=[]
    for width in (1,33,513):
        model=Affine(width);x=torch.randn(3,width)*.25+10000
        plan,rt=make(model,(x,),f'loss-boundary-{width}',loss='cross_entropy')
        try:
            for repetition in range(2):
                target=torch.tensor([0,width//2,width-1]);opt=torch.optim.SGD(model.parameters(),lr=0)
                opt.zero_grad();prediction=model(x);loss=torch.nn.functional.cross_entropy(prediction,target);loss.backward()
                got=rt.train_step({'x':x.cuda()},target.cuda(),0)
                check(got['loss'].reshape(()),loss,f'loss-boundary/{width}/{repetition}/loss')
                check(got['gradients']['parameters']['parameter:bias'],model.bias.grad,f'loss-boundary/{width}/{repetition}/grad')
                results.append(rt.check_canaries());rt.poison_workspace()
        finally:rt.close()
    for kind in ('adam','adamw'):
        for eps in (1e-8,1e-4):
            model=Affine(7);policy=dict(name=kind,beta1=.9,beta2=.999,epsilon=eps,weight_decay=0.0)
            plan,rt=make(model,(torch.zeros(1,7),),f'tiny-{kind}-{eps}',optimizer=policy,optimizer_fusion=True)
            cls=torch.optim.Adam if kind=='adam' else torch.optim.AdamW
            optimizer=cls(model.parameters(),lr=.001,eps=eps,weight_decay=0,foreach=False)
            try:
                for iteration in range(8):
                    seed=torch.tensor([[0,1e-12,-1e-12,1e-9,-1e-9,.01,-.02]])*((-1)**iteration)
                    optimizer.zero_grad();model(torch.zeros(1,7)).backward(seed);optimizer.step()
                    rt.forward({'x':torch.zeros(1,7,device='cuda')});rt.backward({plan['outputs'][0]:seed.cuda()});rt.update(.001)
                    check(rt.weights['parameter:bias'],model.bias,f'tiny/{kind}/{eps}/{iteration}/weight')
                    state=rt.optimizer_state_dict()
                    for key,refkey in [('m','exp_avg'),('v','exp_avg_sq')]:check(state['moments']['parameter:bias'][key],optimizer.state[model.bias][refkey],f'tiny/{kind}/{eps}/{iteration}/{key}')
                results.append(rt.check_canaries())
            finally:rt.close()
    return results

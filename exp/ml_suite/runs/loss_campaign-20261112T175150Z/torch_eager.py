"""PyTorch CUDA eager comparator with explicit input/weight FP16 RNE casts."""
import math
import torch
import torch.nn.functional as F


class TorchEager:
    @staticmethod
    def allocate(shape):
        count=math.prod(shape);base=torch.empty(count+8,device='cuda')
        base[:4].view(torch.int32).fill_(-1515870811);base[-4:].view(torch.int32).fill_(-1515870811)
        return base,base[4:-4].view(shape)

    def __init__(self,plan,weights,kind):
        self.plan,self.kind=plan,kind;self.backings=[];self.weights={}
        for spec in plan['parameters']:
            backing,value=self.allocate(spec['shape']);value.copy_(weights[spec['name']].cuda())
            value=value.detach().requires_grad_(spec.get('requires_grad',True));self.backings.append(backing);self.weights[spec['name']]=value
        self.inputs={}
        for spec in plan['inputs']:
            backing,value=self.allocate(spec['shape']);self.backings.append(backing)
            self.inputs[spec['name']]=value.detach().requires_grad_()
        self.output=plan['outputs'][0]
        self.target_backing,self.target=self.allocate(plan['buffers'][self.output]['shape'])
        self.backings.append(self.target_backing);self.gradients=None

    def rounding(self,x):
        if self.plan.get('math')=='f16_rne_f32_accum':
            return x+(x.to(torch.float16).float()-x).detach()
        return x

    def evaluate(self,steps,values):
        capture_steps=getattr(self,'capture_snapshots',False)
        if capture_steps:self.snapshots=[]
        for step in steps:
            op,names,out=step['kernel'],step['inputs'],step['output']
            xs=[values[n] for n in names];attrs=step.get('attrs',{});shape=self.plan['buffers'][out]['shape']
            if op=='linear':
                value=self.rounding(xs[0])@self.rounding(xs[1]).T
                if len(xs)==3:value=value+xs[2]
                result=value
            elif op=='bmm':result=torch.bmm(self.rounding(xs[0]),self.rounding(xs[1]))
            elif op=='linear_dx':result=self.rounding(xs[0])@self.rounding(xs[1])
            elif op=='linear_dw':result=self.rounding(xs[0]).T@self.rounding(xs[1])
            elif op=='bmm_dx':result=torch.bmm(self.rounding(xs[0]),self.rounding(xs[1]).transpose(-2,-1))
            elif op=='bmm_dw':result=torch.bmm(self.rounding(xs[1]).transpose(-2,-1),self.rounding(xs[0]))
            elif op=='bias_grad':result=xs[0].reshape(-1,xs[0].shape[-1]).sum(0)
            elif op=='relu_backward':result=xs[0]*(xs[1]>0)
            elif op=='add':result=xs[0]+(xs[1] if len(xs)==2 else attrs['scalar'])
            elif op=='mul':result=xs[0]*(xs[1] if len(xs)==2 else attrs['scalar'])
            elif op=='relu':result=torch.relu(xs[0])
            elif op=='copy':result=xs[0].reshape(shape).contiguous()
            elif op=='transpose2d':result=xs[0].transpose(-2,-1).contiguous()
            elif op=='permute':result=xs[0].permute(attrs['axes']).contiguous()
            elif op=='softmax':result=torch.softmax(xs[0],dim=-1)
            elif op=='gelu':result=F.gelu(xs[0],approximate='tanh')
            elif op=='layernorm':result=F.layer_norm(xs[0],(xs[0].shape[-1],),eps=attrs['epsilon'])
            elif op=='layernorm_backward':
                dy,x=xs;center=x-x.mean(dim=-1,keepdim=True);inv=torch.rsqrt(center.square().mean(dim=-1,keepdim=True)+attrs['epsilon'])
                y=center*inv;result=inv*(dy-dy.mean(dim=-1,keepdim=True)-y*(dy*y).mean(dim=-1,keepdim=True))
            elif op=='softmax_backward':
                dy,y=xs;result=y*(dy-(dy*y).sum(dim=-1,keepdim=True))
            elif op=='gelu_backward':
                dy,x=xs;u=(2/math.pi)**.5*(x+.044715*x*x*x);t=torch.tanh(u)
                result=dy*(.5*(1+t)+.5*x*(1-t*t)*(2/math.pi)**.5*(1+3*.044715*x*x))
            elif op=='channel_affine':result=xs[0]*xs[1]+(xs[2] if len(xs)==3 else 0)
            elif op=='channel_mul':result=xs[0]*xs[1]
            elif op=='channel_sum':result=xs[0].reshape(-1,xs[0].shape[-1]).sum(0)
            elif op=='channel_grad':result=(xs[0]*xs[1]).reshape(-1,xs[0].shape[-1]).sum(0)
            elif op=='concat':result=torch.cat(xs,dim=attrs['axis'])
            elif op=='slice_axis':result=xs[0].narrow(attrs['axis'],attrs['start'],shape[attrs['axis']]).contiguous()
            else:raise ValueError('eager comparator does not implement '+op)
            values[out]=result
            if capture_steps:self.snapshots.append((out,result.detach().cpu().clone()))
        return values

    def forward(self):
        self.values=self.evaluate(self.plan['steps'],{**self.weights,**self.inputs})
        self.output_value=self.values[self.output];return self.output_value

    def loss(self):
        self.loss_value=(self.output_value-self.target).square().mean()
        return self.loss_value

    def backward(self):
        self.values[self.plan['backward']['seed_grads'][self.output]]=(self.output_value-self.target)*(2/self.output_value.numel())
        self.values=self.evaluate(self.plan['backward']['steps'],self.values)
        self.gradients={'inputs':{n:self.values[g] for n,g in self.plan['backward']['grad_inputs'].items()},
                        'parameters':{n:self.values[g] for n,g in self.plan['backward']['grad_parameters'].items()}}
        return self.gradients

    def update(self):
        with torch.no_grad():
            for n,g in self.gradients['parameters'].items():self.weights[n].add_(g,alpha=-.01)

    def train(self):
        output=self.forward();self.loss();self.backward();self.update()
        return output

    def fields(self):
        return {'prediction':self.output_value,'loss':self.loss_value,
            **{'dx/'+n:v for n,v in self.gradients['inputs'].items()},
            **{'dw/'+n:v for n,v in self.gradients['parameters'].items()},
            **{'w/'+n:v for n,v in self.weights.items()}}

    def check_guards(self):
        expected=torch.full((4,),-1515870811,device='cuda',dtype=torch.int32)
        for tensor in self.backings:
            assert torch.equal(tensor[:4].view(torch.int32),expected)
            assert torch.equal(tensor[-4:].view(torch.int32),expected)

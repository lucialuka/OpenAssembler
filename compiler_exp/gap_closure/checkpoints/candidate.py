"""Complete owned-state/RNG checkpoint candidate before production implementation."""
import array
import copy
import hashlib
import io
import json
from pathlib import Path
import platform
import random
import zipfile


def pack_state(model,optimizer,cursor):
    import torch
    tensors={};counter=0
    def encode(value):
        nonlocal counter
        if isinstance(value,torch.Tensor):
            counter+=1;name=f'tensors/{counter}.bin'
            raw=bytes(value.tolist()) if value.dtype==torch.uint8 else array.array('f',value.detach().reshape(-1).tolist()).tobytes()
            tensors[name]=raw
            return {'tensor':name,'shape':list(value.shape),'dtype':'u8' if value.dtype==torch.uint8 else 'f32','sha256':hashlib.sha256(raw).hexdigest()}
        if isinstance(value,dict):return {str(k):encode(v) for k,v in value.items()}
        if isinstance(value,(tuple,list)):return [encode(v) for v in value]
        return value
    # Optimizer state uses explicit parameter index, never pickled Python objects.
    state={'weights':model.state_dict(),'optimizer':optimizer.state_dict(),
           'torch_rng':torch.get_rng_state(),'python_rng':random.getstate(),
           'application':{'cursor':cursor,'scheduler':{'last_epoch':cursor,'next_rate':.001},'scaler':None}}
    encoded=encode(state);data=io.BytesIO()
    with zipfile.ZipFile(data,'w',zipfile.ZIP_STORED) as z:
        z.writestr('manifest.json',json.dumps(encoded,allow_nan=False))
        for name,raw in tensors.items():z.writestr(name,raw)
    return data.getvalue()


def restore(raw,model,optimizer):
    import torch
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        def decode(value):
            if isinstance(value,dict) and 'tensor' in value:
                data=z.read(value['tensor']);assert hashlib.sha256(data).hexdigest()==value['sha256']
                return torch.frombuffer(bytearray(data),dtype=torch.uint8 if value['dtype']=='u8' else torch.float32).reshape(value['shape']).clone()
            if isinstance(value,dict):return {k:decode(v) for k,v in value.items()}
            if isinstance(value,list):return [decode(v) for v in value]
            return value
        state=decode(json.loads(z.read('manifest.json')))
    state['optimizer']['state']={int(k):v for k,v in state['optimizer']['state'].items()}
    # Float32 scalar optimizer step is an explicit tensor too.
    model.load_state_dict(state['weights']);optimizer.load_state_dict(state['optimizer'])
    torch.set_rng_state(state['torch_rng']);py=state['python_rng'];random.setstate((py[0],tuple(py[1]),py[2]))
    return state['application']


def run():
    import torch
    torch.set_num_threads(1)
    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__();self.weight=torch.nn.Parameter(torch.linspace(-.5,.5,17));self.bias=torch.nn.Parameter(torch.zeros(17));self.register_buffer('offset',torch.linspace(0,.1,17))
        def forward(self,x):return self.weight*x+self.bias+self.offset
    rows=[]
    for kind,extra in [('sgd',{'momentum':.9,'nesterov':True}),('adam',{'amsgrad':True,'maximize':True}),('adamw',{'amsgrad':True})]:
        torch.manual_seed(35052);random.seed(35052);model=Net()
        cls={'sgd':torch.optim.SGD,'adam':torch.optim.Adam,'adamw':torch.optim.AdamW}[kind]
        opt=cls(model.parameters(),lr=.001,foreach=False,**extra)
        checkpoint=None;expected_outputs=[]
        for step in range(24):
            x=torch.randn(17)*.1;target=torch.randn(17)*.1+random.random()*.01
            opt.zero_grad();y=model(x);((y-target)**2).mean().backward();opt.step();expected_outputs.append(y.detach().clone())
            if step==11:checkpoint=pack_state(model,opt,12)
        final=pack_state(model,opt,24)
        resumed=Net();other=cls(resumed.parameters(),lr=.001,foreach=False,**extra)
        application=restore(checkpoint,resumed,other);assert application['cursor']==12
        for step in range(application['cursor'],24):
            x=torch.randn(17)*.1;target=torch.randn(17)*.1+random.random()*.01
            other.zero_grad();y=resumed(x);((y-target)**2).mean().backward();other.step()
            torch.testing.assert_close(y,expected_outputs[step],rtol=0,atol=0)
        actual=pack_state(resumed,other,24)
        # ZIP headers contain current wall time; compare every owned content byte.
        with zipfile.ZipFile(io.BytesIO(final)) as a,zipfile.ZipFile(io.BytesIO(actual)) as b:
            assert a.namelist()==b.namelist()
            assert all(a.read(n)==b.read(n) for n in a.namelist())
        rows.append({'optimizer':kind,'passed':True,'updates':24,'resume_cursor':12,'all_tensor_rng_json_bytes_equal':True})
    return {'schema':1,'gap':'G35','passed':True,'python':platform.python_version(),'torch':torch.__version__,
            'seed':35052,'candidate_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'rows':rows,'hardware':'not run by this CPU serialization candidate'}
if __name__=='__main__':
    result=run();Path(__file__).with_name('candidate-results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

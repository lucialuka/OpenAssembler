import torch
from openassembler.modelfront.decode import DecodeSession

class Prefill(torch.nn.Module):
    def forward(self,x):return torch.nn.functional.scaled_dot_product_attention(x,x,x,is_causal=True),x.clone(),x.clone()
class Decoder(torch.nn.Module):
    def forward(self,x,past_k,past_v):
        k=torch.cat((past_k,x),dim=-2);v=torch.cat((past_v,x),dim=-2)
        return torch.nn.functional.scaled_dot_product_attention(x,k,v),k,v

def check_decode(root,make,check):
    prefix=torch.randn(2,2,3,5)*.25
    pp,pr=make(Prefill(),(prefix,),'prefill')
    buckets={}
    for length in (3,4,5,6):
        _,rt=make(Decoder(),(torch.randn(2,2,1,5),torch.randn(2,2,length,5),torch.randn(2,2,length,5)),f'decode-{length}')
        buckets[length]=rt
    session=DecodeSession(pr,buckets)
    records=[]
    try:
        prediction=session.prefill({'x':prefix.cuda()});full=prefix
        check(prediction,torch.nn.functional.scaled_dot_product_attention(full,full,full,is_causal=True),'cache/prefill')
        for length in (3,4,5,6):
            token=torch.randn(2,2,1,5)*.25;full=torch.cat((full,token),dim=-2)
            prediction=session.step({'x':token.cuda()})
            expected=torch.nn.functional.scaled_dot_product_attention(full,full,full,is_causal=True)
            check(prediction,expected[:,:,-1:,:],'cache/'+str(length)+'/prediction')
            for i,cache in enumerate(session.cache):check(cache,full,f'cache/{length}/{i}')
            records.append({'length':length,'guards':buckets[length].check_canaries(),'launches':list(buckets[length].launch_ledger)})
        before=session.cache
        try:session.step({'x':token.cuda()})
        except ValueError:pass
        else:raise AssertionError('unknown bucket admitted')
        assert session.cache is before
        session.cache[0].add_(1)
        try:session.step({'x':token.cuda()})
        except ValueError:pass
        else:raise AssertionError('mutated cache admitted')
        session.reset();assert session.cache is None
    finally:session.close()
    return records

"""Independent complete-state correctness before paired full-training timing."""
import json,math,random,statistics,sys,time,subprocess
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch
import torch
from openassembler.modelfront import kernels_tensor,runtime,kernels
from openassembler.modelfront.graph import capture_model,extract_weights
from openassembler.modelfront.optimization import optimize_plan
from openassembler.modelfront.reference import CPUReference
from openassembler.modelfront.profiling import audit_profile
from exp.pack_training.candidate import recipe,dual_pack,ReuseRuntime


class MLP(torch.nn.Module):
    def __init__(self,k,n):
        super().__init__();self.up=torch.nn.Linear(k,n);self.down=torch.nn.Linear(n,k)
    def forward(self,x):return self.down(torch.relu(self.up(x)))+x
class Branch(torch.nn.Module):
    def __init__(self,k,n):
        super().__init__();self.q=torch.nn.Linear(k,n);self.k=torch.nn.Linear(k,n);self.v=torch.nn.Linear(k,n)
    def forward(self,x):return self.q(x)+self.k(x)+self.v(x)


def probe():
    import struct
    from openassembler.modelfront.kernels_tensor import build_tensor_kernels,TensorRecipe
    from openassembler.gpu.cuda_driver import Driver
    records=[];torch.empty(1,device='cuda');driver=Driver()
    for shape in [(17,65),(128,128),(129,130),(256,128)]:
        stage,pr,pc=dual_pack(shape,'packed')
        folder=Path('results/probe')/f'{shape[0]}-{shape[1]}';folder.mkdir(parents=True,exist_ok=True)
        rec=TensorRecipe('probe','probe',{}, {}, (stage,))
        inventory=build_tensor_kernels(folder,[rec],sass_listings=False)
        module=driver.load((folder/inventory[stage.name]['module']).read_bytes());fn=driver.function(module,stage.name)
        count=math.prod(shape)
        specials=[0,0x80000000,1,0x80000001,0x33000000,0x33000001,0x33800000,0x387fffff,0x38800000,
                  0x3f801000,0x3f803000,0xbf801000,0x477fe000,0x477ff000,0x7f800000,0xff800000,
                  0x7fc00001,0xffc00001,0x007fffff,0x807fffff]
        rng=random.Random(1911)
        for pattern in ['special','raw','normal']:
            bits=[specials[i%len(specials)] if pattern=='special' else rng.randrange(2**32) for i in range(count)]
            values=torch.frombuffer(bytearray(struct.pack('<'+'I'*count,*bits)),dtype=torch.float32).reshape(shape).clone()
            if pattern=='normal':values=torch.randn(shape)*.2;bits=list(struct.unpack('<'+'I'*count,values.numpy().tobytes()))
            source=values.cuda();backing=torch.empty(2*pr*pc+16,device='cuda',dtype=torch.float16)
            driver.memset(backing.data_ptr(),0xA5,backing.numel()*2);dest=backing[8:-8]
            launch(driver,fn,stage.launch,[source,dest]);torch.cuda.synchronize()
            expect=[]
            for raw in bits:
                if raw&0x7fffffff>0x7f800000:h=(raw>>16&0x8000)|0x7e00
                else:
                    f=struct.unpack('<f',struct.pack('<I',raw))[0]
                    try:h=int.from_bytes(struct.pack('<e',f),'little')
                    except OverflowError:h=(raw>>16&0x8000)|0x7c00
                expect.append(h if h<32768 else h-65536)
            expected=torch.zeros(pr,pc,dtype=torch.int16);expected[:shape[0],:shape[1]]=torch.tensor(expect,dtype=torch.int16).view(shape)
            actual=dest.view(torch.int16).cpu()
            assert torch.equal(actual[:pr*pc].view(pr,pc),expected),(shape,pattern,'row')
            assert torch.equal(actual[pr*pc:].view(pc,pr),expected.T),(shape,pattern,'transpose')
            assert torch.equal(source.view(torch.int32).cpu(),values.view(torch.int32))
            assert driver.dtoh(backing.data_ptr(),16)==b'\xa5'*16
            assert driver.dtoh(backing.data_ptr()+backing.numel()*2-16,16)==b'\xa5'*16
            records.append({'shape':shape,'padded':[pr,pc],'pattern':pattern,'exact':True,'guards':True})
        driver.unload(module)
    return {'passed':True,'records':records}


def launch(driver,fn,contract,tensors):
    import ctypes as C
    args=[C.c_uint64(t.data_ptr()) for t in tensors];params=(C.c_void_p*len(args))(*[C.cast(C.pointer(x),C.c_void_p) for x in args]);c=contract
    driver.ck(driver.cu.cuLaunchKernel(fn,*c['grid'],c['threads'],1,1,c.get('smem',0),C.c_void_p(torch.cuda.current_stream().cuda_stream),params,None),'pack trial')


def fields(run,result):
    return {'prediction':result['prediction'],'loss':result['loss'].reshape(()),
        **{'dx/'+n:t for n,t in result['gradients']['inputs'].items()},
        **{'dw/'+n:t for n,t in result['gradients']['parameters'].items()},**{'w/'+n:t for n,t in run.weights.items()}}


def paired(functions,repeats=4,samples=15):
    graphs={};stream=torch.cuda.current_stream()
    for name,fn in functions.items():
        for _ in range(3):fn()
        stream.synchronize();g=torch.cuda.CUDAGraph()
        with torch.cuda.graph(g,stream=stream):
            for _ in range(repeats):fn()
        graphs[name]=g
    rows=[];rng=random.Random(9117)
    for round_ in range(3):
        data={n:[] for n in functions};orders=[]
        for _ in range(samples):
            order=list(functions);rng.shuffle(order);orders.append(order)
            for n in order:
                a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                a.record();graphs[n].replay();b.record();b.synchronize();data[n].append(a.elapsed_time(b)/repeats)
        rows.append({'round':round_,'orders':orders,'samples_ms':data,'medians_ms':{n:statistics.median(v) for n,v in data.items()}})
    return rows


def graph_case(label,shape,n,branch=False):
    hidden=n;k=shape[-1];model=(Branch(k,n) if branch else MLP(k,n));plan=optimize_plan(capture_model(model,[torch.randn(shape)*.15],math_mode='f16_rne_f32_accum'),reuse_storage=True)
    weights=extract_weights(model,plan);root=Path('results')/label;root.mkdir(parents=True)
    reference=CPUReference(plan,weights);runs={};patcher=ExitStack();original=kernels_tensor.emit_tensor_recipe;enabled=[False]
    def dispatch(*args,**kwargs):return recipe(*args,**kwargs) if enabled[0] else original(*args,**kwargs)
    production=globals().get('PRODUCTION',False)
    if not production:patcher.enter_context(patch.object(kernels_tensor,'emit_tensor_recipe',dispatch))
    available=True
    try:kernels.prepare(plan)
    except ValueError:available=False
    if available:
        kernels.build_bundle(plan,weights,root/'baseline',sass_listings=False,cache=False);runs['baseline']=runtime.Runtime.load(root/'baseline')
    enabled[0]=True
    candidate_plan={**plan,'tensor_packing':'dual'} if production else plan
    kernels.build_bundle(candidate_plan,weights,root/'candidate',sass_listings=False,cache=False)
    runs['candidate']=(runtime.Runtime if production else ReuseRuntime).load(root/'candidate')
    input_name=plan['inputs'][0]['name'];output=plan['outputs'][0];outputshape=plan['buffers'][output]['shape']
    inputs={input_name:torch.empty(shape,device='cuda')};target=torch.empty(outputshape,device='cuda');checks=[];bitwise_differences=[]
    for it in range(4):
        x=torch.randn(shape)*.15;y=torch.randn(outputshape)*.15;inputs[input_name].copy_(x);target.copy_(y)
        ref=reference.train_step({input_name:x},y,.01)
        expected={'prediction':ref['prediction'],'loss':ref['loss'].reshape(()),
            **{'dx/'+n:v for n,v in ref['gradients']['inputs'].items()},**{'dw/'+n:v for n,v in ref['gradients']['parameters'].items()},
            **{'w/'+n:v for n,v in reference.weights.items()}}
        previous=None
        for name,run in runs.items():
            enabled[0]=name=='candidate'
            if it:run.poison_workspace()
            result=run.train_step(inputs,target,.01);run.synchronize();values={n:v.detach().cpu().clone() for n,v in fields(run,result).items()}
            for field,actual in values.items():
                good=torch.isclose(actual,expected[field],rtol=.002,atol=2e-5)
                assert bool(good.all()),(label,it,name,field,float((actual-expected[field]).abs().max()))
                if previous is not None and not torch.equal(actual.view(torch.int32),previous[field].view(torch.int32)):
                    bitwise_differences.append({'iteration':it,'field':field,'max_abs':float((actual-previous[field]).abs().max())})
                    if label not in ('wide','maximum'):raise AssertionError((label,it,'bitwise',field))
            previous=values;run.check_canaries();assert torch.equal(inputs[input_name].cpu(),x) and torch.equal(target.cpu(),y)
            checks.append({'iteration':it,'method':name,'fields':len(values),'passed':True})
    audit={};counts={};parts={}
    for name,run in runs.items():
        enabled[0]=name=='candidate'
        for attempt in range(3):
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
                run.train_step(inputs,target,.01);run.synchronize()
            try:audit[name]=audit_profile(prof,run.launch_ledger)
            except ValueError:
                if attempt==2:raise
            else:break
        counts[name]=len(run.launch_ledger)
        from collections import Counter
        stages={s.name:s.op for r in run.recipes.values() for s in r.stages}
        parts[name]=dict(Counter(stages.get(s['entry'],'other') for s in run.launch_ledger))
    # Each graph records fresh per-iteration packs; Python caches execute during capture only.
    functions={}
    for name,run in runs.items():
        def fn(name=name,run=run):
            enabled[0]=name=='candidate'
            return run.train_step(inputs,target,.01)
        functions[name]=fn
    timings=paired(functions)
    # Validate actual changed-input captured replay, not only uncaptured checks.
    replay_checks=[]
    for name,run in runs.items():
        enabled[0]=name=='candidate'
        with torch.no_grad():
            for n,w in run.weights.items():w.copy_(weights[n].to('cuda'))
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph,stream=torch.cuda.current_stream()):result=run.train_step(inputs,target,.01)
        refmodel=CPUReference(plan,weights)
        for it in range(3):
            x=torch.randn(shape)*.15;y=torch.randn(outputshape)*.15;inputs[input_name].copy_(x);target.copy_(y)
            run.poison_workspace();graph.replay();run.synchronize();ref=refmodel.train_step({input_name:x},y,.01)
            for category in ('inputs','parameters'):
                for key,tensor in result['gradients'][category].items():torch.testing.assert_close(tensor.cpu(),ref['gradients'][category][key],rtol=.002,atol=2e-5)
            torch.testing.assert_close(result['prediction'].cpu(),ref['prediction'],rtol=.002,atol=2e-5)
            torch.testing.assert_close(result['loss'].cpu().reshape(()),ref['loss'].reshape(()),rtol=.002,atol=2e-5)
            assert torch.equal(inputs[input_name].cpu(),x) and torch.equal(target.cpu(),y)
            for key,tensor in run.weights.items():torch.testing.assert_close(tensor.cpu(),refmodel.weights[key],rtol=.002,atol=2e-5)
            run.check_canaries();replay_checks.append({'method':name,'iteration':it,'passed':True})
    record={'label':label,'shape':list(shape),'hidden':hidden,'branch':branch,'existing_contract':available,'checks':checks,'replay_checks':replay_checks,
        'counts':counts,'stages':parts,'audits':audit,'timings':timings,'device':torch.cuda.get_device_name(),
        'bitwise_differences':bitwise_differences,'allocated_bytes':{name:sum(t.numel()*t.element_size() for t in run._allocations) for name,run in runs.items()}}
    if available:record['speedups']=[r['medians_ms']['baseline']/r['medians_ms']['candidate'] for r in timings]
    for r in runs.values():r.close()
    patcher.close();(root/'result.json').write_text(json.dumps(record,indent=2));print(label,record.get('speedups'),counts,flush=True);return record


def main(phase):
    global PRODUCTION
    PRODUCTION=phase.startswith('production')
    torch.set_num_threads(1);torch.manual_seed(9913);torch.backends.cuda.matmul.allow_tf32=False
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream());torch.cuda.set_stream(stream)
    Path('results').mkdir(exist_ok=True)
    cases=[('mlp128',(128,128),256,False),('qkv128',(128,128),128,True),('mlp256',(256,128),256,False),
           ('batch',(2,64,128),256,False),('tail',(17,65),33,False),('odd',(129,130),127,False),('rank3_tail',(2,7,65),33,False),
           ('single',(1,129),65,False),('wide',(128,256),512,False),('maximum',(128,1024),1024,False)]
    if phase=='bridge':
        import tempfile,base64
        from openassembler.modelfront.frontend import load_config
        from openassembler.modelfront.worker import process
        from openassembler.torchfront.archive import pack,unpack
        from openassembler.torchfront.frontend import source_digest
        records=[]
        for mode in ['direct','cuda_graph']:
            with tempfile.TemporaryDirectory() as d:
                source=Path(d);(source/'model.py').write_text('import torch\ndef create_model():\n return torch.nn.Sequential(torch.nn.Linear(65,33),torch.nn.ReLU(),torch.nn.Linear(33,65))\n')
                (source/'openassembler.toml').write_text('[model]\nentrypoint="model.py"\ninput_shapes=[[17,65]]\nmath="f16_rne_f32_accum"\ntensor_packing="dual"\nmode="training"\nexecution="'+mode+'"\nreuse_storage=true\nsass_listings=false\nrtol=0.002\natol=0.00002\nsteps=4\n')
                config=load_config(source);files={p.name:p.read_bytes() for p in source.iterdir()}
                answer=process({'action':'compile','config':config,'source_root':str(source),'source_digest':source_digest(files),'source':base64.b64encode(pack(files)).decode(),'verify':True})
                assert 'error' not in answer,answer
                output=Path('results/bridge')/mode;unpack(base64.b64decode(answer.pop('artifacts')),output)
                records.append({'execution':mode,'report':answer['report']})
        result={'passed':True,'records':records}
    elif phase=='probe':result=probe()
    elif phase.startswith(('case:','production:')):
        label=phase.split(':',1)[1];case=next(c for c in cases if c[0]==label)
        record=graph_case(*case);return
    else:
        records=[]
        selected=cases[7:9] if phase.endswith('heldout') else [cases[1]] if phase.endswith('repeat') else cases[:7]
        for case in selected:
            proc=subprocess.run([sys.executable,'-m','exp.pack_training.campaign',('production:' if PRODUCTION else 'case:')+case[0]],capture_output=True,text=True,timeout=210)
            Path('results',case[0]+'.log').write_text(proc.stdout+proc.stderr)
            path=Path('results',case[0],'result.json')
            if not path.is_file():raise RuntimeError(proc.stdout+proc.stderr)
            records.append(json.loads(path.read_text()))
        result={'passed':True,'records':records}
    Path('results/result.json').write_text(json.dumps(result,indent=2,allow_nan=False))
if __name__=='__main__':main(sys.argv[1] if len(sys.argv)>1 else 'probe')

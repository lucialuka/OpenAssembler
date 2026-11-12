"""Build and validate native FP16-boundary linear recipes on one bounded A100."""
from __future__ import annotations
from dataclasses import asdict
from datetime import datetime, timezone
import base64
import hashlib
import json
from pathlib import Path
import modal

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1] if len(HERE.parents)>1 else Path.cwd()
app=modal.App("openassembler-native-model-tensor")
image=(modal.Image.from_registry("pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel")
       .add_local_python_source("openassembler"))


def build(tag, address_cse=False):
    from openassembler.modelfront.kernels_tensor import emit_tensor_recipe,build_tensor_kernels,MATH_MODE
    folder=HERE/"builds"/tag;folder.mkdir(parents=True,exist_ok=False)
    def spec(shape):return {"shape":list(shape),"dtype":"f32"}
    recipes=[]
    for m,n,k in [(128,128,128),(128,256,512),(1024,512,512)]:
        for op,inputs,output in [
            ("linear",[(m,k),(n,k),(n,)],(m,n)),
            ("linear_dx",[(m,n),(n,k)],(m,k)),
            ("linear_dw",[(m,n),(m,k)],(n,k)),
        ]:
            recipes.append(emit_tensor_recipe(op,[spec(x) for x in inputs],spec(output),{"math_mode":MATH_MODE}))
    if address_cse:
        from exp.native_model_tensor.address_reuse import recipe_candidate
        recipes += [recipe_candidate(recipe) for recipe in recipes]
    records=build_tensor_kernels(folder,recipes)
    payload={"schema":1,"tag":tag,"math_mode":MATH_MODE,"recipes":[asdict(x) for x in recipes],"kernels":records,
             "binaries":{name:base64.b64encode((folder/r['module']).read_bytes()).decode() for name,r in records.items()},
             "references":{name:base64.b64encode((folder/r['module']).with_name('reference.cubin').read_bytes()).decode()
                           for name,r in records.items()},
             "compiler_source_sha256":{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                                        for p in (ROOT/'openassembler').rglob('*.py')}}
    (folder/'payload.json').write_text(json.dumps(payload,indent=2))
    return folder/'payload.json'


@app.function(image=image,gpu='A100-40GB',timeout=600,max_containers=1,retries=0)
def execute(payload):
    import ctypes as C
    import random
    import statistics
    import struct
    import torch
    from openassembler.gpu.cuda_driver import Driver
    torch.cuda.init();anchor=torch.empty(1,device='cuda');torch.manual_seed(20261112)
    torch.backends.cuda.matmul.allow_tf32=False
    driver=Driver();modules={};functions={}
    methods=('oa','ptxas') if payload.get('references') else ('oa',)
    for name,r in payload['kernels'].items():
        for method in methods:
            binary=base64.b64decode(payload['binaries' if method=='oa' else 'references'][name])
            if method=='oa':assert hashlib.sha256(binary).hexdigest()==r['cubin_sha256']
            key=(name,method);modules[key]=driver.load(binary);functions[key]=driver.function(modules[key],name)
            if method=='oa':assert driver.func_attrs(functions[key])['num_regs']==r['registers']
    def launch(stage,values,method='oa'):
        args=[C.c_uint64(values[x].data_ptr()) for x in [*stage['inputs'],stage['output']]]
        params=(C.c_void_p*len(args))(*[C.cast(C.pointer(a),C.c_void_p) for a in args])
        cfg=stage['launch']
        fn=functions[(stage['name'],method)]
        if cfg['smem']>49152:driver.ck(driver.cu.cuFuncSetAttribute(fn,8,cfg['smem']),'dynamic shared opt-in')
        driver.ck(driver.cu.cuLaunchKernel(fn,*cfg['grid'],cfg['threads'],1,1,cfg['smem'],
                                          C.c_void_p(torch.cuda.current_stream().cuda_stream),params,None),'native tensor stage')
    results=[];timings=[];conversion=[];retained_graphs={};paired=[]
    try:
        conversion_stages={}
        for recipe in payload['recipes']:
            for stage in recipe['stages']:
                if stage['op']=='pack_f16_rne':
                    key=(stage['signature']['transpose'],stage['signature'].get('address_policy','original'))
                    conversion_stages.setdefault(key,stage)
        for stage in conversion_stages.values():
            shape=stage['signature']['shape'];count=shape[0]*shape[1]
            rng=random.Random(719)
            edges=[0,0x80000000,1,0x80000001,0x33000000,0x33000001,0x33800000,0x387fffff,
                   0x38800000,0x3f801000,0x3f803000,0xbf801000,0x477fe000,0x477ff000,
                   0x7f800000,0xff800000,0x7fc00001,0xffc00001]
            bits=edges+[rng.getrandbits(32) for _ in range(count-len(edges))]
            halves=[]
            for raw in bits:
                value=struct.unpack('<f',struct.pack('<I',raw))[0]
                if raw&0x7fffffff>0x7f800000:h=(raw>>16&0x8000)|0x7e00
                else:
                    try:h=int.from_bytes(struct.pack('<e',value),'little')
                    except OverflowError:h=(raw>>16&0x8000)|0x7c00
                halves.append(h if h<32768 else h-65536)
            x=torch.tensor([v if v<0x80000000 else v-0x100000000 for v in bits],dtype=torch.int32,device='cuda').view(torch.float32).view(shape)
            y=torch.empty(shape[::-1] if stage['signature']['transpose'] else shape,dtype=torch.float16,device='cuda')
            expected=torch.tensor(halves,dtype=torch.int16,device='cuda').view(shape)
            if stage['signature']['transpose']:expected=expected.T.contiguous()
            for method in methods:
                values={stage['inputs'][0]:x,stage['output']:y};launch(stage,values,method);torch.cuda.synchronize()
                conversion.append({'entry':stage['name'],'method':method,'count':count,
                                   'passed':bool((y.view(torch.int16)==expected).all())})
        for recipe in payload['recipes']:
            shapes=recipe['signature']['inputs'];outshape=recipe['signature']['output']
            for pattern in ('zero','integer','normal'):
                values={};storages={}
                for i,shape in enumerate(shapes):
                    if pattern=='zero':x=torch.zeros(shape,device='cuda')
                    elif pattern=='integer':x=torch.randint(-2,3,shape,device='cuda').float()
                    else:x=torch.randn(shape,device='cuda')*.1
                    values['a'+str(i)]=x
                original={k:v.clone() for k,v in values.items()}
                for name,spec in {**recipe['workspaces'],'output':{'shape':outshape,'dtype':'f32'}}.items():
                    dtype=torch.float16 if spec['dtype']=='f16' else torch.float32
                    count=1
                    for d in spec['shape']:count*=d
                    storage=torch.full((count+256,),-123.,dtype=dtype,device='cuda')
                    storages[name]=storage;values[name]=storage[128:-128].view(spec['shape'])
                a,b=values['a0'],values['a1']
                if recipe['op']=='linear':expected=a.half().double()@b.half().double().T
                elif recipe['op']=='linear_dx':expected=a.half().double()@b.half().double()
                else:expected=a.half().double().T@b.half().double()
                if len(shapes)==3:expected=expected+values['a2'].double()
                pack_checks=[]
                for repeat in range(3):
                    for name in storages:values[name].fill_(float('nan'))
                    for stage in recipe['stages']:
                        launch(stage,values)
                        if stage['op']=='pack_f16_rne':
                            shape=stage['signature']['shape'];x=values[stage['inputs'][0]]
                            truth=(x.T.contiguous() if stage['signature']['transpose'] else x).half()
                            torch.cuda.synchronize()
                            pack_checks.append(bool((values[stage['output']].view(torch.int16)==truth.view(torch.int16)).all()))
                    torch.cuda.synchronize()
                    actual=values['output'].double();difference=(actual-expected).abs()
                    relative_l2=float(torch.linalg.vector_norm(actual-expected)/torch.linalg.vector_norm(expected).clamp_min(1e-30))
                    finite=bool(torch.isfinite(actual).all())
                    exact=bool((actual==expected).all())
                    canaries=all(bool((v[:128]==-123).all() and (v[-128:]==-123).all()) for v in storages.values())
                    immutable=all(bool((values[k]==v).all()) for k,v in original.items())
                    bitwise=True
                    if 'ptxas' in methods:
                        native_bits=values['output'].view(torch.int32).clone()
                        for stage in recipe['stages']:launch(stage,values,'ptxas')
                        torch.cuda.synchronize();bitwise=bool((native_bits==values['output'].view(torch.int32)).all())
                    passed=all(pack_checks) and finite and canaries and immutable and bitwise and (exact if pattern!='normal' else relative_l2<2e-5)
                    results.append({'recipe':recipe['name'],'op':recipe['op'],'shape':recipe['signature'],
                                    'pattern':pattern,'repeat':repeat,'pack_bitwise':all(pack_checks),
                                    'finite':finite,'canaries':canaries,'inputs_immutable':immutable,
                                    'same_source_ptxas_bitwise':bitwise,
                                    'relative_l2_fp64':relative_l2,'max_absolute_error':float(difference.max()),
                                    'exact_fp64':exact,'passed':passed})
            graphs={}
            for method in methods:
                g=torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    for _ in range(5):
                        for stage in recipe['stages']:launch(stage,values,method)
                graphs[method]=g
                values['output'].fill_(float('nan'));g.replay();torch.cuda.synchronize()
                assert bool(torch.isfinite(values['output']).all())
            # CUDA graph capture retains raw addresses, so keep all backing
            # tensors alive until both original/candidate graphs are measured.
            retained_graphs[recipe['name']]={'graphs':graphs,'values':values,'storages':storages}
            for round_index in range(3):
                samples={method:[] for method in methods}
                order=[method for _ in range(10) for method in methods];random.Random(300+round_index).shuffle(order)
                for method in order:
                    begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                    begin.record();graphs[method].replay();end.record();end.synchronize()
                    samples[method].append(begin.elapsed_time(end)/5)
                timings.append({'recipe':recipe['name'],'op':recipe['op'],'round':round_index,
                                'samples_ms':samples,'median_ms':{k:statistics.median(v) for k,v in samples.items()},
                                'scope':'complete native pack+GEMM+optional-bias pipeline, five invocations per CUDA graph'})
            print(recipe['op'],outshape,'passed',all(x['passed'] for x in results if x['recipe']==recipe['name']),flush=True)
        for name,baseline in retained_graphs.items():
            if name+'_pack_cse' not in retained_graphs:continue
            candidate=retained_graphs[name+'_pack_cse']
            runners={'oa_original':baseline['graphs']['oa'],'oa_pack_cse':candidate['graphs']['oa'],
                     'ptxas_original':baseline['graphs']['ptxas'],'ptxas_pack_cse':candidate['graphs']['ptxas']}
            for round_index in range(3):
                samples={method:[] for method in runners}
                order=[method for _ in range(15) for method in runners];random.Random(901+round_index).shuffle(order)
                for method in order:
                    begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                    begin.record();runners[method].replay();end.record();end.synchronize()
                    samples[method].append(begin.elapsed_time(end)/5)
                medians={k:statistics.median(v) for k,v in samples.items()}
                paired.append({'baseline_recipe':name,'round':round_index,'samples_ms':samples,'median_ms':medians,
                               'oa_baseline_over_candidate':medians['oa_original']/medians['oa_pack_cse'],
                               'scope':'shuffled four-method complete pipeline; paired original/CSE/native/ptxas; five invocations per graph'})
    finally:
        torch.cuda.synchronize()
        for module in modules.values():driver.unload(module)
    return {'schema':1,'tag':payload['tag'],'device':torch.cuda.get_device_name(),'math_mode':payload['math_mode'],
            'torch':torch.__version__,'cuda':torch.version.cuda,'results':results,
            'conversion_checks':conversion,'timings':timings,
            'paired_address_ablation':paired,
            'compiler_source_sha256':payload['compiler_source_sha256'],'kernels':payload['kernels']}


@app.local_entrypoint()
def main(tag:str='tensor-linear',payload:str='',address_cse:bool=False):
    path=Path(payload) if payload else build(tag,address_cse)
    value=json.loads(path.read_text())
    if 'references' not in value:
        folder=path.parent
        value['references']={name:base64.b64encode((folder/r['module']).with_name('reference.cubin').read_bytes()).decode()
                             for name,r in value['kernels'].items()}
    result=execute.remote(value)
    folder=HERE/'runs';folder.mkdir(exist_ok=True)
    target=folder/(tag+'-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'.json')
    target.write_text(json.dumps(result,indent=2));print(target)

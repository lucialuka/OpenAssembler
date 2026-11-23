"""Versioned complete owned-state checkpoints: JSON + bounded raw tensor bytes.

No pickle or executable host payload. Validation and allocation precede a single
binding commit. Checkpoints belong to exact saved native programs; application
cursor/scheduler/scaler values are explicit finite JSON data returned to callers.
"""
from __future__ import annotations

import array
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace
import zipfile

from .kernels import canonical
from .optimizer import optimizer_options

FORMAT='openassembler-owned-checkpoint'
VERSION=1
MAX_MANIFEST_BYTES=1024*1024


def _hash(raw):return hashlib.sha256(raw).hexdigest()


def _json_value(value):
    if value is None or type(value) in (str,bool,int):return True
    if type(value) is float:return math.isfinite(value)
    if type(value) is list:return all(_json_value(v) for v in value)
    if type(value) is dict:return all(type(k)is str and _json_value(v) for k,v in value.items())
    return False


def _application(value):
    value={} if value is None else value
    if type(value)is not dict or not _json_value(value):raise ValueError('checkpoint application state must be finite JSON data')
    return json.loads(canonical(value))


def _program(runtime):
    return {'arch':'sm_80','plan_sha256':_hash(canonical(runtime.plan)),
            'source_sha256':runtime.ledger['source_sha256'],'cubin_sha256':runtime.ledger['cubin_sha256'],
            'tensor_cubins':{n:r['cubin_sha256'] for n,r in runtime.ledger.get('tensor_kernels',{}).items()}}


def _boundary(runtime):
    if (runtime._closed or getattr(runtime,'_checkpoint_replay_owner',False)
            or runtime._tape is not None or runtime._gradients is not None
            or runtime._backward_ready):
        raise ValueError('checkpoint requires an ordinary live runtime at a completed update or discarded-inference boundary')
    if runtime._device is not None and runtime.torch.cuda.current_stream(runtime._device).cuda_stream!=runtime._stream_id:
        raise ValueError('checkpoint requires the runtime initialization CUDA stream')


def _raw_f32(value,torch,shape):
    if not isinstance(value,torch.Tensor) or value.dtype!=torch.float32 or list(value.shape)!=list(shape):
        raise ValueError('checkpoint needs complete FP32 model and optimizer tensors')
    cpu=value.detach().cpu().contiguous()
    raw=bytes(cpu.reshape(-1).view(torch.uint8).tolist())
    if sys.byteorder!='little':
        data=array.array('I');data.frombytes(raw);data.byteswap();raw=data.tobytes()
    return raw


def _tensor_path(role):return 'tensors/'+_hash(role.encode())+'.bin'


def _encode_tensor(role,raw,shape,dtype,files):
    path=_tensor_path(role);files[path]=raw
    return {'path':path,'shape':list(shape),'dtype':dtype,'sha256':_hash(raw)}


def _python_rng(state):
    if (not isinstance(state,(list,tuple)) or len(state)!=3 or type(state[0])is not int or state[0]!=3
            or not isinstance(state[1],(list,tuple)) or len(state[1])!=625
            or any(type(v)is not int or not 0<=v<2**32 for v in state[1][:-1])
            or type(state[1][-1])is not int or not 0<=state[1][-1]<=624
            or state[2] is not None and (type(state[2])is not float or not math.isfinite(state[2]))):
        raise ValueError('invalid Python CPU RNG state')
    normalized=(state[0],tuple(state[1]),state[2]);random.Random().setstate(normalized)
    return normalized


def save_checkpoint(runtime,path,*,application=None,include_rng=True):
    _boundary(runtime)
    if runtime._device is None:raise ValueError('initialize the native runtime before saving a checkpoint')
    if type(include_rng)is not bool:raise ValueError('include_rng must be Boolean')
    application=_application(application);runtime.synchronize();torch=runtime.torch;files={}
    expected={p['name']:p for p in runtime.plan['parameters']}
    if set(runtime.weights)!=set(expected):raise ValueError('checkpoint model state inventory mismatch')
    model={n:_encode_tensor('model/'+n,_raw_f32(runtime.weights[n],torch,s['shape']),s['shape'],'f32',files) for n,s in expected.items()}
    state=runtime.optimizer_state_dict();runtime._validate_optimizer_state_dict(state,allow_nonfinite=True)
    encoded={'schema':state['schema'],'options':state['options'],'step':state['step'],'moments':{}}
    for n,pair in state['moments'].items():
        encoded['moments'][n]={k:_encode_tensor('optimizer/'+n+'/'+k,_raw_f32(v,torch,expected[n]['shape']),expected[n]['shape'],'f32',files) for k,v in pair.items()}
    rng=None
    if include_rng:
        py=_python_rng(random.getstate());raw=bytes(torch.get_rng_state().tolist())
        rng={'python':py,'python_runtime':list(sys.version_info[:2]),'torch_runtime':torch.__version__.split('+')[0],
             'torch':_encode_tensor('rng/torch',raw,[len(raw)],'u8',files)}
    manifest={'format':FORMAT,'version':VERSION,'program':_program(runtime),'model':model,
              'optimizer':encoded,'rng':rng,'application':application}
    data=canonical(manifest)
    if len(data)>MAX_MANIFEST_BYTES:raise ValueError('checkpoint manifest exceeds its 1 MiB format bound')
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=None
    try:
        with tempfile.NamedTemporaryFile(prefix='.'+path.name+'.',suffix='.tmp',dir=path.parent,delete=False) as handle:
            temporary=Path(handle.name)
            with zipfile.ZipFile(handle,'w',zipfile.ZIP_STORED) as z:
                for name,raw in sorted({'manifest.json':data,**files}.items()):
                    z.writestr(zipfile.ZipInfo(name,date_time=(1980,1,1,0,0,0)),raw)
            handle.flush();os.fsync(handle.fileno())
        digest=_hash(temporary.read_bytes())
        os.replace(temporary,path)
    finally:
        if temporary is not None:temporary.unlink(missing_ok=True)
    return {'format':FORMAT,'version':VERSION,'path':str(path),'step':state['step'],
            'sha256':digest,'application':application,'rng_included':include_rng}


def read_checkpoint(runtime,path,*,restore_rng=True):
    """Fully validate CPU payloads before any device allocation or RNG mutation."""
    if type(restore_rng)is not bool:raise ValueError('restore_rng must be Boolean')
    torch=runtime.torch;expected={p['name']:p for p in runtime.plan['parameters']};used={'manifest.json'}
    try:
        with zipfile.ZipFile(path) as z:
            names=z.namelist()
            if len(names)!=len(set(names)) or 'manifest.json' not in names:raise ValueError('duplicate or missing checkpoint entries')
            info=z.getinfo('manifest.json')
            if info.file_size>MAX_MANIFEST_BYTES or info.compress_type!=zipfile.ZIP_STORED:raise ValueError('invalid checkpoint manifest size or compression')
            def unique_object(pairs):
                result={}
                for key,value in pairs:
                    if key in result:raise ValueError('duplicate checkpoint JSON key')
                    result[key]=value
                return result
            manifest=json.loads(z.read('manifest.json'),object_pairs_hook=unique_object,
                                parse_constant=lambda value:(_ for _ in ()).throw(ValueError('nonfinite JSON')))
            if (type(manifest)is not dict or set(manifest)!={'format','version','program','model','optimizer','rng','application'}
                    or manifest['format']!=FORMAT or type(manifest['version'])is not int or manifest['version']!=VERSION
                    or manifest['program']!=_program(runtime)):
                raise ValueError('checkpoint format or exact native program identity mismatch')
            app=_application(manifest['application'])
            def tensor(role,entry,shape,dtype):
                size=math.prod(shape)*(1 if dtype=='u8' else 4);path=_tensor_path(role)
                if (type(entry)is not dict or set(entry)!={'path','shape','dtype','sha256'} or entry['path']!=path
                        or entry['shape']!=list(shape) or any(type(v)is not int for v in entry['shape'])
                        or entry['dtype']!=dtype or path in used):raise ValueError('invalid checkpoint tensor metadata')
                used.add(path);info=z.getinfo(path)
                if info.file_size!=size or info.compress_type!=zipfile.ZIP_STORED:raise ValueError('invalid checkpoint tensor byte count or compression')
                raw=z.read(path)
                if len(raw)!=size or _hash(raw)!=entry['sha256']:raise ValueError('checkpoint tensor integrity mismatch')
                if dtype=='u8':return torch.tensor(list(raw),dtype=torch.uint8)
                if sys.byteorder!='little':
                    values=array.array('I');values.frombytes(raw);values.byteswap();raw=values.tobytes()
                return torch.frombuffer(bytearray(raw),dtype=torch.float32).reshape(shape).clone()
            if type(manifest['model'])is not dict or set(manifest['model'])!=set(expected):raise ValueError('checkpoint model state inventory mismatch')
            weights={n:tensor('model/'+n,manifest['model'][n],spec['shape'],'f32') for n,spec in expected.items()}
            opt=manifest['optimizer']
            if type(opt)is not dict or set(opt)!={'schema','options','step','moments'} or type(opt['schema'])is not int or type(opt['moments'])is not dict:
                raise ValueError('invalid checkpoint optimizer metadata')
            state={**opt,'moments':{}}
            for n,pair in opt['moments'].items():
                if n not in expected or type(pair)is not dict:raise ValueError('checkpoint optimizer state inventory mismatch')
                state['moments'][n]={k:tensor('optimizer/'+n+'/'+k,entry,expected[n]['shape'],'f32') for k,entry in pair.items()}
            runtime._validate_optimizer_state_dict(state,allow_nonfinite=True)
            rng=manifest['rng'];host_rng=None
            if rng is not None:
                if (type(rng)is not dict or set(rng)!={'python','python_runtime','torch_runtime','torch'}
                        or rng['python_runtime']!=list(sys.version_info[:2]) or rng['torch_runtime']!=torch.__version__.split('+')[0]):
                    raise ValueError('checkpoint CPU RNG runtime compatibility mismatch')
                py=_python_rng(rng['python'])
                tr=tensor('rng/torch',rng['torch'],[torch.get_rng_state().numel()],'u8')
                torch.Generator(device='cpu').set_state(tr)
                host_rng=(py,tr)
            elif restore_rng:raise ValueError('checkpoint has no CPU RNG state; request restore_rng=False explicitly')
            if used!=set(names):raise ValueError('checkpoint has unaccounted or missing entries')
        return {'weights':weights,'optimizer':state,'application':app,'rng':host_rng if restore_rng else None}
    except (KeyError,TypeError,zipfile.BadZipFile,RuntimeError,UnicodeDecodeError) as exc:
        raise ValueError('invalid owned-state checkpoint: '+str(exc)) from exc


@dataclass
class StagedCheckpoint:
    weights:dict
    moments:dict
    step:int
    allocations:list
    guards:list
    application:dict
    rng:object


def stage_checkpoint(runtime,path,*,device=None,restore_rng=True):
    _boundary(runtime)
    state=read_checkpoint(runtime,path,restore_rng=restore_rng)
    torch=runtime.torch
    def cuda_device(value):
        result=torch.device('cuda',value) if type(value)is int else torch.device(value)
        if result.type!='cuda':raise ValueError('native checkpoint restoration requires a CUDA device')
        return torch.device('cuda',torch.cuda.current_device()) if result.index is None else result
    if runtime._device is None:
        choices={t.device for t in runtime.weights.values() if getattr(t,'is_cuda',False)}
        chosen=device if device is not None else next(iter(choices)) if len(choices)==1 else torch.device('cuda',0)
        runtime._initialize(cuda_device(chosen))
    elif device is not None and cuda_device(device)!=torch.device(runtime._device):
        raise ValueError('checkpoint device differs from initialized runtime')
    # Reuse the runtime's guarded allocator on a separate ownership ledger.
    from .runtime import Runtime
    staged=SimpleNamespace(torch=torch,_device=runtime._device,_driver=runtime._driver,_allocations=[],_guards=[])
    def allocate(value):
        tensor=Runtime._allocate(staged,list(value.shape));tensor.copy_(value);return tensor
    with torch.cuda.device(runtime._device),torch.cuda.stream(runtime._stream),runtime._context_scope():
        weights={n:allocate(v) for n,v in state['weights'].items()}
        moments={n:{k:allocate(v) for k,v in pair.items()} for n,pair in state['optimizer']['moments'].items()}
        runtime._stream.synchronize()
    return StagedCheckpoint(weights,moments,state['optimizer']['step'],staged._allocations,staged._guards,state['application'],state['rng'])


def commit_checkpoint(runtime,staged):
    """Commit only completed staging; recover prior CPU RNG on commit failure."""
    # Superseded checkpoint allocations and optimizer buffers are no longer
    # executable workspace. External Tensor/Parameter references retain their
    # own storage; the runtime must not keep stale bindings alive indefinitely.
    previous={v.data_ptr() for pair in runtime._optimizer_state.values() for v in pair.values()}
    checkpoint_ids={id(a) for a in getattr(runtime,'_checkpoint_backings',[])}
    retained=[(a,g) for a,g in zip(runtime._allocations,runtime._guards)
              if id(a) not in checkpoint_ids and a.data_ptr()+16 not in previous]
    allocations=[a for a,_ in retained]+staged.allocations
    guards=[g for _,g in retained]+staged.guards
    if staged.rng is not None:
        previous=(random.getstate(),runtime.torch.get_rng_state())
        try:
            runtime.torch.set_rng_state(staged.rng[1]);random.setstate(staged.rng[0])
        except Exception:
            runtime.torch.set_rng_state(previous[1]);random.setstate(previous[0]);raise
    runtime.weights=staged.weights
    runtime._optimizer_state=staged.moments;runtime._optimizer_step=staged.step
    runtime._allocations=allocations;runtime._guards=guards;runtime._checkpoint_backings=list(staged.allocations)
    runtime._last=None;runtime._saved_versions={};runtime._gradients=None;runtime._tape=None;runtime._backward_ready=False
    runtime._packed_ready.clear();runtime._forward_epoch+=1
    return {'step':staged.step,'application':staged.application,'rng_restored':staged.rng is not None,
            'binding_policy':'restored runtime owns new model/state tensors; old external bindings and outputs are stale'}

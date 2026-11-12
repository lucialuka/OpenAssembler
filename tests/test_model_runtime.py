"""Native artifact ownership/identity gates and reusable runtime state guards."""
from contextlib import nullcontext
import ctypes as C
import hashlib
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest

from openassembler.modelfront.kernels import build_bundle, canonical, prepare
from openassembler.modelfront.runtime import Runtime, load_weights, validate_bundle


def plan():
    return {"schema":1,"inputs":[{"name":"x","shape":[3],"dtype":"f32"}],"parameters":[],
            "buffers":{"x":{"shape":[3],"dtype":"f32"},"y":{"shape":[3],"dtype":"f32"},
                       "seed":{"shape":[3],"dtype":"f32"},"dx":{"shape":[3],"dtype":"f32"}},
            "steps":[{"kernel":"relu","inputs":["x"],"output":"y","attrs":{}}],"outputs":["y"],
            "backward":{"steps":[{"kernel":"relu_backward","inputs":["seed","x"],"output":"dx","attrs":{}}],
                        "seed_grads":{"y":"seed"},"grad_inputs":{"x":"dx"},"grad_parameters":{}}}


@pytest.fixture(scope="module")
def compiled_bundle(tmp_path_factory):
    from openassembler import oracle
    if not oracle.available("ptxas") or not oracle.available("nvdisasm"):
        pytest.skip("native bundle integrity fixture needs CUDA toolkit")
    root=tmp_path_factory.mktemp("model-native-bundle")
    build_bundle(plan(), {}, root)
    return root


@pytest.mark.oracle
def test_bundle_validates_real_oa_text_without_diagnostics(compiled_bundle,tmp_path):
    shutil.copytree(compiled_bundle,tmp_path,dirs_exist_ok=True)
    shutil.rmtree(tmp_path/"sources")
    shutil.rmtree(tmp_path/"native"/".build")
    shutil.rmtree(tmp_path/"native"/"sass")
    p,ledger,raw=validate_bundle(tmp_path)
    assert p==plan()
    assert len(ledger["kernels"])==6
    assert raw.startswith(b"\x7fELF")


@pytest.mark.oracle
def test_omitting_optional_listings_preserves_native_cubin_bytes(compiled_bundle,tmp_path):
    build_bundle(plan(), {}, tmp_path, sass_listings=False)
    _, ledger, raw = validate_bundle(tmp_path)
    _, original, before = validate_bundle(compiled_bundle)
    assert raw == before
    assert ledger['kernels'] == original['kernels']
    assert not list(tmp_path.rglob('*.sass'))


@pytest.mark.oracle
@pytest.mark.parametrize("mutation",["plan","binary","reference","owner","text","extra","launch"])
def test_rejects_tampered_or_non_native_artifacts(compiled_bundle,tmp_path,mutation):
    shutil.copytree(compiled_bundle,tmp_path,dirs_exist_ok=True)
    ledger=json.loads((tmp_path/"native-ledger.json").read_text())
    record=next(iter(ledger["kernels"].values()))
    if mutation=="plan":
        p=json.loads((tmp_path/"plan.json").read_text());p["steps"][0]["kernel"]="copy"
        (tmp_path/"plan.json").write_bytes(canonical(p))
    elif mutation=="binary":
        binary=tmp_path/ledger["cubin"];raw=bytearray(binary.read_bytes());raw[-1]^=1;binary.write_bytes(raw)
    elif mutation=="reference": ledger["cubin"]="native/.build/reference/native.cubin"
    elif mutation=="owner": record["compiled_by"]="ptxas-fallback"
    elif mutation=="text": record["text_sha256"]="0"*64
    elif mutation=="extra": ledger["kernels"]["untracked_vendor_kernel"]=record.copy()
    elif mutation=="launch": record["launch"]["threads"]=256
    (tmp_path/"native-ledger.json").write_bytes(canonical(ledger))
    with pytest.raises(ValueError): validate_bundle(tmp_path)


class Tensor:
    sequence=0x1000
    def __init__(self,shape,device="cuda:0",dtype="torch.float32",contiguous=True):
        self.shape=tuple(shape);self.device=device;self.dtype=dtype;self.is_cuda=device.startswith("cuda")
        self.contiguous=contiguous;self.pointer=Tensor.sequence;Tensor.sequence+=0x1000;self._version=0
    def data_ptr(self):return self.pointer
    def is_contiguous(self):return self.contiguous
    def narrow(self,axis,start,length):
        out=Tensor([length],self.device,self.dtype);out.pointer=self.pointer+4*start;return out
    def view(self,shape):
        out=Tensor(shape,self.device,self.dtype);out.pointer=self.pointer;return out


class Torch:
    float32="torch.float32"
    def __init__(self):
        self.allocations=[];self.stream=SimpleNamespace(cuda_stream=123,synchronize=lambda:None)
        self.cuda=SimpleNamespace(device=lambda device:nullcontext(),current_stream=lambda device:self.stream,
            get_device_properties=lambda device:SimpleNamespace(major=8,minor=0,name="NVIDIA A100-SXM4-40GB"))
        self.autograd=SimpleNamespace(graph=SimpleNamespace(increment_version=lambda tensor:setattr(tensor,"_version",tensor._version+1)))
    def empty(self,shape,*,dtype,device):
        out=Tensor(shape,device,dtype);self.allocations.append(out);return out


def make_runtime(monkeypatch,p=None):
    from openassembler.modelfront import runtime as module
    p=plan() if p is None else p
    _,steps,kernels=prepare(p)
    ledger={"cubin_sha256":"verified","kernels":{name:{"registers":16,"text_sha256":"native-text"} for name in kernels}}
    monkeypatch.setattr(module,"validate_bundle",lambda root,plan=None:(p,ledger,b"verified-cubin"))
    counts={"modules":0,"functions":0,"launches":0,"unloads":0}
    class Driver:
        def __init__(self):self.cu=SimpleNamespace(cuLaunchKernel=self.launch)
        def load(self,raw):counts["modules"]+=1;return 1
        def function(self,mod,name):counts["functions"]+=1;return C.c_void_p(2)
        def func_attrs(self,fn):return {"binary_version":80,"num_regs":16}
        def ck(self,result,what):assert result==0
        def memset(self,*args):pass
        def launch(self,*args):counts["launches"]+=1;assert args[-3].value==123;return 0
        def unload(self,module):counts["unloads"]+=1
    monkeypatch.setattr("openassembler.gpu.cuda_driver.Driver",Driver)
    torch=Torch();return Runtime(p,{},"/unused",torch=torch),torch,counts


def test_runtime_reuses_buffers_modules_functions_and_refreshes_inputs(monkeypatch):
    runtime,torch,counts=make_runtime(monkeypatch)
    x=Tensor([3]);one=runtime.forward({"x":x});allocations=len(torch.allocations)
    two=runtime.forward({"x":Tensor([3])})
    assert one["y"] is two["y"]
    assert runtime._last["x"] is not x
    assert "x" not in runtime.buffers
    assert len(torch.allocations)==allocations
    assert counts["modules"]==1 and counts["functions"]==len(runtime.kernels)
    assert counts["launches"]==2 and len(runtime.launch_ledger)==1
    runtime.close();assert counts["unloads"]==1
    with pytest.raises(RuntimeError,match="closed"):runtime.forward({"x":x})


def test_backward_explicit_seed_and_tape_mutation_guards(monkeypatch):
    runtime,torch,counts=make_runtime(monkeypatch)
    x=Tensor([3]);runtime.forward({"x":x},require_tape=True)
    with pytest.raises(RuntimeError,match="outstanding"):runtime.forward({"x":x})
    with pytest.raises(ValueError,match="explicit"):runtime.backward()
    x._version+=1
    with pytest.raises(RuntimeError,match="modified"):runtime.backward({"y":Tensor([3])})
    runtime.discard_tape();runtime.forward({"x":Tensor([3])},require_tape=True)
    gradient=runtime.backward({"y":Tensor([3])})
    assert gradient["inputs"]["x"] is runtime.buffers["dx"]
    with pytest.raises(RuntimeError,match="completed forward"):runtime.backward({"y":Tensor([3])})
    runtime.close()


def test_backward_rejects_user_mutation_of_saved_output(monkeypatch):
    runtime, _, _ = make_runtime(monkeypatch)
    output = runtime.forward({'x': Tensor([3])}, require_tape=True)['y']
    output._version += 1
    with pytest.raises(RuntimeError, match='modified before backward'):
        runtime.backward({'y': Tensor([3])})
    runtime.close()


def test_runtime_uses_guarded_liveness_slots_without_reallocating(monkeypatch):
    from openassembler.modelfront.optimization import optimize_plan
    p = plan()
    p['buffers'].update({n: {'shape': [3], 'dtype': 'f32'} for n in ['a', 'b']})
    p['steps'] = [{'kernel': 'relu', 'inputs': [source], 'output': output}
                  for source, output in [('x', 'a'), ('a', 'b'), ('b', 'y')]]
    p = optimize_plan(p, reuse_storage=True)
    runtime, torch, _ = make_runtime(monkeypatch, p)
    one = runtime.forward({'x': Tensor([3])}, require_tape=True)
    assert runtime.buffers['a'].data_ptr() == one['y'].data_ptr()
    assert runtime.buffers['b'].data_ptr() == runtime.buffers['seed'].data_ptr()
    assert runtime.buffers['a'].data_ptr() != runtime.buffers['b'].data_ptr()
    assert len(runtime._storage_slots) == p['storage']['physical_slots'] == 3
    allocated = len(torch.allocations)
    gradients = runtime.backward({'y': Tensor([3])})
    assert gradients['inputs']['x'].data_ptr() != one['y'].data_ptr()
    runtime.forward({'x': Tensor([3])})
    assert len(torch.allocations) == allocated
    runtime.close()


@pytest.mark.parametrize("value",[Tensor([4]),Tensor([3],device="cpu"),Tensor([3],dtype="torch.float16"),Tensor([3],contiguous=False)])
def test_shape_dtype_layout_guards_before_loading(monkeypatch,value):
    runtime,torch,counts=make_runtime(monkeypatch)
    with pytest.raises(ValueError,match="contiguous CUDA FP32"):runtime.forward({"x":value})
    assert counts["modules"]==0


def test_stream_switch_rejects_workspace_races(monkeypatch):
    runtime,torch,counts=make_runtime(monkeypatch)
    runtime.forward({"x":Tensor([3])});torch.stream.cuda_stream=456
    with pytest.raises(ValueError,match="initialization CUDA stream"):runtime.forward({"x":Tensor([3])})
    assert counts["launches"]==1
    runtime.close()


def test_weight_integrity_is_checked_before_torch_allocation(tmp_path):
    p={"parameters":[{"name":"weight","shape":[2],"dtype":"f32"}]}
    (tmp_path/"weight.bin").write_bytes(b"\0"*8)
    metadata={"weight":{"path":"weight.bin","shape":[2],"dtype":"f32","sha256":"bad"}}
    (tmp_path/"weights.json").write_bytes(canonical(metadata))
    with pytest.raises(ValueError,match="integrity"):load_weights(tmp_path,p,object())
    metadata["weight"]["path"]="../weight.bin"
    (tmp_path/"weights.json").write_bytes(canonical(metadata))
    with pytest.raises(ValueError,match="escaping"):load_weights(tmp_path,p,object())


def test_profiler_audit_requires_exact_names_and_counts():
    from openassembler.modelfront.runtime import audit_profiler_events
    event=lambda name:SimpleNamespace(name=name,device_type="DeviceType.CUDA")
    ledger=[{"entry":"oa_model_a"},{"entry":"oa_model_a"},{"entry":"oa_pack_b"}]
    events=[event("oa_model_a"),event("oa_model_a"),event("oa_pack_b")]
    assert audit_profiler_events(events,ledger)["observed_compute_launches"]==3
    for bad in [[],events[:-2],events+[event("vendor_gemm")],events+[event("unexpected_memcpy_compute")],
                events+[event("unexpected_memset_compute")],events+[event("Memcpy HtoD")]]:
        with pytest.raises(ValueError,match="audit failed"):audit_profiler_events(bad,ledger)


def test_tensor_mode_requires_admitted_full_tile_contracts():
    p=plan();p["math"]="f16_rne_f32_accum"
    p["inputs"]=[{"name":"x","shape":[17,8],"dtype":"f32"},
                 {"name":"w","shape":[16,8],"dtype":"f32"}]
    p["buffers"]={"y":{"shape":[17,16],"dtype":"f32"}}
    p["steps"]=[{"kernel":"linear","inputs":["x","w"],"output":"y","attrs":{}}]
    p.pop("backward")
    with pytest.raises(ValueError):prepare(p)


def test_tensor_mode_recipes_replace_every_linear_forward_and_backward():
    from openassembler.modelfront.kernels import tensor_recipes
    p={"schema":1,"math":"f16_rne_f32_accum",
       "inputs":[{"name":"x","shape":[128,128],"dtype":"f32"}],
       "parameters":[{"name":"w","shape":[128,128],"dtype":"f32"}],
       "buffers":{name:{"shape":[128,128],"dtype":"f32"} for name in ["y","seed","dx","dw"]},
       "steps":[{"kernel":"linear","inputs":["x","w"],"output":"y","attrs":{"m":128,"n":128,"k":128}}],
       "outputs":["y"],"backward":{"seed_grads":{"y":"seed"},"grad_inputs":{"x":"dx"},
       "grad_parameters":{"w":"dw"},"steps":[
           {"kernel":"linear_dx","inputs":["seed","w"],"output":"dx","attrs":{}},
           {"kernel":"linear_dw","inputs":["seed","x"],"output":"dw","attrs":{}}]}}
    specs,steps,kernels=prepare(p)
    assert all(s.get("tensor") for s in steps)
    assert len(tensor_recipes(p,specs))==3
    assert not any(k.op in ("linear","linear_dx","linear_dw") for k in kernels.values())

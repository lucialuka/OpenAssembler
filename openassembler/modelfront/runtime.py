"""Guarded reusable launch plans for OA-emitted SM80 model kernels.

Torch is used only for storage, metadata and CUDA stream access. All graph
arithmetic, derivatives, loss and updates launch audited OA cubin functions.
"""
from __future__ import annotations

import ctypes as C
import hashlib
import importlib
import json
import math
from pathlib import Path
from contextlib import contextmanager

from .kernels import build_bundle, canonical, emit_kernel, prepare, recipe_record, shape_of, tensor_recipes


def _file(root, relative):
    if not isinstance(relative, str) or Path(relative).is_absolute():
        raise ValueError("artifact paths must be relative to the model bundle")
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError(f"missing or escaping model artifact: {relative}")
    return path


def validate_bundle(root, plan=None):
    """Validate bytes, every executable function, launch contracts and plan identity."""
    from openassembler.container.elf import Elf, EM_CUDA
    root = Path(root)
    if plan is None:
        plan = json.loads(_file(root, "plan.json").read_text())
    ledger = json.loads(_file(root, "native-ledger.json").read_text())
    if ledger.get("schema") != 1 or ledger.get("backend") != "openassembler-native" or ledger.get("arch") != "sm_80":
        raise ValueError("model bundle must use the OA native SM80 backend")
    if ledger.get("plan_sha256") != hashlib.sha256(canonical(plan)).hexdigest():
        raise ValueError("model plan identity differs from its native kernel ledger")
    specs, steps, kernels = prepare(plan)
    if ledger.get("steps") != steps or set(ledger.get("kernels", {})) != set(kernels):
        raise ValueError("native kernel inventory does not cover the complete model plan")
    # Deployment archives may omit diagnostics/PTX. Regenerate source contracts
    # from the plan so a changed source identity is rejected without needing it.
    source = ".version 8.7\n.target sm_80\n.address_size 64\n" + "\n".join(kernels[name].ptx for name in sorted(kernels))
    if hashlib.sha256(source.encode()).hexdigest() != ledger.get("source_sha256"):
        raise ValueError("native kernel source identity does not match this compiler")
    relative = ledger.get("cubin")
    if not isinstance(relative, str) or not relative.startswith("native/cubin/") or not relative.endswith(".cubin"):
        raise ValueError("native runtime cannot load references, PTX or arbitrary modules")
    raw = _file(root, relative).read_bytes()
    if hashlib.sha256(raw).hexdigest() != ledger.get("cubin_sha256"):
        raise ValueError("native cubin integrity check failed")
    try:
        elf = Elf.parse(raw)
    except Exception as exc:
        raise ValueError("invalid native CUDA cubin") from exc
    if elf.machine != EM_CUDA or elf.flags & 255 != 80:
        raise ValueError("native model requires an SM80 CUDA cubin")
    sections = {s.name[6:]: s.data for s in elf.sections if s.name.startswith(".text.") and s.data}
    if set(sections) != set(kernels):
        raise ValueError("native cubin has missing or unaccounted executable functions")
    for name, kernel in kernels.items():
        record = ledger["kernels"][name]
        if (record.get("compiled_by") != "openassembler" or record.get("signature") != kernel.signature
                or record.get("launch") != kernel.launch
                or record.get("text_sha256") != hashlib.sha256(sections[name]).hexdigest()
                or type(record.get("registers")) is not int or not 1 <= record["registers"] <= 255):
            raise ValueError(f"native kernel ownership/identity mismatch: {name}")
    recipes = tensor_recipes(plan,specs)
    expected_recipes = {name:recipe_record(r) for name,r in recipes.items()}
    if ledger.get("tensor_recipes", {}) != expected_recipes:
        raise ValueError("native tensor recipe inventory does not match this model plan")
    stages = {s.name:s for r in recipes.values() for s in r.stages}
    if set(ledger.get("tensor_kernels",{})) != set(stages):
        raise ValueError("native tensor modules have missing or unknown functions")
    for name, stage in stages.items():
        record = ledger["tensor_kernels"][name]
        path = record.get("module")
        if (not isinstance(path,str) or not path.startswith("native_tensor/") or not path.endswith("/native.cubin")
                or record.get("compiled_by") != "openassembler" or record.get("entry") != name
                or record.get("signature") != stage.signature or record.get("launch") != stage.launch
                or record.get("source_sha256") != hashlib.sha256(stage.ptx.encode()).hexdigest()
                or type(record.get("registers")) is not int or not 1 <= record["registers"] <= 255):
            raise ValueError(f"native tensor function ownership/contract mismatch: {name}")
        binary = _file(root,path).read_bytes()
        if hashlib.sha256(binary).hexdigest() != record.get("cubin_sha256"):
            raise ValueError("native tensor cubin integrity check failed")
        elf = Elf.parse(binary)
        texts = {s.name[6:]:s.data for s in elf.sections if s.name.startswith(".text.") and s.data}
        if (elf.machine != EM_CUDA or elf.flags & 255 != 80 or set(texts) != {name}
                or hashlib.sha256(texts[name]).hexdigest() != record.get("text_sha256")):
            raise ValueError("native tensor cubin has wrong architecture or unaccounted text")
    return plan, ledger, raw


def save_weights(weights, plan, root):
    """Store only raw little-endian FP32 bytes, never executable pickle payloads."""
    import array
    import sys
    root = Path(root)
    metadata = {}
    for spec in plan.get("parameters", []):
        name = spec["name"]
        tensor = weights.get(name, weights.get(spec.get("target")))
        if tensor is None or tuple(tensor.shape) != shape_of(spec) or str(tensor.dtype) != "torch.float32":
            raise ValueError(f"missing or incompatible native model weight: {name}")
        values = array.array("f", tensor.detach().cpu().contiguous().reshape(-1).tolist())
        if sys.byteorder != "little":
            values.byteswap()
        raw = values.tobytes()
        relative = "weights/" + hashlib.sha256(name.encode()).hexdigest() + ".bin"
        path = root / relative; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(raw)
        metadata[name] = {"path": relative, "shape": list(spec["shape"]), "dtype": "f32",
                          "sha256": hashlib.sha256(raw).hexdigest()}
    (root / "weights.json").write_bytes(canonical(metadata))
    return metadata


def load_weights(root, plan, torch, *, device="cuda"):
    root = Path(root)
    metadata = json.loads(_file(root, "weights.json").read_text())
    if isinstance(metadata, list):
        metadata = {x["name"]: x for x in metadata}
    expected = {s["name"]: s for s in plan.get("parameters", [])}
    if set(metadata) != set(expected):
        raise ValueError("weight inventory does not match model parameters")
    weights = {}
    # Validate the complete inventory before allocating any GPU parameter.
    checked = {}
    for name, spec in expected.items():
        entry = metadata[name]
        if shape_of(entry) != shape_of(spec):
            raise ValueError(f"weight metadata mismatch: {name}")
        raw = _file(root, entry.get("path")).read_bytes()
        if len(raw) != 4*math.prod(shape_of(spec)) or hashlib.sha256(raw).hexdigest() != entry.get("sha256"):
            raise ValueError(f"weight integrity check failed: {name}")
        checked[name] = raw
    for name, raw in checked.items():
        cpu = torch.frombuffer(bytearray(raw), dtype=torch.float32).reshape(expected[name]["shape"])
        weights[name] = cpu.to(device=device)
    return weights


def audit_profiler_events(events, launch_ledger):
    """Compare independently observed compute events with actual launch entries.

    This conservative event-list interface treats every CUDA activity as
    compute. For traces containing DMA or GPU annotations, use
    profiling.audit_profile, which classifies activity categories explicitly.
    """
    from collections import Counter
    observed=[]
    for event in events:
        if str(getattr(event,"device_type","")).endswith("CUDA"):
            name=event.name
            observed.append(name)
    expected=Counter(record["entry"] for record in launch_ledger)
    counts=Counter(observed)
    unknown=sorted(set(counts)-set(expected))
    mismatches={name:{"expected":count,"observed":counts.get(name,0)} for name,count in expected.items() if counts.get(name,0)!=count}
    if not observed or unknown or mismatches:
        raise ValueError(f"native GPU launch audit failed: unknown={unknown}, count_mismatches={mismatches}, observed={len(observed)}")
    return {"passed":True,"observed_compute_launches":len(observed),"kernel_counts":dict(counts),
            "unknown_compute_kernels":[],"scope":"independent CUDA profiler kernel activities matched by name/count to driver launch ledger"}


class Runtime:
    def __init__(self, plan, weights, build_dir, *, torch=None):
        self.root = Path(build_dir)
        # Freeze the caller's mutable plan before deriving execution guards.
        frozen_plan = json.loads(canonical(plan))
        self.plan, self.ledger, self._cubin = validate_bundle(self.root, frozen_plan)
        self.specs, self.steps, self.kernels = prepare(self.plan)
        self.recipes = tensor_recipes(self.plan,self.specs)
        self._kernel_records = {**self.ledger["kernels"], **self.ledger.get("tensor_kernels",{})}
        self.torch = torch or importlib.import_module("torch")
        self.weights = dict(weights or {})
        self.buffers = {}
        self._module = None
        self._functions = {}
        self._driver = None
        self._device = None
        self._last = None
        self._tape = None
        self._gradients = None
        self._training = {}
        self._optimizer_state = {}
        self._optimizer_step = 0
        self._checkpoint_backings = []
        self._forward_epoch = 0
        self.launches = []
        self._closed = False
        self._backward_ready = False
        self._stream_id = None
        self._stream = None
        self._allocations = []
        self._guards = []
        self._tensor_modules = []
        self._tensor_workspaces = {}
        self._context = None
        self._initialization_failed = False
        self._saved_versions = {}
        self._storage_slots = []
        self._tensor_bindings = {}
        self._packed_ready = set()
        self._tensor_schedule = None
        if self.recipes:
            from .tensor_schedule import workspace_plan
            self._tensor_schedule = workspace_plan(self.plan,self.steps,self.recipes)

    @classmethod
    def load(cls, output_root, weights=None, *, torch=None):
        root = Path(output_root)
        plan, _, _ = validate_bundle(root)
        torch = torch or importlib.import_module("torch")
        if weights is None:
            weights = load_weights(root, plan, torch) if plan.get("parameters") else {}
        return cls(plan, weights, root, torch=torch)

    def _check(self, tensor, spec, name):
        dtype = spec.get("dtype", "f32")
        expected = {"f32":"torch.float32", "i64":"torch.int64", "bool":"torch.bool"}.get(dtype)
        if (not getattr(tensor, "is_cuda", False) or str(tensor.dtype) != expected
                or tuple(tensor.shape) != shape_of(spec, dtypes=("f32","i64","bool")) or not tensor.is_contiguous()
                or tensor.data_ptr() % {"f32":4,"i64":8,"bool":1}.get(dtype,4)):
            raise ValueError(f"{name} needs a contiguous CUDA {'FP32' if dtype == 'f32' else dtype} tensor of shape {spec['shape']}")
        device = tensor.device
        if self._device is not None and device != self._device:
            raise ValueError("all tensors in one native plan must share the same CUDA device")
        return device

    def _initialize(self, device):
        if self._closed:
            raise RuntimeError("native model runtime is closed")
        if self._initialization_failed:
            raise RuntimeError("native runtime initialization previously failed; create a new runtime")
        if self._module is not None:
            return
        torch = self.torch
        props = torch.cuda.get_device_properties(device)
        if (props.major, props.minor) != (8, 0) or "A100" not in props.name:
            raise ValueError("native model execution requires an A100 (SM80)")
        self._device = device
        self._initialization_failed = True
        from openassembler.gpu.cuda_driver import Driver
        # Reuse the application's CUDA context; never create a separate one.
        with torch.cuda.device(device):
            torch.cuda.current_stream(device)
            self._stream = torch.cuda.current_stream(device)
            self._stream_id = self._stream.cuda_stream
            self._driver = Driver()
            if hasattr(self._driver.cu,"cuCtxGetCurrent"):
                self._driver.cu.cuCtxGetCurrent.argtypes=[C.POINTER(C.c_void_p)]
                self._driver.cu.cuCtxSetCurrent.argtypes=[C.c_void_p]
                self._context=C.c_void_p()
                self._driver.ck(self._driver.cu.cuCtxGetCurrent(C.byref(self._context)),"native graph current context")
                if not self._context.value:
                    raise RuntimeError("native model requires the application's active CUDA context")
            self._module = self._driver.load(self._cubin)
            for name, kernel in self.kernels.items():
                fn = self._driver.function(self._module, name)
                actual = self._driver.func_attrs(fn)
                if actual.get("binary_version") != 80 or actual.get("num_regs") != self.ledger["kernels"][name]["registers"]:
                    raise ValueError(f"loaded native function resource mismatch: {name}")
                self._functions[name] = fn
            for name,record in self.ledger.get("tensor_kernels",{}).items():
                binary = _file(self.root,record["module"]).read_bytes()
                # validate_bundle already checked the same file; guard mutation
                # between construction and lazy loading too.
                if hashlib.sha256(binary).hexdigest() != record["cubin_sha256"]:
                    raise ValueError("native tensor module changed before loading")
                module = self._driver.load(binary)
                self._tensor_modules.append(module)
                fn = self._driver.function(module,name)
                attrs = self._driver.func_attrs(fn)
                if attrs.get("binary_version") != 80 or attrs.get("num_regs") != record["registers"]:
                    raise ValueError("loaded native tensor resources differ from their ledger")
                if record["launch"]["smem"] > 49152:
                    self._driver.ck(self._driver.cu.cuFuncSetAttribute(fn,8,record["launch"]["smem"]),"native tensor shared memory opt-in")
                self._functions[name] = fn
            external = {s["name"] for s in self.plan.get("inputs", []) + self.plan.get("parameters", [])}
            layout = self.plan.get('storage')
            if layout is None:
                self.buffers = {name: self._allocate(spec['shape'], dtype=spec.get("dtype","f32"))
                                for name, spec in self.plan.get('buffers', {}).items()
                                if name not in external and name not in self.plan.get('aliases', {})}
            else:
                self._storage_slots = [self._allocate([slot['elements']]) for slot in layout['slots']]
                self.buffers = {name: self._storage_slots[index].view(self.specs[name]['shape'])
                                for name, index in layout['assignments'].items()}
            for output in self.plan.get("outputs", []):
                spec = self.specs[output]
                self._training[output] = {"loss": self._allocate([]), "seed": self._allocate(spec["shape"])}
                if self.plan.get("loss") == "cross_entropy":
                    self._training[output]["rows"] = self._allocate([spec["shape"][0]])
            if self._tensor_schedule is not None:
                for name,spec in self._tensor_schedule['allocations'].items():
                    self._tensor_workspaces[name] = self._allocate(spec['shape'],dtype=spec['dtype'])
                for output,bindings in self._tensor_schedule['bindings'].items():
                    self._tensor_bindings[output] = {
                        name:self._tensor_workspaces[spec['allocation']].view(-1).narrow(
                            0,spec['offset'],math.prod(spec['shape'])).view(spec['shape'])
                        for name,spec in bindings.items()}
            else:
                for recipe in self.recipes.values():
                    for name,spec in recipe.workspaces.items():
                        self._tensor_workspaces[name] = self._allocate(spec["shape"],dtype="f16")
        self._initialization_failed = False

    def _allocate(self, shape, *, dtype="f32"):
        count = math.prod(shape)
        torch_dtype = getattr(self.torch, {"f32":"float32", "f16":"float16", "i64":"int64", "bool":"bool"}[dtype])
        itemsize = {"f32":4,"f16":2,"i64":8,"bool":1}[dtype]
        pad = 16 // itemsize
        backing = self.torch.empty([count+2*pad], dtype=torch_dtype,
                                   device=self._device)
        tensor = backing.narrow(0, pad, count).view(shape)
        self._allocations.append(backing)
        start, end = backing.data_ptr(), backing.data_ptr()+16+count*itemsize
        self._driver.memset_async(start, 0xA5, 16, self._stream_id)
        self._driver.memset_async(end, 0xA5, 16, self._stream_id)
        self._guards.append((start,end))
        return tensor

    def poison_workspace(self):
        """Poison every intermediate with NaN bits before independent replay proof."""
        if self._module is None:
            raise RuntimeError("initialize native runtime with forward before poisoning workspace")
        self.synchronize()
        with self._context_scope():
            for tensor in [*self.buffers.values(), *(v for x in self._training.values() for v in x.values())]:
                self._driver.memset_async(tensor.data_ptr(), 0xFF, tensor.numel()*tensor.element_size(), self._stream_id)
            for tensor in self._tensor_workspaces.values():
                self._driver.memset_async(tensor.data_ptr(),0xFF,tensor.numel()*tensor.element_size(),self._stream_id)

    def check_canaries(self):
        """Synchronize and verify both ends of every private workspace allocation."""
        self.synchronize()
        with self._context_scope():
            bad = [i for i,(start,end) in enumerate(self._guards)
                   if self._driver.dtoh(start,16) != b"\xa5"*16 or self._driver.dtoh(end,16) != b"\xa5"*16]
        if bad:
            raise RuntimeError(f"native model workspace canary corruption: {bad}")
        return {"passed":True,"allocations":len(self._guards)}

    def _launch(self, kernel, inputs, output, *, lr=None, correction=None):
        args = [C.c_uint64(t.data_ptr()) for t in [*inputs, output]]
        if kernel.op in ("sgd", "sgd_update"):
            rate = kernel.attrs.get("lr", 0.01) if lr is None else lr
            if isinstance(rate, bool) or not isinstance(rate, (int, float)) or not math.isfinite(rate) or rate < 0:
                raise ValueError("learning rate must be finite and nonnegative")
            from .kernels import _float
            _float(rate)
            args.append(C.c_float(rate))
        if kernel.op in ("adam_update", "adam_fused"):
            args.extend(C.c_float(x) for x in (lr, *correction))
        params = (C.c_void_p*len(args))(*[C.cast(C.pointer(a), C.c_void_p) for a in args])
        stream = self.torch.cuda.current_stream(self._device).cuda_stream
        if stream != self._stream_id:
            raise ValueError("native runtime workspace is bound to its initialization CUDA stream")
        # Inputs may have been allocated on another stream. Record their use so
        # PyTorch's storage allocator cannot recycle them during async launches.
        for tensor in [*inputs, output]:
            record_stream = getattr(tensor, "record_stream", None)
            if record_stream is not None:
                record_stream(self._stream)
        launch = kernel.launch
        with self._context_scope():
            self._driver.ck(self._driver.cu.cuLaunchKernel(self._functions[kernel.name], *launch["grid"],
                            launch["threads"], 1, 1, launch["smem"], C.c_void_p(stream), params, None), "native model launch")
        # Record the actual function selected at the launch boundary. The
        # bounded latest-run ledger avoids growing with training iterations.
        self.launches.append({"entry": kernel.name, "compiled_by": "openassembler",
                              "cubin_sha256": self._kernel_records[kernel.name].get("cubin_sha256",self.ledger["cubin_sha256"]),
                              "text_sha256": self._kernel_records[kernel.name]["text_sha256"],
                              "launch": kernel.launch})

    @contextmanager
    def _context_scope(self):
        # Autograd may call backward on another CPU thread; driver current
        # contexts are thread local even when torch's device index is unchanged.
        previous=C.c_void_p()
        if self._context is not None:
            self._driver.ck(self._driver.cu.cuCtxGetCurrent(C.byref(previous)),"save CUDA context")
            self._driver.ck(self._driver.cu.cuCtxSetCurrent(self._context),"bind native graph context")
        try:
            yield
        finally:
            if self._context is not None:
                self._driver.ck(self._driver.cu.cuCtxSetCurrent(previous),"restore CUDA context")

    def _run(self, phase, values):
        for step in self.steps:
            if step["phase"] == phase:
                if step.get('alias'):
                    values[step['output']] = values[step['inputs'][0]].view(self.specs[step['output']]['shape'])
                elif step.get("tensor"):
                    recipe = self.recipes[step["entry"]]
                    bindings = {f"a{i}":values[name] for i,name in enumerate(step["inputs"])}
                    bindings.update(self._tensor_bindings[step['output']] if self._tensor_schedule is not None else self._tensor_workspaces)
                    bindings["output"] = values[step["output"]]
                    pack_slots = self._tensor_schedule['pack_slots'][step['output']] if self._tensor_schedule is not None else {}
                    for stage in recipe.stages:
                        key = pack_slots.get(stage.output)
                        if key is not None:
                            if key in self._packed_ready:
                                continue
                            self._packed_ready.add(key)
                        self._launch(stage,[bindings[x] for x in stage.inputs],bindings[stage.output])
                else:
                    self._launch(self.kernels[step["entry"]], [values[x] for x in step["inputs"]], values[step["output"]])

    def forward(self, inputs, parameters=None, *, require_tape=False):
        if self._tape is not None:
            raise RuntimeError("native runtime has an outstanding backward tape; finish backward or discard_tape first")
        if parameters is not None:
            self.weights = dict(parameters)
        if isinstance(inputs, (list, tuple)):
            if len(inputs) != len(self.plan.get("inputs", [])):
                raise ValueError("native model input count mismatch")
            inputs = dict(zip((s["name"] for s in self.plan["inputs"]), inputs))
        if set(inputs) != {s["name"] for s in self.plan.get("inputs", [])}:
            raise ValueError("native model input names mismatch")
        values = dict(inputs)
        for spec in self.plan.get("parameters", []):
            name = spec["name"]
            if name not in self.weights and spec.get("target") in self.weights:
                self.weights[name] = self.weights[spec["target"]]
            if name not in self.weights:
                raise ValueError(f"missing native model parameter {name}")
            values[name] = self.weights[name]
        devices = {self._check(values[s["name"]], s, s["name"]) for s in self.plan.get("inputs", []) + self.plan.get("parameters", [])}
        if len(devices) != 1:
            raise ValueError("native graph needs tensors on exactly one CUDA device")
        # Index validation runs on CPU before *any* native launch or state mutation.
        # This synchronizing check is visible host overhead, never a GPU fallback.
        for domain in self.plan.get("index_domains", []):
            indices = values[domain["name"]].detach().cpu()
            if bool(((indices < domain["lower"]) | (indices >= domain["upper"])).any()):
                raise ValueError(f"{domain['name']}: embedding index outside [{domain['lower']}, {domain['upper']})")
        self._initialize(next(iter(devices)))
        # The fixed captured graph includes all first uses every iteration.
        # No validity decision survives an SGD update or a changed external input.
        self._packed_ready.clear()
        values.update(self.buffers)
        self.launches = []
        with self.torch.cuda.device(self._device):
            self._run("forward", values)
        self._last = values
        self._forward_epoch += 1
        # Saved outputs/intermediates can alias user-visible model results.
        # Protect them as well as external inputs/weights: an in-place edit of
        # a saved ReLU/softmax result must not silently alter its derivative.
        self._saved_versions = {name: (t.data_ptr(), getattr(t, "_version", 0))
                                for name, t in values.items()}
        self._backward_ready = True
        self._gradients = None
        if require_tape:
            self._tape = dict(self._saved_versions)
        return {name: values[name] for name in self.plan.get("outputs", [])}

    def discard_tape(self):
        self._tape = None
        self._backward_ready = False

    @property
    def launch_ledger(self):
        return self.launches

    def kernel_resources(self):
        """Return loaded driver resources and saved identities without launching work."""
        if self._module is None or self._closed:
            raise RuntimeError('initialize a live native runtime before inspecting kernel resources')
        from openassembler.container.elf import Elf
        result, modules = {}, {}
        with self._context_scope():
            for name, function in self._functions.items():
                record = self._kernel_records[name]
                relative = record.get('module', self.ledger['cubin'])
                if relative not in modules:
                    binary = _file(self.root, relative).read_bytes()
                    sections = {section.name[6:]:len(section.data) for section in Elf.parse(binary).sections
                                if section.name.startswith('.text.')}
                    listing = self.root / (relative.replace('/cubin/', '/sass/'))
                    listing = listing.with_suffix('.sass')
                    modules[relative] = (len(binary), sections, listing.stat().st_size if listing.is_file() else None)
                size, sections, listing_size = modules[relative]
                result[name] = {'signature':record['signature'],'module':relative,
                                'module_cubin_bytes':size,'kernel_text_bytes':sections[name],
                                'module_sass_bytes':listing_size,
                                'driver':self._driver.func_attrs(function),
                                'dynamic_shared_bytes':record['launch']['smem']}
        return result

    def synchronize(self):
        if self._device is not None:
            self._stream.synchronize()

    def backward(self, seeds=None, *, zero_missing=False):
        if self._last is None or not self._backward_ready:
            raise RuntimeError("native backward requires a completed forward")
        for name, identity in self._saved_versions.items():
            tensor = self._last[name]
            if (tensor.data_ptr(), getattr(tensor, "_version", 0)) != identity:
                raise RuntimeError("a saved native graph input/parameter was modified before backward")
        spec = self.plan.get("backward", {})
        if not spec.get("seed_grads"):
            raise ValueError("native model plan has no backward graph")
        if seeds is None:
            seeds = {}
        elif isinstance(seeds, (list, tuple)):
            if len(seeds) != len(spec["seed_grads"]):
                raise ValueError("native gradient seed count mismatch")
            seeds = dict(zip(spec["seed_grads"], seeds))
        if set(seeds) - set(spec["seed_grads"]):
            raise ValueError("unknown native gradient output seed")
        values = dict(self._last)
        with self.torch.cuda.device(self._device):
            for output, seed in spec["seed_grads"].items():
                if output in seeds:
                    self._check(seeds[output], self.specs[seed], seed)
                    values[seed] = seeds[output]
                elif zero_missing or math.prod(shape_of(self.specs[output])) == 1:
                    self._launch(emit_kernel("fill", [], self.specs[seed], {"value": 0.0 if zero_missing else 1.0}), [], values[seed])
                else:
                    raise ValueError("non-scalar native outputs require explicit gradient seeds")
            self._run("backward", values)
        self._tape = None
        self._backward_ready = False
        self._gradients = {"inputs": {name: values[g] for name, g in spec.get("grad_inputs", {}).items()},
                           "parameters": {name: values[g] for name, g in spec.get("grad_parameters", {}).items()}}
        return self._gradients

    def optimizer_state_dict(self):
        """CPU snapshot of all moments and step; no borrowed device storage."""
        from .optimizer import optimizer_options
        self.synchronize()
        return {"schema":1, "options":optimizer_options(self.plan.get("optimizer",{})),
                "step":self._optimizer_step,
                "moments":{n:{k:v.detach().cpu().clone() for k,v in pair.items()}
                           for n,pair in self._optimizer_state.items()}}

    def _validate_optimizer_state_dict(self, state, *, allow_nonfinite=False):
        from .optimizer import optimizer_options,optimizer_state_keys
        options=optimizer_options(self.plan.get("optimizer",{}))
        if (not isinstance(state,dict) or set(state)!={"schema","options","step","moments"} or state["schema"]!=1
                or state["options"]!=options or type(state["step"]) is not int or state["step"]<0):
            raise ValueError("optimizer checkpoint policy or step mismatch")
        keys=optimizer_state_keys(options)
        expected=set(self.plan["backward"]["grad_parameters"]) if state["step"] and keys else set()
        if not isinstance(state["moments"],dict) or set(state["moments"])!=expected:raise ValueError("optimizer checkpoint parameter names mismatch")
        for n,pair in state["moments"].items():
            if not isinstance(pair,dict) or set(pair)!=set(keys):raise ValueError("optimizer checkpoint needs complete policy state")
            for k,v in pair.items():
                if (not isinstance(v,self.torch.Tensor) or v.device.type!="cpu" or v.dtype!=self.torch.float32 or list(v.shape)!=self.specs[n]["shape"]
                        or not allow_nonfinite and not bool(self.torch.isfinite(v).all())
                        or k in ("v","max_v") and bool((v<0).any())):
                    raise ValueError("invalid optimizer checkpoint moment")
            if "max_v" in pair and bool((pair["max_v"]<pair["v"]).any()):
                raise ValueError("AMSGrad maximum must dominate the second moment")

    def load_optimizer_state_dict(self, state):
        if self._device is None or self._tape is not None or self._gradients is not None:
            raise ValueError("initialize forward and discard its tape before loading optimizer state")
        self._validate_optimizer_state_dict(state)
        previous={v.data_ptr() for pair in self._optimizer_state.values() for v in pair.values()}
        # All validation precedes replacement. Copies are host-to-device transfers.
        staged={}
        with self.torch.cuda.device(self._device),self.torch.cuda.stream(self._stream):
            for name,pair in state["moments"].items():
                staged[name]={}
                for key,value in pair.items():
                    tensor=self._allocate(self.specs[name]["shape"])
                    tensor.copy_(value.contiguous())
                    staged[name][key]=tensor
            self._stream.synchronize()
        retained=[(a,g) for a,g in zip(self._allocations,self._guards) if a.data_ptr()+16 not in previous]
        self._allocations=[a for a,_ in retained];self._guards=[g for _,g in retained]
        self._checkpoint_backings=[a for a in getattr(self,'_checkpoint_backings',[]) if a.data_ptr()+16 not in previous]
        self._optimizer_state,self._optimizer_step=staged,state["step"]

    def save_checkpoint(self, path, *, application=None, include_rng=True):
        """Atomically save all owned model/optimizer state and explicit host state."""
        from .checkpoint import save_checkpoint
        return save_checkpoint(self,path,application=application,include_rng=include_rng)

    def load_checkpoint(self, path, *, device=None, restore_rng=True):
        """Restore a complete checkpoint into new owned bindings at a boundary.

        Returned application metadata is data, not executable host orchestration.
        Old external tensor bindings are replaced and must be obtained again
        from ``weights``. Existing exposed outputs are invalidated.
        """
        from .checkpoint import stage_checkpoint,commit_checkpoint
        staged=stage_checkpoint(self,path,device=device,restore_rng=restore_rng)
        return commit_checkpoint(self,staged)

    def update(self, lr=0.01):
        from .optimizer import optimizer_options,optimizer_state_keys
        from .kernels import _float
        options=optimizer_options(self.plan.get("optimizer",{}))
        if type(lr) not in (int,float) or not math.isfinite(lr) or lr<0:raise ValueError("learning rate must be finite and nonnegative")
        _float(lr)
        if self._gradients is None:raise RuntimeError("native optimizer requires backward gradients")
        # Preflight the whole parameter set before updating any member.
        for name in self._gradients["parameters"]:
            w=self.weights[name]
            if (w.data_ptr(),getattr(w,"_version",0))!=self._saved_versions[name]:
                raise RuntimeError("a native parameter was replaced or modified before optimizer update")
        step=self._optimizer_step+1
        with self.torch.cuda.device(self._device):
            for name,gradient in self._gradients["parameters"].items():
                weight=self.weights[name];spec=self.specs[name]
                if options["name"]=="sgd":
                    if options=={"name":"sgd"}:
                        self._launch(emit_kernel("sgd",[spec]*2,spec),[weight,gradient],weight,lr=lr)
                    else:
                        args=[weight,gradient]
                        if options.get("momentum",0):
                            if name not in self._optimizer_state:
                                buffer=self._allocate(spec["shape"])
                                self._launch(emit_kernel("sgd_buffer_init",[spec]*2,spec,options),[gradient,weight],buffer)
                                self._optimizer_state[name]={"momentum":buffer}
                            else:
                                buffer=self._optimizer_state[name]["momentum"]
                                self._launch(emit_kernel("sgd_buffer",[spec]*3,spec,options),[buffer,gradient,weight],buffer)
                            args.append(buffer)
                        self._launch(emit_kernel("sgd_update",[spec]*len(args),spec,options),args,weight,lr=lr)
                else:
                    if name not in self._optimizer_state:
                        pair={k:self._allocate(spec["shape"]) for k in optimizer_state_keys(options)}
                        with self._context_scope():
                            for value in pair.values():self._driver.memset_async(value.data_ptr(),0,value.numel()*4,self._stream_id)
                        self._optimizer_state[name]=pair
                    pair=self._optimizer_state[name]
                    correction=(lr/(1-options["beta1"]**step),math.sqrt(1-options["beta2"]**step))
                    if self.plan.get("optimizer_fusion", False):
                        self._launch(emit_kernel("adam_fused",[spec]*4,spec,options),
                                     [weight,pair["m"],pair["v"],gradient],weight,lr=lr,correction=correction)
                    else:
                        for suffix in ("m","v"):
                            self._launch(emit_kernel("adam_"+suffix,[spec]*3,spec,options),
                                         [pair[suffix],gradient,weight],pair[suffix])
                        second=pair["v"]
                        if options.get("amsgrad",False):
                            self._launch(emit_kernel("adam_max",[spec]*2,spec,options),[pair["max_v"],second],pair["max_v"])
                            second=pair["max_v"]
                        self._launch(emit_kernel("adam_update",[spec]*4,spec,options),
                                     [weight,pair["m"],second,gradient],weight,lr=lr,correction=correction)
                increment=getattr(getattr(self.torch.autograd,"graph",None),"increment_version",None)
                if increment:increment(weight)
        self._optimizer_step=step
        self._gradients=None

    def train_step(self, inputs, target, lr=None):
        if lr is not None:
            from .kernels import _float
            if type(lr) not in (int,float) or not math.isfinite(lr) or lr<0:
                raise ValueError("learning rate must be finite and nonnegative")
            _float(lr)
        training = self.plan.get("training")
        if training:
            values = dict(inputs)
            values[training["target"]] = target
            outputs = self.forward(values, require_tape=True)
            loss_name = training["loss_output"]
            # A model output can remain exposed for diagnostics, but only the
            # scalar loss seeds its compiled backward graph.
            gradients = self.backward({})
            loss = outputs[loss_name]
            prediction = outputs[training["model_output"]]
            rate = training.get("learning_rate", 0.01) if lr is None else lr
        else:
            if len(self.plan.get("outputs", [])) != 1:
                raise ValueError("native MSE training requires a single model output")
            name = self.plan["outputs"][0]
            logits_spec = self.specs[name]
            ce = self.plan.get("loss", "mse") == "cross_entropy"
            target_spec = {"shape":[logits_spec["shape"][0]],"dtype":"i64"} if ce else logits_spec
            self._check(target, target_spec, "target")
            input_values=inputs.values() if isinstance(inputs,dict) else inputs
            if any(value.device != target.device for value in input_values):
                raise ValueError("target and model inputs must share the CUDA device")
            if ce:
                labels = target.detach().cpu()
                if bool(((labels<0)|(labels>=logits_spec["shape"][1])).any()):
                    raise ValueError("cross_entropy labels outside the class range; ignore_index is unsupported")
            outputs = self.forward(inputs, require_tape=True)
            name, prediction = next(iter(outputs.items()))
            scratch = self._training[name]
            with self.torch.cuda.device(self._device):
                if ce:
                    row_spec={"shape":[logits_spec["shape"][0]]}
                    self._launch(emit_kernel("ce_rows",[logits_spec,target_spec],row_spec),[prediction,target],scratch["rows"])
                    self._launch(emit_kernel("mean",[row_spec],{"shape":[]}),[scratch["rows"]],scratch["loss"])
                    self._launch(emit_kernel("ce_grad",[logits_spec,target_spec],logits_spec),[prediction,target],scratch["seed"])
                else:
                    self._launch(emit_kernel("mse", [self.specs[name]]*2, {"shape": []}), [prediction, target], scratch["loss"])
                    self._launch(emit_kernel("mse_grad", [self.specs[name]]*2, self.specs[name]), [prediction, target], scratch["seed"])
            gradients = self.backward({name: scratch["seed"]})
            loss, rate = scratch["loss"], 0.01 if lr is None else lr
        self.update(rate)
        return {"loss": loss, "outputs": outputs, "prediction": prediction, "gradients": gradients}

    def close(self):
        if self._module is not None:
            with self.torch.cuda.device(self._device):
                self._stream.synchronize()
                with self._context_scope():
                    for module in self._tensor_modules:
                        self._driver.unload(module)
                    self._tensor_modules = []
                    self._driver.unload(self._module)
            self._module = None
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __del__(self):
        try:
            if not getattr(self, "_closed", True):
                self.close()
        except Exception:
            # Destructors cannot report asynchronous device errors safely.
            # Explicit close()/context-manager calls retain normal diagnostics.
            pass


NativeRuntime = Runtime

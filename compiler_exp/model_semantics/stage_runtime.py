from pathlib import Path
P=Path(__file__).parent/'candidate/openassembler/modelfront'
def edit(name,old,new):
 p=P/name;s=p.read_text()
 if old not in s:raise ValueError((name,old[:80]))
 p.write_text(s.replace(old,new,1))
edit('runtime.py','        self._training = {}','        self._training = {}\n        self._optimizer_state = {}\n        self._optimizer_step = 0')
edit('runtime.py','''if (not getattr(tensor, "is_cuda", False) or str(tensor.dtype) != "torch.float32"
                or tuple(tensor.shape) != shape_of(spec) or not tensor.is_contiguous()
                or tensor.data_ptr() % 4):''','''dtype = spec.get("dtype", "f32")
        expected = {"f32":"torch.float32", "i64":"torch.int64", "bool":"torch.bool"}.get(dtype)
        if (not getattr(tensor, "is_cuda", False) or str(tensor.dtype) != expected
                or tuple(tensor.shape) != shape_of(spec, dtypes=("f32","i64","bool")) or not tensor.is_contiguous()
                or tensor.data_ptr() % {"f32":4,"i64":8,"bool":1}.get(dtype,4)):''')
edit('runtime.py','needs a contiguous CUDA FP32 tensor','needs a contiguous CUDA tensor of the declared dtype')
edit('runtime.py','self._allocate(spec[\'shape\'])','self._allocate(spec[\'shape\'], dtype=spec.get("dtype","f32"))')
edit('runtime.py','''                self._training[output] = {"loss": self._allocate([]), "seed": self._allocate(spec["shape"])}''','''                self._training[output] = {"loss": self._allocate([]), "seed": self._allocate(spec["shape"])}
                if self.plan.get("loss") == "cross_entropy":
                    self._training[output]["rows"] = self._allocate([spec["shape"][0]])''')
edit('runtime.py','''        itemsize = 4 if dtype == "f32" else 2
        pad = 16 // itemsize
        backing = self.torch.empty([count+2*pad], dtype=self.torch.float32 if dtype == "f32" else self.torch.float16,''','''        torch_dtype = {"f32": self.torch.float32, "f16": self.torch.float16, "i64": self.torch.int64, "bool": self.torch.bool}[dtype]
        itemsize = {"f32":4,"f16":2,"i64":8,"bool":1}[dtype]
        pad = 16 // itemsize
        backing = self.torch.empty([count+2*pad], dtype=torch_dtype,''')
edit('runtime.py','math.prod(tensor.shape)*4','tensor.numel()*tensor.element_size()')
edit('runtime.py','def _launch(self, kernel, inputs, output, *, lr=None):','def _launch(self, kernel, inputs, output, *, lr=None, correction=None):')
edit('runtime.py','        params = (C.c_void_p*len(args))','        if kernel.op == "adam_update":\n            args.extend(C.c_float(x) for x in (lr, *correction))\n        params = (C.c_void_p*len(args))')
edit('runtime.py','        self._initialize(next(iter(devices)))','''        # Index validation runs on CPU before *any* native launch or state mutation.
        # This synchronizing check is visible host overhead, never a GPU fallback.
        for domain in self.plan.get("index_domains", []):
            indices = values[domain["name"]].detach().cpu()
            if bool(((indices < domain["lower"]) | (indices >= domain["upper"])).any()):
                raise ValueError(f"{domain['name']}: embedding index outside [{domain['lower']}, {domain['upper']})")
        self._initialize(next(iter(devices)))''')
a=P/'runtime.py';s=a.read_text();start=s.index('    def update(self, lr=0.01):');end=s.index('    def train_step(',start)
s=s[:start]+'''    def optimizer_state_dict(self):
        """CPU snapshot of all moments and step; no borrowed device storage."""
        from .optimizer import optimizer_options
        self.synchronize()
        return {"schema":1, "options":optimizer_options(self.plan.get("optimizer",{})),
                "step":self._optimizer_step,
                "moments":{n:{k:v.detach().cpu().clone() for k,v in pair.items()}
                           for n,pair in self._optimizer_state.items()}}

    def load_optimizer_state_dict(self, state):
        from .optimizer import optimizer_options
        options=optimizer_options(self.plan.get("optimizer",{}))
        if self._device is None or self._tape is not None or self._gradients is not None:
            raise ValueError("initialize forward and discard its tape before loading optimizer state")
        if (set(state)!={"schema","options","step","moments"} or state["schema"]!=1
                or state["options"]!=options or type(state["step"]) is not int or state["step"]<0):
            raise ValueError("optimizer checkpoint policy or step mismatch")
        expected=set(self.plan["backward"]["grad_parameters"]) if state["step"] and options["name"]!="sgd" else set()
        if set(state["moments"])!=expected:raise ValueError("optimizer checkpoint parameter names mismatch")
        for n,pair in state["moments"].items():
            if set(pair)!={"m","v"}:raise ValueError("optimizer checkpoint needs both moments")
            for k,v in pair.items():
                if (v.device.type!="cpu" or v.dtype!=self.torch.float32 or list(v.shape)!=self.specs[n]["shape"]
                        or not bool(self.torch.isfinite(v).all()) or k=="v" and bool((v<0).any())):
                    raise ValueError("invalid optimizer checkpoint moment")
        # All validation precedes replacement. Copies are host-to-device transfers.
        staged={n:{k:v.to(self._device).contiguous() for k,v in pair.items()} for n,pair in state["moments"].items()}
        self._optimizer_state,self._optimizer_step=staged,state["step"]

    def update(self, lr=0.01):
        from .optimizer import optimizer_options
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
                    self._launch(emit_kernel("sgd",[spec]*2,spec),[weight,gradient],weight,lr=lr)
                else:
                    if name not in self._optimizer_state:
                        pair={k:self._allocate(spec["shape"]) for k in ("m","v")}
                        with self._context_scope():
                            for value in pair.values():self._driver.memset(value.data_ptr(),0,value.numel()*4)
                        self._optimizer_state[name]=pair
                    pair=self._optimizer_state[name]
                    for suffix in ("m","v"):
                        self._launch(emit_kernel("adam_"+suffix,[spec]*3,spec,options),
                                     [pair[suffix],gradient,weight],pair[suffix])
                    self._launch(emit_kernel("adam_update",[spec]*4,spec,options),
                                 [weight,pair["m"],pair["v"],gradient],weight,lr=lr,
                                 correction=(lr/(1-options["beta1"]**step),math.sqrt(1-options["beta2"]**step)))
                increment=getattr(getattr(self.torch.autograd,"graph",None),"increment_version",None)
                if increment:increment(weight)
        self._optimizer_step=step
        self._gradients=None

'''+s[end:];a.write_text(s)
edit('runtime.py','''            outputs = self.forward(inputs, require_tape=True)
            name, prediction = next(iter(outputs.items()))
            self._check(target, self.specs[name], "target")''','''            name = self.plan["outputs"][0]
            logits_spec = self.specs[name]
            ce = self.plan.get("loss", "mse") == "cross_entropy"
            target_spec = {"shape":[logits_spec["shape"][0]],"dtype":"i64"} if ce else logits_spec
            self._check(target, target_spec, "target")
            if ce:
                labels = target.detach().cpu()
                if bool(((labels<0)|(labels>=logits_spec["shape"][1])).any()):
                    raise ValueError("cross_entropy labels outside the class range; ignore_index is unsupported")
            outputs = self.forward(inputs, require_tape=True)
            name, prediction = next(iter(outputs.items()))''')
edit('runtime.py','''                self._launch(emit_kernel("mse", [self.specs[name]]*2, {"shape": []}), [prediction, target], scratch["loss"])
                self._launch(emit_kernel("mse_grad", [self.specs[name]]*2, self.specs[name]), [prediction, target], scratch["seed"])''','''                if ce:
                    row_spec={"shape":[logits_spec["shape"][0]]}
                    self._launch(emit_kernel("ce_rows",[logits_spec,target_spec],row_spec),[prediction,target],scratch["rows"])
                    self._launch(emit_kernel("mean",[row_spec],{"shape":[]}),[scratch["rows"]],scratch["loss"])
                    self._launch(emit_kernel("ce_grad",[logits_spec,target_spec],logits_spec),[prediction,target],scratch["seed"])
                else:
                    self._launch(emit_kernel("mse", [self.specs[name]]*2, {"shape": []}), [prediction, target], scratch["loss"])
                    self._launch(emit_kernel("mse_grad", [self.specs[name]]*2, self.specs[name]), [prediction, target], scratch["seed"])''')
edit('reference.py','value.dtype != self.torch.float32','value.dtype != {"f32":self.torch.float32,"i64":self.torch.int64,"bool":self.torch.bool}[spec.get("dtype","f32")]')
edit('reference.py','            elif op == "add":','''            elif op == "embedding":
                result = torch.nn.functional.embedding(xs[0],xs[1])
            elif op == "embedding_grad":
                result = torch.zeros(dims)
                result.index_add_(0,xs[0].reshape(-1),xs[1].reshape(-1,dims[1]))
            elif op == "sum_to":
                result = xs[0].sum_to_size(dims)
            elif op == "broadcast_add":
                result = xs[0]+xs[1]
            elif op == "broadcast_mul":
                result = xs[0]*xs[1]
            elif op == "masked_softmax":
                mask = (torch.ones(xs[0].shape[-2:],dtype=torch.bool).tril() if attrs["causal"] else xs[1] if len(xs)>1 else torch.ones_like(xs[0],dtype=torch.bool))
                masked=xs[0].masked_fill(~mask,-float("inf"))
                result=torch.nan_to_num(torch.softmax(masked,dim=-1),nan=0.0)
            elif op == "add":''')
# Explicit replay rejection: host-validated dynamic indices and host Adam counters cannot be captured safely.
edit('replay.py','        self.runtime = Runtime(','        if plan.get("index_domains") or plan.get("loss") == "cross_entropy" or plan.get("optimizer",{}).get("name","sgd") != "sgd":\n            raise ValueError("indexed inputs, cross_entropy and Adam require direct execution")\n        self.runtime = Runtime(')
print('runtime overlay staged')

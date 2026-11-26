"""CPU-only mathematical controls for native graph numeric contracts.

This interpreter cannot accept CUDA tensors and is never a deployment fallback.
FP32 capture is additionally checked against the original Module/autograd. The
explicit half mode prescribes independent operand RNE casts for every forward,
input-gradient and weight-gradient matrix product, with FP32 master state.
"""
from __future__ import annotations

import math

from .graph import ModelGraphError, _torch


class CPUReference:
    def __init__(self, plan, weights):
        self.torch = _torch()
        self.plan = plan
        self.weights = {spec["name"]: self._check(weights[spec["name"]], spec).detach().clone()
                        for spec in plan["parameters"]}
        if plan.get("math", "fp32") not in ("fp32", "f16_rne_f32_accum"):
            raise ModelGraphError("unsupported CPU reference numerical contract")
        self.state = None
        self._optimizer = None

    def _check(self, value, spec):
        if (not isinstance(value, self.torch.Tensor) or value.device.type != "cpu"
                or value.dtype != {"f32":self.torch.float32,"i64":self.torch.int64,"bool":self.torch.bool}[spec.get("dtype","f32")] or list(value.shape) != spec["shape"]):
            raise ModelGraphError(f"mathematical reference requires shape-matched CPU tensors of declared dtype {spec.get('dtype', 'f32')}")
        return value

    def _matrix(self, a, b):
        if self.plan.get("math") == "f16_rne_f32_accum":
            a, b = a.to(self.torch.float16).float(), b.to(self.torch.float16).float()
            if not bool(self.torch.isfinite(a).all()) or not bool(self.torch.isfinite(b).all()):
                raise ModelGraphError("matrix operand overflows explicit FP16 finite-input contract")
        return a @ b

    def _steps(self, steps):
        torch = self.torch
        for step in steps:
            op, names, out = step["kernel"], step["inputs"], step["output"]
            xs, attrs = [self.state[x] for x in names], step.get("attrs", {})
            dims = self.plan["buffers"][out]["shape"]
            if op == "linear":
                result = self._matrix(xs[0], xs[1].t())
                if len(xs) == 3:
                    result = result + xs[2]
            elif op == "linear_dx":
                result = self._matrix(xs[0], xs[1])
            elif op == "linear_dw":
                result = self._matrix(xs[0].t(), xs[1])
            elif op == "bmm":
                result = self._matrix(xs[0], xs[1])
            elif op == "bmm_dx":
                result = self._matrix(xs[0], xs[1].transpose(-2, -1))
            elif op == "bmm_dw":
                result = self._matrix(xs[1].transpose(-2, -1), xs[0])
            elif op == "conv2d":
                result = torch.nn.functional.conv2d(xs[0], xs[1], xs[2] if len(xs) == 3 else None,
                                                    stride=attrs["stride"], padding=attrs["padding"],
                                                    dilation=attrs["dilation"], groups=1)
            elif op == "conv2d_dx":
                result = torch.nn.grad.conv2d_input(dims, xs[1], xs[0], stride=attrs["stride"],
                                                    padding=attrs["padding"], dilation=attrs["dilation"], groups=1)
            elif op == "conv2d_dw":
                result = torch.nn.grad.conv2d_weight(xs[1], dims, xs[0], stride=attrs["stride"],
                                                     padding=attrs["padding"], dilation=attrs["dilation"], groups=1)
            elif op == "bias_grad":
                result = xs[0].sum(dim=0)
            elif op == "relu":
                result = torch.relu(xs[0])
            elif op == "relu_backward":
                result = xs[0] * (xs[1] > 0)
            elif op == "sigmoid":
                result = torch.sigmoid(xs[0])
            elif op == "sigmoid_backward":
                result = xs[0] * (1 - xs[1]) * xs[1]
            elif op == "embedding":
                result = torch.nn.functional.embedding(xs[0],xs[1])
            elif op == "embedding_grad":
                result = torch.zeros(dims)
                result.index_add_(0,xs[0].reshape(-1),xs[1].reshape(-1,dims[1]))
            elif op == "sum_to":
                result = xs[0].sum_to_size(dims)
            elif op == "broadcast_to":
                result = xs[0].expand(dims).contiguous().clone()
            elif op == "broadcast_add":
                result = xs[0]+xs[1]
            elif op == "broadcast_mul":
                result = xs[0]*xs[1]
            elif op == "masked_softmax":
                mask = (torch.ones(xs[0].shape[-2:],dtype=torch.bool).tril() if attrs["causal"] else xs[1] if len(xs)>1 else torch.ones_like(xs[0],dtype=torch.bool))
                masked=xs[0].masked_fill(~mask,-float("inf"))
                result=torch.nan_to_num(torch.softmax(masked,dim=-1),nan=0.0)
            elif op == "add":
                result = xs[0] + (xs[1] if len(xs) == 2 else attrs["scalar"])
            elif op == "mul":
                result = xs[0] * (xs[1] if len(xs) == 2 else attrs["scalar"])
            elif op == "copy":
                result = xs[0].reshape(dims).contiguous().clone()
            elif op == "transpose2d":
                result = xs[0].t().contiguous()
            elif op == "permute":
                result = xs[0].permute(attrs["axes"]).contiguous()
            elif op == "softmax":
                result = torch.softmax(xs[0], dim=-1)
            elif op == "softmax_backward":
                result = xs[1] * (xs[0] - (xs[0] * xs[1]).sum(dim=-1, keepdim=True))
            elif op == "gelu":
                result = torch.nn.functional.gelu(xs[0], approximate="tanh")
            elif op == "gelu_backward":
                x = xs[1]
                u = math.sqrt(2 / math.pi) * (x + 0.044715 * x * x * x)
                t = torch.tanh(u)
                derivative = 0.5 * (1+t) + 0.5*x*(1-t*t)*math.sqrt(2/math.pi)*(1+3*0.044715*x*x)
                result = xs[0] * derivative
            elif op == "layernorm":
                result = torch.nn.functional.layer_norm(xs[0], (xs[0].shape[-1],), eps=attrs["epsilon"])
            elif op == "layernorm_backward":
                dy, x = xs
                centered = x - x.mean(dim=-1, keepdim=True)
                inverse = torch.rsqrt((centered*centered).mean(dim=-1, keepdim=True) + attrs["epsilon"])
                normalized = centered * inverse
                result = inverse * (dy-dy.mean(dim=-1, keepdim=True)-normalized*(dy*normalized).mean(dim=-1, keepdim=True))
            elif op in ("channel_affine", "channel_mul"):
                result = xs[0] * xs[1]
                if len(xs) == 3:
                    result = result + xs[2]
            elif op == "channel_grad":
                result = (xs[0]*xs[1]).reshape(-1, xs[0].shape[-1]).sum(dim=0)
            elif op == "channel_sum":
                result = xs[0].reshape(-1, xs[0].shape[-1]).sum(dim=0)
            elif op == "concat":
                result = torch.cat(xs, dim=attrs["axis"])
            elif op == "slice_axis":
                result = xs[0].narrow(attrs["axis"], attrs["start"], dims[attrs["axis"]]).contiguous()
            elif op == "fill":
                result = torch.full(dims, attrs.get("value", 1.0), dtype=torch.float32)
            elif op == "mse":
                result = ((xs[0] - xs[1]) ** 2).mean().reshape(dims)
            elif op == "mse_grad":
                result = (xs[0] - xs[1]) * (2 / xs[0].numel())
                if len(xs) == 3:
                    result = result * xs[2]
            elif op == "sgd":
                result = xs[0] - attrs["lr"] * xs[1]
            else:
                raise ModelGraphError(f"mathematical reference has no formula for {op!r}")
            self.state[out] = self._check(result, self.plan["buffers"][out])

    def forward(self, inputs):
        if isinstance(inputs, (tuple, list)):
            if len(inputs) != len(self.plan["inputs"]):
                raise ModelGraphError("mathematical reference input count differs from capture")
            inputs = dict(zip((x["name"] for x in self.plan["inputs"]), inputs))
        if set(inputs) != {x["name"] for x in self.plan["inputs"]}:
            raise ModelGraphError("mathematical reference input names differ from capture")
        with self.torch.no_grad():
            self.state = {**self.weights,
                          **{s["name"]: self._check(inputs[s["name"]], s) for s in self.plan["inputs"]}}
            self._steps(self.plan["steps"])
        return {name: self.state[name] for name in self.plan["outputs"]}

    def backward(self, seeds=None, *, zero_missing=False):
        if self.state is None:
            raise ModelGraphError("mathematical backward requires a forward state")
        seeds = seeds or {}
        backward = self.plan["backward"]
        if set(seeds) - set(backward["seed_grads"]):
            raise ModelGraphError("mathematical reference received an unknown output derivative")
        with self.torch.no_grad():
            for output, seed in backward["seed_grads"].items():
                spec = self.plan["buffers"][seed]
                if output in seeds:
                    value = self._check(seeds[output], spec)
                elif zero_missing or math.prod(spec["shape"]) == 1:
                    value = self.torch.full(spec["shape"], 0.0 if zero_missing else 1.0)
                else:
                    raise ModelGraphError("non-scalar mathematical backward needs an explicit output derivative")
                self.state[seed] = value
            self._steps(backward["steps"])
        return {"inputs": {name: self.state[value] for name, value in backward["grad_inputs"].items()},
                "parameters": {name: self.state[value] for name, value in backward["grad_parameters"].items()}}

    def train_step(self, inputs, target, lr=0.01):
        if len(self.plan["outputs"]) != 1 or "training" in self.plan:
            raise ModelGraphError("reference train_step requires a single-output base model plan")
        if type(lr) not in (int, float) or not math.isfinite(lr) or lr < 0:
            raise ModelGraphError("SGD learning rate must be finite and nonnegative")
        outputs = self.forward(inputs)
        name, prediction = next(iter(outputs.items()))
        ce=self.plan.get("loss")=="cross_entropy"
        self._check(target, {"shape":[prediction.shape[0]],"dtype":"i64"} if ce else self.plan["buffers"][name])
        with self.torch.no_grad():
            if ce:
                loss=self.torch.nn.functional.cross_entropy(prediction,target)
                seed=self.torch.softmax(prediction,dim=-1)
                seed[self.torch.arange(prediction.shape[0]),target]-=1
                seed/=prediction.shape[0]
            else:
                difference = prediction - target
                loss = (difference * difference).mean()
                seed = difference * (2 / difference.numel())
            gradients = self.backward({name: seed})
            from .optimizer import optimizer_options,torch_optimizer_kwargs
            options=optimizer_options(self.plan.get("optimizer",{}))
            if options=={"name":"sgd"}:
                for parameter,gradient in gradients["parameters"].items():self.weights[parameter]=self.weights[parameter]-lr*gradient
            else:
                if self._optimizer is None:
                    cls={"sgd":self.torch.optim.SGD,"adam":self.torch.optim.Adam,"adamw":self.torch.optim.AdamW}[options["name"]]
                    self._optimizer=cls(list(self.weights.values()),lr=lr,foreach=False,**torch_optimizer_kwargs(options))
                for group in self._optimizer.param_groups:group["lr"]=lr
                for parameter,gradient in gradients["parameters"].items():self.weights[parameter].grad=gradient
                self._optimizer.step();self._optimizer.zero_grad(set_to_none=True)
        return {"loss": loss, "outputs": outputs, "prediction": prediction, "gradients": gradients}

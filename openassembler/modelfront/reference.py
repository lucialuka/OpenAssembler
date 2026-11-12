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

    def _check(self, value, spec):
        if (not isinstance(value, self.torch.Tensor) or value.device.type != "cpu"
                or value.dtype != self.torch.float32 or list(value.shape) != spec["shape"]):
            raise ModelGraphError("mathematical reference requires shape-matched CPU FP32 tensors")
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
                result = xs[0] @ xs[1]
            elif op == "bmm_dx":
                result = xs[0] @ xs[1].transpose(-2, -1)
            elif op == "bmm_dw":
                result = xs[1].transpose(-2, -1) @ xs[0]
            elif op == "bias_grad":
                result = xs[0].sum(dim=0)
            elif op == "relu":
                result = torch.relu(xs[0])
            elif op == "relu_backward":
                result = xs[0] * (xs[1] > 0)
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
        self._check(target, self.plan["buffers"][name])
        with self.torch.no_grad():
            difference = prediction - target
            loss = (difference * difference).mean()
            seed = difference * (2 / difference.numel())
            gradients = self.backward({name: seed})
            for parameter, gradient in gradients["parameters"].items():
                self.weights[parameter] = self.weights[parameter] - lr * gradient
        return {"loss": loss, "outputs": outputs, "prediction": prediction, "gradients": gradients}

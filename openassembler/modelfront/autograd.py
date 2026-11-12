"""PyTorch host/autograd binding for the bounded OA-native graph runtime.

PyTorch owns tensor storage and autograd dispatch. Every arithmetic operation
in forward and backward is performed by the native runtime's OA cubins.
"""
from __future__ import annotations

from pathlib import Path
import weakref

import torch

from .graph import ModelGraphError


def _restore(structure, outputs):
    if "tensor" in structure:
        return outputs[structure["tensor"]]
    if "tuple" in structure:
        return tuple(_restore(x, outputs) for x in structure["tuple"])
    if "list" in structure:
        return [_restore(x, outputs) for x in structure["list"]]
    if "dict" in structure:
        return {key: _restore(value, outputs) for key, value in structure["dict"].items()}
    raise ModelGraphError("invalid native model output structure")


class _NativeCall(torch.autograd.Function):
    @staticmethod
    def forward(ctx, owner, *tensors):
        from .runtime import Runtime

        ninputs = len(owner.plan["inputs"])
        inputs, states = tensors[:ninputs], tensors[ninputs:]
        # A distinct runtime owns this invocation's tape. Multiple outstanding
        # forwards therefore cannot overwrite another invocation's saved values.
        runtime = Runtime(owner.plan, dict(zip(owner._state_names, states)), owner.build_dir)
        values = runtime.forward(dict(zip(owner._input_names, inputs)), require_tape=True)
        owner._autograd_runtimes.add(runtime)
        ctx.runtime = runtime
        ctx.owner = owner
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(*tensors)  # preserve PyTorch's in-place version checks
        return tuple(values[name] for name in owner.plan["outputs"])

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, *seeds):
        # Access saved_tensors to enforce mutation/version checks before launch.
        ctx.saved_tensors
        owner = ctx.owner
        provided = {name: grad for name, grad in zip(owner.plan["outputs"], seeds) if grad is not None}
        if any(not grad.is_contiguous() for grad in provided.values()):
            raise ModelGraphError("native autograd requires contiguous gradient seeds; use explicit contiguous seeds "
                                  "or native_train_step for OA-owned loss and optimizer arithmetic")
        if not provided:
            owner._autograd_runtimes.discard(ctx.runtime)
            return (None,) * (1 + len(owner._input_names) + len(owner._state_names))
        # Unused output derivatives must be zero. The runtime allocates/fills
        # missing output seeds with its OA fill kernel when zero_missing=True.
        result = ctx.runtime.backward(provided, zero_missing=True)
        owner._autograd_runtimes.discard(ctx.runtime)
        return (None, *(result["inputs"].get(name) for name in owner._input_names),
                *(result["parameters"].get(name) for name in owner._state_names))


class CompiledModel(torch.nn.Module):
    """A conventional Module whose graph forward/backward executes OA kernels.

    Move it and input tensors to the same CUDA device before calling it. The
    original parameter names are available through ``native_parameters()``;
    the ordinary ``parameters()`` iterator supports PyTorch interoperability.
    The adapter supports a single first-order backward per forward; higher
    derivatives and repeated retained-graph backward are unsupported.
    Backward seeds must be contiguous; implicit expanded gradients from an
    external sum/reduction require an explicit contiguous seed instead.
    An external PyTorch loss, optimizer or AccumulateGrad may execute vendor
    arithmetic. Use ``native_train_step`` for a complete OA-owned MSE/SGD update.
    Use ``native_forward`` for persistent OA-owned inference without autograd.
    """

    def __init__(self, plan, weights, build_dir):
        super().__init__()
        self.plan, self.build_dir = plan, Path(build_dir)
        self._input_names = tuple(spec["name"] for spec in plan["inputs"])
        self._state_names = tuple(spec["name"] for spec in plan["parameters"])
        self._state_attributes = []
        self._native_runtime = None
        self._training_replay = None
        self._autograd_runtimes = weakref.WeakSet()
        for index, spec in enumerate(plan["parameters"]):
            name = f"_oa_state_{index}"
            self._state_attributes.append(name)
            value = weights[spec["name"]].detach().clone()
            if spec.get("requires_grad", True):
                self.register_parameter(name, torch.nn.Parameter(value))
            else:
                self.register_buffer(name, value)

    def native_parameters(self):
        """Return canonical graph-state names and the live bound tensors."""
        return {name: getattr(self, attr) for name, attr in zip(self._state_names, self._state_attributes)}

    def _apply(self, fn, recurse=True):
        # A move/cast invalidates device bindings and preallocated launch state.
        self.close()
        return super()._apply(fn, recurse=recurse)

    def close(self):
        """Release the persistent native inference/training runtime, if created."""
        if self._native_runtime is not None:
            self._native_runtime.close()
            self._native_runtime = None
        if self._training_replay is not None:
            self._training_replay.close()
            self._training_replay = None

    def prepare_native_training(self, inputs, target, *, learning_rate=.01):
        """Capture one OA MSE/backward/SGD step with fixed external storage.

        Change input/target values in place on the preparation stream, then call
        the returned object's run(). Learning rate, pointer/layout and parameter
        bindings are fixed; parameter edits require closing and preparing again.
        Capture leaves parameter bytes unchanged. Returned step tensors borrow
        the replay's workspace until the next run.
        """
        from .replay import TrainingReplay
        if self._autograd_runtimes:
            raise RuntimeError('finish outstanding autograd tapes before preparing native training')
        inputs = self._persistent_binding(inputs)
        self.close()
        replay = TrainingReplay(self.plan, self.native_parameters(), self.build_dir,
                                inputs, target, learning_rate=learning_rate)
        replay._owner_ref = weakref.ref(self)
        self._training_replay = replay
        return replay

    def _persistent_binding(self, inputs):
        """Normalize exact inputs and refresh live state before persistent calls."""
        from .runtime import Runtime

        if self._autograd_runtimes:
            raise RuntimeError("native inference/training cannot mix with an outstanding autograd tape; "
                               "finish backward or release the autograd outputs first")
        if self._training_replay is not None and not self._training_replay._closed:
            raise RuntimeError('close the prepared training replay before ordinary native calls')
        if isinstance(inputs, torch.Tensor):
            inputs = (inputs,)
        if isinstance(inputs, (tuple, list)):
            if len(inputs) != len(self._input_names):
                raise ModelGraphError(f"native model expects {len(self._input_names)} tensor inputs")
            inputs = dict(zip(self._input_names, inputs))
        if not isinstance(inputs, dict) or set(inputs) != set(self._input_names):
            raise ModelGraphError("native model input names differ from its captured inputs")
        if self._native_runtime is not None and getattr(self._native_runtime, "_tape", None) is not None:
            raise RuntimeError("persistent native runtime has an outstanding backward tape")
        weights = self.native_parameters()
        if self._native_runtime is None:
            self._native_runtime = Runtime(self.plan, weights, self.build_dir)
        else:
            self._native_runtime.weights = weights
        return inputs

    def native_forward(self, inputs):
        """Run OA-only inference using persistent modules and workspace storage.

        Inputs may be a tensor, positional sequence or exact name mapping.
        Current parameters are rebound on every call. No autograd tape is
        created. Returned tensors preserve the captured output structure and
        borrow runtime storage: a subsequent native call may overwrite them.
        Finish any conventional autograd backward before using this route.
        """
        inputs = self._persistent_binding(inputs)
        with torch.no_grad():
            outputs = self._native_runtime.forward(inputs, require_tape=False)
        structure = self.plan.get("output_structure", {"tensor": self.plan["outputs"][0]})
        return _restore(structure, outputs)

    def native_train_step(self, inputs, target, *, learning_rate=0.01):
        """Run FP32 forward, mean-square loss, all gradients and SGD through OA.

        Inputs may be a tensor, a positional sequence or an exact name mapping.
        This method bypasses PyTorch autograd and optimizer arithmetic entirely.
        Returned gradients describe this step and replace each parameter's
        ``.grad`` binding; no cross-step gradient accumulation is implied.
        """
        inputs = self._persistent_binding(inputs)
        with torch.no_grad():
            result = self._native_runtime.train_step(inputs, target, lr=learning_rate)
        gradients = result["gradients"]["parameters"]
        for spec, attr in zip(self.plan["parameters"], self._state_attributes):
            if spec.get("requires_grad", True):
                getattr(self, attr).grad = gradients.get(spec["name"])
        return result

    def forward(self, *inputs):
        if self._training_replay is not None and not self._training_replay._closed:
            raise RuntimeError('close the prepared training replay before conventional autograd')
        if len(inputs) != len(self._input_names):
            raise ModelGraphError(f"native model expects {len(self._input_names)} tensor inputs")
        states = tuple(getattr(self, name) for name in self._state_attributes)
        values = _NativeCall.apply(self, *inputs, *states)
        outputs = dict(zip(self.plan["outputs"], values))
        structure = self.plan.get("output_structure", {"tensor": self.plan["outputs"][0]})
        return _restore(structure, outputs)


def compile_model(model, example_inputs, output_dir, *, opt_level="opt", math_mode="fp32",
                  storage_views=True, reuse_storage=False, sass_listings=True):
    """Build an exact-shape CPU model and return its native CUDA Module.

    CUDA toolkit tools are required for the build. CPU capture never executes
    the user's model on GPU; move the returned Module to CUDA explicitly.
    FP32 is the default; explicit f16_rne_f32_accum rounds each linear-product
    operand to FP16 while retaining FP32 storage, accumulation and output.
    """
    from .graph import capture_model, extract_weights
    from .optimization import optimize_plan
    from .runtime import build_bundle

    plan = optimize_plan(capture_model(model, example_inputs, math_mode=math_mode),
                         storage_views=storage_views, reuse_storage=reuse_storage)
    weights = extract_weights(model, plan)
    build_bundle(plan, weights, output_dir, opt_level=opt_level, sass_listings=sass_listings)
    return CompiledModel(plan, weights, output_dir)


def load_model(output_dir):
    """Load a saved native model bundle into a CPU-owned Module for CUDA use."""
    from .runtime import load_weights, validate_bundle

    plan, _, _ = validate_bundle(output_dir)
    weights = load_weights(output_dir, plan, torch, device="cpu")
    return CompiledModel(plan, weights, output_dir)

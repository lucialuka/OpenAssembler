"""Bounded training replay of validated OA kernels with fixed device pointers."""
from __future__ import annotations

import math
from openassembler.modelfront.runtime import Runtime


class TrainingReplay:
    """Own one runtime and captured training iteration, with stable external storage.

    Capturing enqueues no training update. The first run performs the first step.
    Input/target values may change in place on the bound stream. Parameters may
    change only through this object; mutation or replacement requires recapture.
    Result tensors borrow this runtime's storage until the next run.
    """

    def __init__(self, plan, weights, root, inputs, target, *, learning_rate=.01):
        import torch
        from openassembler.modelfront.kernels import _float
        _float(learning_rate)
        if isinstance(learning_rate, bool) or not math.isfinite(learning_rate) or learning_rate < 0:
            raise ValueError('learning rate must be finite and nonnegative')
        self.runtime = Runtime(plan, weights, root)
        self.inputs, self.target = dict(inputs), target
        self._closed = False
        self._graph = None
        self._torch = torch
        self._plan = self.runtime.plan
        self._weights = dict(weights)
        self._rate = learning_rate
        self._external = {}
        self._owner_ref = None
        try:
            with torch.no_grad():
                # Initialize functions/workspaces on the caller's chosen stream;
                # warm the forward path without changing parameters.
                output = self.runtime.forward(self.inputs)
                if len(output) != 1:
                    raise ValueError('training replay requires one model output')
                self.runtime._check(target, self.runtime.specs[next(iter(output))], 'target')
                self.runtime.discard_tape()
                self.runtime.synchronize()
                self._graph = torch.cuda.CUDAGraph()
                # CUDA requires capture on a non-default stream. The resulting
                # graph replays on the original caller stream; no storage moves.
                original = self.runtime._stream
                capture_stream = torch.cuda.Stream(device=self.runtime._device)
                capture_stream.wait_stream(original)
                try:
                    self.runtime._stream = capture_stream
                    self.runtime._stream_id = capture_stream.cuda_stream
                    with torch.cuda.graph(self._graph, stream=capture_stream):
                        self.result = self.runtime.train_step(self.inputs, target, learning_rate)
                finally:
                    self.runtime._stream = original
                    self.runtime._stream_id = original.cuda_stream
                self._ledger = list(self.runtime.launch_ledger)
            for name, tensor in {**self.inputs, '@target': target,
                                 **{'@weight/' + n: t for n, t in self._weights.items()}}.items():
                self._external[name] = (tensor, tensor.data_ptr(), tuple(tensor.shape), tensor.device)
            self._versions = {n: t._version for n, t in self._weights.items()}
        except Exception:
            self.close()
            raise

    @classmethod
    def load(cls, root, inputs, target, *, learning_rate=.01):
        """Load only validated saved OA cubins/weights and prepare fixed storage."""
        import torch
        from .runtime import validate_bundle, load_weights
        plan, _, _ = validate_bundle(root)
        return cls(plan, load_weights(root, plan, torch), root, inputs, target,
                   learning_rate=learning_rate)

    @property
    def launch_ledger(self):
        return self._ledger

    def run(self):
        torch, runtime = self._torch, self.runtime
        if self._closed:
            raise RuntimeError('native training replay is closed')
        owner = self._owner_ref() if self._owner_ref is not None else None
        if self._owner_ref is not None:
            if owner is None or owner._autograd_runtimes:
                raise RuntimeError('native training replay owner is unavailable or has an outstanding tape')
            current = owner.native_parameters()
            if set(current) != set(self._weights) or any(current[n] is not t for n,t in self._weights.items()):
                raise RuntimeError('native training replay parameter binding changed; prepare a new replay')
        if torch.cuda.current_stream(runtime._device).cuda_stream != runtime._stream_id:
            raise ValueError('native training replay requires its initialization CUDA stream')
        for name, (tensor, pointer, shape, device) in self._external.items():
            if (tensor.data_ptr() != pointer or tuple(tensor.shape) != shape or tensor.device != device
                    or str(tensor.dtype) != 'torch.float32' or not tensor.is_contiguous()):
                raise ValueError('native training replay external storage changed: ' + name)
        for name, tensor in self._weights.items():
            if tensor._version != self._versions[name]:
                raise RuntimeError('native training replay parameter changed outside replay: ' + name)
        for tensor, _, _, _ in self._external.values():
            tensor.record_stream(runtime._stream)
        with torch.cuda.device(runtime._device), runtime._context_scope():
            self._graph.replay()
        for name in self._plan['backward']['grad_parameters']:
            torch.autograd.graph.increment_version(self._weights[name])
        self._versions = {n: t._version for n, t in self._weights.items()}
        if owner is not None:
            for spec in owner.plan['parameters']:
                if spec.get('requires_grad', True):
                    self._weights[spec['name']].grad = self.result['gradients']['parameters'].get(spec['name'])
        return self.result

    def close(self):
        if not self._closed:
            self.runtime.synchronize()
            self._graph = None  # release captured function references before unloading
            self.runtime.close()
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __del__(self):
        try:
            if not getattr(self, '_closed', True):
                self.close()
        except Exception:
            pass

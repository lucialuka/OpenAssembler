"""Untimed per-boundary state checks for native or application experiments.

This helper does not choose tolerances or change model/optimizer arithmetic.
Callers provide their existing state comparator. It retains compact reports,
not all tensors from every update, and refuses incomplete trajectories.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any


def capture_training_state(model, optimizer, *, outputs=None, gradient_phase="retained") -> dict:
    """Clone all named parameters, buffers, gradients and owned optimizer state.

    Missing gradients, empty optimizer state and non-tensor optimizer settings
    are explicit entries. Parameter names replace process-local object IDs.
    This is a diagnostic synchronization/copy path and must stay outside timing.
    """
    import torch

    result: dict[str, Any] = {}

    def flatten(value, key):
        if isinstance(value, torch.Tensor):
            result[key] = value.detach().cpu().clone()
        elif isinstance(value, Mapping):
            if not value:
                result[key] = {}
            for name, item in value.items():
                flatten(item, f"{key}/{name}")
        elif isinstance(value, (tuple, list)):
            if not value:
                result[key] = []
            for index, item in enumerate(value):
                flatten(item, f"{key}/{index}")
        elif isinstance(value, (str, bool, int, float)) or value is None:
            result[key] = value
        else:
            raise TypeError(f"unsupported checkpoint value at {key}: {type(value).__name__}")

    parameters = dict(model.named_parameters())
    names_by_id = {id(parameter): name for name, parameter in parameters.items()}
    for name, parameter in parameters.items():
        flatten(parameter, "parameters/" + name)
        flatten(parameter.grad, f"gradients/{gradient_phase}/{name}")
        # Do not insert into optimizer.state while observing it.
        flatten(optimizer.state.get(parameter, {}), "optimizer/state/" + name)
    for name, value in model.named_buffers():
        flatten(value, "buffers/" + name)
    for index, group in enumerate(optimizer.param_groups):
        for key, value in group.items():
            if key == "params":
                try:
                    value = [names_by_id[id(parameter)] for parameter in value]
                except KeyError as exc:
                    raise ValueError("optimizer contains a parameter not owned by the observed model") from exc
            flatten(value, f"optimizer/param_groups/{index}/{key}")
    for parameter in optimizer.state:
        if id(parameter) not in names_by_id:
            raise ValueError("optimizer state belongs to a parameter outside the observed model")
    if outputs is not None:
        flatten(outputs, "outputs")
    return result


class TrajectoryGate:
    """Compare every required checkpoint in order without retaining tensors.

    A failed checkpoint does not stop subsequent diagnostics. The first failure
    means the first observed boundary outside the supplied gate, not the first
    arithmetic operation or first bitwise difference inside that boundary.
    """

    def __init__(self, required_order: Iterable[tuple[int, str]],
                 comparator: Callable[[dict, dict], dict]):
        self.required_order = list(required_order)
        if not self.required_order or len(set(self.required_order)) != len(self.required_order):
            raise ValueError("required checkpoint order must be nonempty and unique")
        if any(not isinstance(step, int) or isinstance(step, bool) or step < 0
               or not isinstance(phase, str) or not phase for step, phase in self.required_order):
            raise ValueError("checkpoint keys require a nonnegative integer step and a named phase")
        self.comparator = comparator
        self.rows: list[dict] = []

    def observe(self, step: int, phase: str, reference: dict, actual: dict) -> dict:
        index = len(self.rows)
        if index >= len(self.required_order) or (step, phase) != self.required_order[index]:
            expected = self.required_order[index] if index < len(self.required_order) else None
            raise ValueError(f"unexpected checkpoint {(step, phase)!r}; expected {expected!r}")
        diagnostic = self.comparator(reference, actual)
        if not isinstance(diagnostic, dict) or type(diagnostic.get("passed")) is not bool:
            raise ValueError("checkpoint comparator must return a dictionary with Boolean passed")
        row = {"step": step, "phase": phase, **diagnostic}
        self.rows.append(row)
        return row

    def summary(self) -> dict:
        missing = self.required_order[len(self.rows):]
        failed = [row for row in self.rows if not row["passed"]]
        return {
            "passed": not missing and not failed,
            "complete": not missing,
            "expected_checkpoints": len(self.required_order),
            "observed_checkpoints": len(self.rows),
            "failed_checkpoints": len(failed),
            "first_failed_observation": ({"step": failed[0]["step"], "phase": failed[0]["phase"]}
                                         if failed else None),
            "missing_checkpoints": [{"step": step, "phase": phase} for step, phase in missing],
            "scope": "first observed checkpoint outside the caller's gate; not first instruction or exact-bit divergence",
            "checkpoints": list(self.rows),
        }


def application_comparator(*, rtol: float, atol: float, state_rtol: float, state_atol: float):
    """Reuse the production application comparator with caller-fixed policies."""
    from openassembler.torchfront.worker import StateVerificationError, _save_state, compare_states

    def compare(reference: dict, actual: dict) -> dict:
        try:
            result = compare_states(_save_state(reference), _save_state(actual),
                                    rtol, atol, state_rtol, state_atol)
        except StateVerificationError as exc:
            return {"passed": False, "diagnostic": exc.diagnostics}
        return {"passed": True, "diagnostic": result}

    return compare

from __future__ import annotations

import pytest

from exp.whole_stack.trajectory_gate import (
    TrajectoryGate, application_comparator, capture_training_state,
)


def equal(reference, actual):
    return {"passed": reference == actual}


def test_missing_observation_cannot_pass():
    gate = TrajectoryGate([(1, "raw"), (1, "updated")], equal)
    gate.observe(1, "raw", {"x": 1}, {"x": 1})
    report = gate.summary()
    assert not report["passed"] and not report["complete"]
    assert report["first_failed_observation"] is None
    assert report["missing_checkpoints"] == [{"step": 1, "phase": "updated"}]


def test_first_failed_boundary_does_not_hide_later_failures_or_recoveries():
    gate = TrajectoryGate([(1, "raw"), (1, "updated"), (2, "raw"), (2, "updated")], equal)
    for step, phase, value in ((1, "raw", 1), (1, "updated", 2), (2, "raw", 1), (2, "updated", 3)):
        gate.observe(step, phase, {"x": 1}, {"x": value})
    report = gate.summary()
    assert report["complete"] and not report["passed"]
    assert report["failed_checkpoints"] == 2
    assert report["first_failed_observation"] == {"step": 1, "phase": "updated"}
    assert len(report["checkpoints"]) == 4


def test_duplicate_or_skipped_checkpoints_fail_closed():
    gate = TrajectoryGate([(1, "raw"), (1, "updated")], equal)
    with pytest.raises(ValueError, match="unexpected checkpoint"):
        gate.observe(1, "updated", {}, {})
    gate.observe(1, "raw", {}, {})
    with pytest.raises(ValueError, match="unexpected checkpoint"):
        gate.observe(1, "raw", {}, {})


@pytest.mark.parametrize("order", [[], [(1, "raw"), (1, "raw")], [(-1, "raw")], [(1, "")]])
def test_invalid_required_order_is_rejected(order):
    with pytest.raises(ValueError):
        TrajectoryGate(order, equal)


def test_comparator_errors_or_ambiguous_results_never_become_passes():
    gate = TrajectoryGate([(1, "raw")], lambda a, b: {"passed": 1})
    with pytest.raises(ValueError, match="Boolean"):
        gate.observe(1, "raw", {}, {})
    assert gate.summary()["observed_checkpoints"] == 0
    def fail(a, b):
        raise RuntimeError("comparison unavailable")
    gate = TrajectoryGate([(1, "raw")], fail)
    with pytest.raises(RuntimeError, match="unavailable"):
        gate.observe(1, "raw", {}, {})
    assert not gate.summary()["passed"]


def test_full_training_snapshot_is_immutable_and_includes_optimizer_configuration():
    torch = pytest.importorskip("torch")
    model = torch.nn.Linear(2, 2)
    model.register_buffer("counter", torch.tensor(0))
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, eps=1e-8, foreach=False)
    before = capture_training_state(model, optimizer, outputs={"loss": torch.tensor(0.)})
    assert before["gradients/retained/weight"] is None
    assert before["optimizer/state/weight"] == {}
    assert before["optimizer/param_groups/0/eps"] == 1e-8
    assert before["optimizer/param_groups/0/params/0"] == "weight"
    assert "buffers/counter" in before
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    after = capture_training_state(model, optimizer, gradient_phase="raw")
    assert "optimizer/state/weight/exp_avg" in after
    assert "optimizer/state/weight/exp_avg_sq" in after
    assert "optimizer/state/weight/step" in after
    old = after["parameters/weight"].clone()
    with torch.no_grad():
        model.weight.add_(1)
    assert torch.equal(after["parameters/weight"], old)
    assert torch.equal(before["buffers/counter"], torch.tensor(0))


def test_unowned_optimizer_parameters_are_rejected():
    torch = pytest.importorskip("torch")
    model = torch.nn.Linear(2, 2)
    other = torch.nn.Parameter(torch.ones(2))
    optimizer = torch.optim.AdamW([other])
    with pytest.raises(ValueError, match="not owned"):
        capture_training_state(model, optimizer)


def test_application_gate_keeps_missing_keys_metadata_and_nan_as_failures():
    torch = pytest.importorskip("torch")
    compare = application_comparator(rtol=.05, atol=.005, state_rtol=.05, state_atol=.005)
    reference = {"parameters/qkv.bias": torch.zeros(2), "gradients/raw/qkv.bias": torch.ones(2)}
    assert compare(reference, reference)["passed"]
    assert not compare(reference, {"parameters/qkv.bias": torch.zeros(2)})["passed"]
    assert not compare(reference, {**reference, "parameters/qkv.bias": torch.zeros(3)})["passed"]
    assert not compare(reference, {**reference, "parameters/qkv.bias": torch.tensor([float("nan"), 0.])})["passed"]
    assert not compare(reference, {**reference, "parameters/qkv.bias": torch.tensor([.007, 0.])})["passed"]

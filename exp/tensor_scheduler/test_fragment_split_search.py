from dataclasses import replace

import pytest

from openassembler.kernels.gemm import GemmPlan, generate

from fragment_split_search import certify_events, source_for_split, static_summary
from load_next_compute_current import FragmentEvent, fragment_events


@pytest.mark.parametrize("split", range(1, 8))
def test_every_split_preserves_full_work_and_accumulator_order(split):
    certificate = certify_events(fragment_events(4, split_n=split))
    assert certificate["loads"] == 4
    assert certificate["hmma"] == 128
    assert certificate["live_window_overwrite"] is False


def test_certificate_rejects_a_historical_loaded_map_with_live_slot_overwrite():
    events = list(fragment_events())
    # Before any k=0 HMMA consumes slot0, loading k=1 there is an overwrite.
    events.insert(1, FragmentEvent("load", 1, 0))
    with pytest.raises(ValueError, match="before its last use"):
        certify_events(events)


def test_certificate_rejects_omitted_chain_update_and_wrong_current_slot():
    events = list(fragment_events())
    with pytest.raises(ValueError, match="omits"):
        certify_events(events[:-1])
    first = next(i for i, event in enumerate(events) if event.kind == "hmma")
    events[first] = replace(events[first], slot=1)
    with pytest.raises(ValueError, match="unloaded or overwritten"):
        certify_events(events)


def test_split4_exactly_matches_integrated_opt_in_and_preserves_controls():
    plan = GemmPlan(1024, 4096, 1024)
    source, certificate = source_for_split(plan, split_n=4)
    assert source == generate(GemmPlan(1024, 4096, 1024, fragment_schedule="load_next"))
    assert certificate["hmma_control"] == 8
    assert certificate["epoch_drain"] == 15
    assert certificate["hardware_validated"] is False


def test_split_placements_native_encode_the_same_work_and_sync_envelope():
    plan = GemmPlan(128, 128, 128)
    rows = [static_summary(plan, split_n=split) for split in (2, 4, 6)]
    opcodes = [row["static"]["opcodes"] for row in rows]
    for name in ("HMMA", "LDSM", "LDGSTS", "LDGDEPBAR", "DEPBAR", "BAR"):
        assert len({counts[name] for counts in opcodes}) == 1
    assert len({row["certificate"]["source_sha256"] for row in rows}) == 3


def test_experiment_refuses_untested_policy_combinations():
    with pytest.raises(ValueError, match="baseline"):
        source_for_split(GemmPlan(1024, 4096, 1024, pointer_mode="direct"), split_n=2)

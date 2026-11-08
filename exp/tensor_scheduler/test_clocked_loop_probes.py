from pathlib import Path


SOURCE = Path(__file__).with_name("clocked_loop_probes.py").read_text()


def test_clocked_probe_reads_clock64_and_subtracts_empty_loop():
    assert "%clock64" in SOURCE
    assert "empty_loop" in SOURCE
    assert "overhead_subtracted_cycles" in SOURCE


def test_clocked_probe_covers_one_two_three_chain_cases():
    assert "for chains in (1, 2, 3)" in SOURCE
    assert "cycles_per_hmma" in SOURCE


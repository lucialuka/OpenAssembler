from dependency_scheduler import (
    EdgeKind,
    TensorInstruction,
    TensorLatency,
    build_dependency_graph,
    classify_edges,
    measure_schedule,
    search_candidates,
    schedule,
)


def _names(result):
    return [item.instruction.name for item in result]


def test_independent_hmma_chains_interleave_but_dependent_chain_waits():
    insns = [
        TensorInstruction("HMMA.A0", writes=("a0",), kind="hmma", accumulator_group="A", source_index=0),
        TensorInstruction("HMMA.B0", writes=("b0",), kind="hmma", accumulator_group="B", source_index=1),
        TensorInstruction("HMMA.A1", reads=("a0",), writes=("a1",), kind="hmma", accumulator_group="A", source_index=2),
        TensorInstruction("HMMA.B1", reads=("b0",), writes=("b1",), kind="hmma", accumulator_group="B", source_index=3),
    ]
    latency = TensorLatency(accum_raw=8, hmma_issue=2)
    graph = build_dependency_graph(insns, latency)
    result = schedule(graph, latency)

    assert _names(result) == ["HMMA.A0", "HMMA.B0", "HMMA.A1", "HMMA.B1"]
    cycles = {item.instruction.name: item.issue_cycle for item in result}
    assert cycles["HMMA.A1"] - cycles["HMMA.A0"] >= 8
    assert cycles["HMMA.B1"] - cycles["HMMA.B0"] >= 8
    assert cycles["HMMA.B0"] - cycles["HMMA.A0"] >= 2


def test_operand_readiness_and_observer_edges_are_separate():
    insns = [
        TensorInstruction("CP_ASYNC", kind="async_copy", token_writes=("g0",), source_index=0),
        TensorInstruction("WAIT_GROUP", kind="wait_group", token_reads=("g0",), token_writes=("s0",), source_index=1),
        TensorInstruction("LDSM", writes=("fragA",), kind="ldsm", token_reads=("s0",), source_index=2),
        TensorInstruction("HMMA", reads=("acc", "fragA"), writes=("acc",), kind="hmma", accumulator_group="A", operands=("fragA",), source_index=3),
        TensorInstruction("STG", reads=("acc",), kind="store", source_index=4),
    ]
    latency = TensorLatency(async_to_wait=2, wait_to_ldsm=3, ldsm_to_hmma=5, observer=11)
    graph = build_dependency_graph(insns, latency)
    kinds = classify_edges(graph)
    assert kinds[EdgeKind.OPERAND_READY.value] >= 3
    assert kinds[EdgeKind.OBSERVER.value] >= 1
    result = schedule(graph, latency)
    cycles = {item.instruction.name: item.issue_cycle for item in result}
    assert cycles["WAIT_GROUP"] - cycles["CP_ASYNC"] >= 2
    assert cycles["LDSM"] - cycles["WAIT_GROUP"] >= 3
    assert cycles["HMMA"] - cycles["LDSM"] >= 5
    assert cycles["STG"] - cycles["HMMA"] >= 11


def test_source_order_is_retained_when_no_independent_work_exists():
    insns = [
        TensorInstruction("HMMA0", writes=("a0",), kind="hmma", accumulator_group="A", source_index=0),
        TensorInstruction("HMMA1", reads=("a0",), writes=("a1",), kind="hmma", accumulator_group="A", source_index=1),
        TensorInstruction("STG", reads=("a1",), kind="store", source_index=2),
    ]
    result = schedule(build_dependency_graph(insns), TensorLatency())
    assert _names(result) == ["HMMA0", "HMMA1", "STG"]


def test_address_war_is_separate_from_targeted_observer_delay():
    insns = [
        TensorInstruction("LDSM", reads=("addr",), writes=("frag",), kind="ldsm", source_index=0),
        TensorInstruction("REUSE_ADDR", reads=("addr",), writes=("addr",), kind="scalar", source_index=1),
        TensorInstruction("HMMA", reads=("frag",), writes=("acc",), kind="hmma", accumulator_group="A", source_index=2),
        TensorInstruction("STG", reads=("acc",), kind="store", source_index=3),
    ]
    latency = TensorLatency(address_war=2, observer=15, ldsm_to_hmma=1, hmma_issue=1)
    graph = build_dependency_graph(insns, latency)
    kinds = classify_edges(graph)
    assert kinds[EdgeKind.ADDRESS_WAR.value] == 1
    assert kinds[EdgeKind.OBSERVER.value] == 1
    result = schedule(graph, latency)
    cycles = {item.instruction.name: item.issue_cycle for item in result}
    assert cycles["REUSE_ADDR"] - cycles["LDSM"] >= 2
    assert cycles["STG"] - cycles["HMMA"] >= 15


def test_candidate_search_exposes_issue_and_observer_tradeoff():
    insns = []
    for i in range(4):
        insns.extend([
            TensorInstruction(f"HMMA.A{i}", reads=(f"a{i - 1}",) if i else (),
                              writes=(f"a{i}",), kind="hmma", accumulator_group="A", source_index=len(insns)),
            TensorInstruction(f"HMMA.B{i}", reads=(f"b{i - 1}",) if i else (),
                              writes=(f"b{i}",), kind="hmma", accumulator_group="B", source_index=len(insns) + 1),
        ])
    insns.append(TensorInstruction("STG", reads=("a3",), kind="store", source_index=len(insns)))
    rows = search_candidates(insns)
    assert {row["name"] for row in rows} >= {"baseline_hmma8_targeted_observer", "independent_issue1_short_observer"}
    assert all(row["metrics"].accum_edges == 6 for row in rows)
    assert all(row["metrics"].observer_edges == 1 for row in rows)
    fastest = rows[0]
    assert fastest["metrics"].hmma_issue in (1, 2)
    assert fastest["metrics"].cycles <= max(row["metrics"].cycles for row in rows)


"""Experiment-only tensor dependency scheduler for SM80.

This module deliberately does not import or replace the production backend.  It
models the four dependency classes that the production tensor path currently
collapses into HMMA8 plus a fixed drain:

* accumulator RAW latency;
* independent HMMA initiation spacing;
* operand readiness (LDSM/cp.async/wait-group/barrier edges); and
* observer dependencies (stores, scalar consumers, barriers and control flow).

The model is intentionally small and deterministic so that a later A100 probe
can replace the default distances without changing the scheduling interface.
It schedules a synthetic instruction stream, not PTX or SASS directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from heapq import heappop, heappush
from typing import Mapping, Sequence


class EdgeKind(str, Enum):
    ACCUM_RAW = "accum_raw"
    HMMA_ISSUE = "hmma_issue"
    OPERAND_READY = "operand_ready"
    OBSERVER = "observer"
    ADDRESS_WAR = "address_war"
    REGISTER_RAW = "register_raw"
    MEMORY_ORDER = "memory_order"
    CONTROL = "control"


@dataclass(frozen=True)
class TensorLatency:
    """Clock distances for a candidate A100 schedule.

    The defaults are legacy synthetic hypotheses, not measured A100 latencies.
    Use the explicit partial clocked profile for a bounded24/8 model fit.
    `hmma_issue` is independent of `accum_raw`: the
    former limits a tensor-pipe reservation while the latter protects a chain
    that consumes the same accumulator group.
    """

    accum_raw: int = 8
    hmma_issue: int = 2
    ldsm_to_hmma: int = 4
    async_to_wait: int = 1
    wait_to_ldsm: int = 1
    observer: int = 15
    address_war: int = 1
    register_raw: int = 1
    memory_order: int = 1
    control: int = 1

    def __post_init__(self) -> None:
        if min(self.__dict__.values()) < 0:
            raise ValueError("tensor dependency distances must be non-negative")


@dataclass(frozen=True)
class TensorInstruction:
    """Small instruction description used by the experiment scheduler."""

    name: str
    reads: tuple[str, ...] = ()
    writes: tuple[str, ...] = ()
    kind: str = "scalar"
    accumulator_group: str | None = None
    operands: tuple[str, ...] = ()
    token_reads: tuple[str, ...] = ()
    token_writes: tuple[str, ...] = ()
    source_index: int = 0

    @property
    def is_hmma(self) -> bool:
        return self.kind == "hmma" or self.name.startswith("HMMA")

    @property
    def is_observer(self) -> bool:
        return self.kind in {"store", "scalar_consumer", "barrier", "control"}

    @property
    def is_memory(self) -> bool:
        return self.kind in {"load", "store", "async_copy", "ldsm", "barrier"}


@dataclass(frozen=True)
class DependencyEdge:
    producer: int
    consumer: int
    kind: EdgeKind
    distance: int
    resource: str | None = None


@dataclass(frozen=True)
class ScheduledInstruction:
    instruction_index: int
    issue_cycle: int
    instruction: TensorInstruction


@dataclass
class DependencyGraph:
    instructions: tuple[TensorInstruction, ...]
    edges: list[DependencyEdge] = field(default_factory=list)

    def predecessors(self, index: int) -> list[DependencyEdge]:
        return [edge for edge in self.edges if edge.consumer == index]

    def successors(self, index: int) -> list[DependencyEdge]:
        return [edge for edge in self.edges if edge.producer == index]


def _append_edge(
    edges: list[DependencyEdge],
    seen: set[tuple[int, int, EdgeKind, str | None]],
    producer: int,
    consumer: int,
    kind: EdgeKind,
    distance: int,
    resource: str | None = None,
) -> None:
    if producer == consumer:
        return
    key = (producer, consumer, kind, resource)
    if key in seen:
        return
    seen.add(key)
    edges.append(DependencyEdge(producer, consumer, kind, distance, resource))


def build_dependency_graph(
    instructions: Sequence[TensorInstruction],
    latency: TensorLatency | None = None,
) -> DependencyGraph:
    """Build explicit dependency edges while retaining source-order safety.

    The graph does not add an artificial edge between every HMMA.  The global
    initiation spacing is a resource constraint in :func:`schedule`, which is
    what permits independent accumulator chains to interleave.
    """

    latency = latency or TensorLatency()
    insns = tuple(instructions)
    edges: list[DependencyEdge] = []
    seen: set[tuple[int, int, EdgeKind, str | None]] = set()

    last_write: dict[str, int] = {}
    readers: dict[str, list[int]] = {}
    last_token_write: dict[str, int] = {}
    last_memory: int | None = None

    for consumer, ins in enumerate(insns):
        # Register RAW, including tensor accumulator reads.
        for reg in ins.reads:
            producer = last_write.get(reg)
            if producer is not None:
                p = insns[producer]
                kind = (
                    EdgeKind.ACCUM_RAW
                    if p.is_hmma and ins.is_hmma
                    and p.accumulator_group is not None
                    and p.accumulator_group == ins.accumulator_group
                    else EdgeKind.REGISTER_RAW
                )
                distance = latency.accum_raw if kind is EdgeKind.ACCUM_RAW else latency.register_raw
                _append_edge(edges, seen, producer, consumer, kind, distance, reg)

        # An instruction that overwrites a register must wait for earlier
        # asynchronous readers (LDSM/loads/stores) to consume the old value.
        # This is an address-WAR/read-completion edge, not an HMMA observer
        # edge. Keeping it separate prevents the targeted-drain policy from
        # accidentally imposing observer latency on pointer reuse.
        for reg in ins.writes:
            for reader in readers.get(reg, ()):
                p = insns[reader]
                kind = EdgeKind.ADDRESS_WAR if p.kind in {"ldsm", "load", "async_copy"} else EdgeKind.REGISTER_RAW
                distance = latency.address_war if kind is EdgeKind.ADDRESS_WAR else latency.register_raw
                _append_edge(edges, seen, reader, consumer, kind, distance, reg)
            if reg in last_write:
                _append_edge(edges, seen, last_write[reg], consumer, EdgeKind.REGISTER_RAW, latency.register_raw, reg)

        # cp.async / wait_group / barriers are represented as token edges.
        for token in ins.token_reads:
            producer = last_token_write.get(token)
            if producer is not None:
                p = insns[producer]
                distance = (
                    latency.async_to_wait
                    if p.kind == "async_copy" and ins.kind == "wait_group"
                    else latency.wait_to_ldsm
                    if p.kind == "wait_group" and ins.kind == "ldsm"
                    else latency.control
                )
                _append_edge(edges, seen, producer, consumer, EdgeKind.OPERAND_READY, distance, token)

        # LDSM completion must precede an HMMA that consumes its fragment.
        if ins.is_hmma:
            for producer in range(consumer - 1, -1, -1):
                p = insns[producer]
                if p.kind != "ldsm":
                    continue
                if not ins.operands or set(ins.operands) & set(p.writes):
                    _append_edge(edges, seen, producer, consumer, EdgeKind.OPERAND_READY, latency.ldsm_to_hmma)

        # Preserve relative ordering for memory operations and synchronization.
        if ins.is_memory and last_memory is not None:
            _append_edge(edges, seen, last_memory, consumer, EdgeKind.MEMORY_ORDER, latency.memory_order)
        if ins.is_memory:
            last_memory = consumer

        # Tensor results become visible to scalar/memory/control observers only
        # after the configured observer distance. This replaces an unconditional
        # drain with a targeted edge to the actual observer.
        if ins.is_observer:
            for producer in range(consumer - 1, -1, -1):
                p = insns[producer]
                if p.is_hmma and set(p.writes) & set(ins.reads):
                    _append_edge(edges, seen, producer, consumer, EdgeKind.OBSERVER, latency.observer)

        for token in ins.token_writes:
            last_token_write[token] = consumer
        for reg in ins.writes:
            last_write[reg] = consumer
            readers[reg] = []
        for reg in ins.reads:
            readers.setdefault(reg, []).append(consumer)

    return DependencyGraph(insns, edges)


def schedule(
    graph: DependencyGraph,
    latency: TensorLatency | None = None,
) -> list[ScheduledInstruction]:
    """List-schedule a graph with a global independent-HMMA reservation.

    Only ready instructions can issue.  If a ready HMMA would violate the
    independent initiation distance, the scheduler tries other ready work and
    advances the clock only when no legal instruction remains.
    """

    latency = latency or TensorLatency()
    n = len(graph.instructions)
    succ: list[list[DependencyEdge]] = [[] for _ in range(n)]
    indegree = [0] * n
    ready_cycle = [0] * n
    for edge in graph.edges:
        succ[edge.producer].append(edge)
        indegree[edge.consumer] += 1

    ready: list[tuple[int, int]] = []
    for i, degree in enumerate(indegree):
        if degree == 0:
            heappush(ready, (-len(succ[i]), graph.instructions[i].source_index))

    # The heap stores source positions; use a lookup because source_index is
    # stable and unique for a lowered block in this experiment.
    by_source = {ins.source_index: i for i, ins in enumerate(graph.instructions)}
    issue_cycle = 0
    last_hmma_cycle: int | None = None
    result: list[ScheduledInstruction] = []
    issued: set[int] = set()

    while ready:
        candidates = []
        while ready:
            _, source = heappop(ready)
            i = by_source[source]
            if i in issued:
                continue
            candidates.append(i)

        chosen: int | None = None
        for i in sorted(candidates, key=lambda x: (ready_cycle[x], graph.instructions[x].source_index)):
            ins = graph.instructions[i]
            if ready_cycle[i] > issue_cycle:
                continue
            if ins.is_hmma and last_hmma_cycle is not None:
                if issue_cycle - last_hmma_cycle < latency.hmma_issue:
                    continue
            chosen = i
            break

        if chosen is None:
            # Put all candidates back and advance to the next meaningful cycle.
            for i in candidates:
                heappush(ready, (-len(succ[i]), graph.instructions[i].source_index))
            issue_cycle = max(issue_cycle + 1, min(ready_cycle[i] for i in candidates))
            continue

        for i in candidates:
            if i != chosen:
                heappush(ready, (-len(succ[i]), graph.instructions[i].source_index))
        ins = graph.instructions[chosen]
        result.append(ScheduledInstruction(chosen, issue_cycle, ins))
        issued.add(chosen)
        if ins.is_hmma:
            last_hmma_cycle = issue_cycle
        for edge in succ[chosen]:
            ready_cycle[edge.consumer] = max(ready_cycle[edge.consumer], issue_cycle + edge.distance)
            indegree[edge.consumer] -= 1
            if indegree[edge.consumer] == 0:
                heappush(ready, (-len(succ[edge.consumer]), graph.instructions[edge.consumer].source_index))
        issue_cycle += 1

    if len(result) != n:
        raise ValueError("dependency graph contains a cycle")
    return result


def classify_edges(graph: DependencyGraph) -> Mapping[str, int]:
    """Return a compact edge-class count for experiment reports."""

    counts = {kind.value: 0 for kind in EdgeKind}
    for edge in graph.edges:
        counts[edge.kind.value] += 1
    return counts


@dataclass(frozen=True)
class ScheduleMetrics:
    """Static model metrics used to rank experiment-only candidates."""

    cycles: int
    hmma_count: int
    non_tensor_count: int
    observer_edges: int
    address_war_edges: int
    accum_edges: int
    operand_edges: int
    hmma_issue: int
    hardware_measurement: bool = False
    cycle_estimate_scope: str = "uncalibrated static dependency-model estimate"

    @property
    def hmma_density(self) -> float:
        return self.hmma_count / max(1, self.cycles)


def measure_schedule(result: Sequence[ScheduledInstruction], graph: DependencyGraph,
                     latency: TensorLatency | None = None) -> ScheduleMetrics:
    """Summarize a schedule without pretending it is a cycle-accurate model."""

    latency = latency or TensorLatency()
    kinds = classify_edges(graph)
    hmma_count = sum(item.instruction.is_hmma for item in result)
    non_tensor_count = len(result) - hmma_count
    cycles = (result[-1].issue_cycle + 1) if result else 0
    return ScheduleMetrics(
        cycles=cycles,
        hmma_count=hmma_count,
        non_tensor_count=non_tensor_count,
        observer_edges=kinds[EdgeKind.OBSERVER.value],
        address_war_edges=kinds[EdgeKind.ADDRESS_WAR.value],
        accum_edges=kinds[EdgeKind.ACCUM_RAW.value],
        operand_edges=kinds[EdgeKind.OPERAND_READY.value],
        hmma_issue=latency.hmma_issue,
    )


def candidate_policies() -> tuple[tuple[str, TensorLatency], ...]:
    """Return historical toy hypotheses, including unvalidated short distances.

    Names are retained for old experiment consumers.  None is a measured A100
    policy or a safe control table; use latency_profiles for an explicit,
    provenance-labelled partial calibration.  These values must not be encoded.
    """

    return (
        ("baseline_hmma8_targeted_observer", TensorLatency(accum_raw=8, hmma_issue=2, observer=15)),
        ("independent_issue1_targeted_observer", TensorLatency(accum_raw=8, hmma_issue=1, observer=15)),
        ("independent_issue1_short_observer", TensorLatency(accum_raw=8, hmma_issue=1, observer=7)),
        ("independent_issue2_short_observer", TensorLatency(accum_raw=8, hmma_issue=2, observer=7)),
        ("independent_issue1_ldsm2", TensorLatency(accum_raw=8, hmma_issue=1, ldsm_to_hmma=4, address_war=2, observer=15)),
    )


def search_candidates(instructions: Sequence[TensorInstruction],
                      policies: Sequence[tuple[str, TensorLatency]] | None = None
                      ) -> list[dict]:
    """Schedule a workload under bounded policies and rank by model cycles.

    Results include edge counts so a faster candidate that silently drops
    address-WAR or observer edges is visible and can be rejected by callers.
    """

    rows = []
    for name, latency in policies or candidate_policies():
        graph = build_dependency_graph(instructions, latency)
        result = schedule(graph, latency)
        metrics = measure_schedule(result, graph, latency)
        rows.append({"name": name, "latency": latency, "order": [item.instruction.name for item in result],
                     "metrics": metrics, "hardware_measurement": False,
                     "encoding_allowed": False,
                     "evidence_scope": "static model hypothesis; not a measured schedule"})
    return sorted(rows, key=lambda row: (row["metrics"].cycles, row["metrics"].non_tensor_count, row["name"]))


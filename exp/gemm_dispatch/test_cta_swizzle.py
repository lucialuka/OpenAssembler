from __future__ import annotations

from openassembler.kernels.gemm import GemmPlan, generate

from .cta_swizzle import TileGrid, SwizzleSpec, candidate_specs, grid_shape, logical_tile, locality_metrics, rewrite_grouped_n_prologue, traversal
from .heuristic_manifest import build_manifest


def test_grouped_n_matches_public_cutlass_coordinate_formula():
    grid = TileGrid(16, 32)
    spec = SwizzleSpec("grouped_n", 8)
    assert grid_shape(grid, spec) == (128, 4)
    assert logical_tile(0, 0, grid, spec) == (0, 0)
    assert logical_tile(7, 0, grid, spec) == (0, 7)
    assert logical_tile(8, 0, grid, spec) == (1, 0)
    assert logical_tile(127, 3, grid, spec) == (15, 31)


def test_traversals_are_bijections_and_morton_requires_full_groups():
    grid = TileGrid(16, 16)
    expected = {(m, n) for m in range(16) for n in range(16)}
    for spec in candidate_specs(grid):
        assert set(traversal(grid, spec)) == expected
    assert locality_metrics(grid, SwizzleSpec("grouped_n", 8))["tiles"] == 256


def test_grouped_n_rewrite_is_source_only_and_records_new_launch_grid():
    plan = GemmPlan(2048, 4096, 128)
    source = generate(plan)
    rewritten, metadata = rewrite_grouped_n_prologue(source, 16, 32, 8)
    assert rewritten != source
    assert "shr.u32 %r700,%r1,3;" in rewritten
    assert metadata["launch_grid"] == [128, 4, 1]
    assert "grouped_n" in metadata["cta_order"]


def test_manifest_is_explicit_and_does_not_promote_grouped_n():
    manifest = build_manifest()
    assert manifest["scope"].startswith("experiment-only")
    grouped = [row for row in manifest["cases"] if row["candidate"].startswith("grouped_n")]
    assert grouped and all(row["status"] == "untested" for row in grouped)
    assert "held_out_timing" in manifest["contract"]["required_gates"]

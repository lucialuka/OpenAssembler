"""CPU model and PTX rewrite for Ampere CTA traversal experiments.

The production GEMM generator currently launches one CTA per logical tile with
``m_fast`` or a fixed Morton-8 mapping.  This module models CUTLASS's public
``GemmIdentityThreadblockSwizzle<N>`` policy as an experiment.  It keeps the
mathematical tile coordinates explicit, emits a source-only prologue rewrite,
and records the required launch grid.  No production plan or CLI option is
changed here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class TileGrid:
    m_tiles: int
    n_tiles: int

    def __post_init__(self) -> None:
        if self.m_tiles <= 0 or self.n_tiles <= 0:
            raise ValueError("tile counts must be positive")

    @classmethod
    def from_shape(cls, m: int, n: int, tile: int = 128) -> "TileGrid":
        if m <= 0 or n <= 0 or m % tile or n % tile:
            raise ValueError("shape must be positive and divisible by tile")
        return cls(m // tile, n // tile)


@dataclass(frozen=True)
class SwizzleSpec:
    name: str
    width: int | None = None


def _morton_decode8(value: int) -> tuple[int, int]:
    """Decode six interleaved bits as the production Morton-8 mapping does."""

    x = y = 0
    for bit in range(3):
        x |= ((value >> (2 * bit)) & 1) << bit
        y |= ((value >> (2 * bit + 1)) & 1) << bit
    return x, y


def grid_shape(grid: TileGrid, spec: SwizzleSpec) -> tuple[int, int]:
    """Return the physical CUDA grid (x, y) for an experiment candidate."""

    if spec.name in {"m_fast", "morton8"}:
        return grid.m_tiles, grid.n_tiles
    if spec.name == "grouped_n":
        width = spec.width or 8
        if width not in (1, 2, 4, 8):
            raise ValueError("grouped_n width must be 1, 2, 4 or 8")
        if grid.n_tiles < width or grid.n_tiles % width:
            raise ValueError("grouped_n requires a width that divides N tiles")
        return grid.m_tiles * width, grid.n_tiles // width
    raise ValueError(f"unknown CTA order: {spec.name}")


def logical_tile(physical_x: int, physical_y: int, grid: TileGrid,
                 spec: SwizzleSpec) -> tuple[int, int]:
    """Map a physical CTA coordinate to logical (M-tile, N-tile)."""

    gx, gy = grid_shape(grid, spec)
    if not (0 <= physical_x < gx and 0 <= physical_y < gy):
        raise ValueError("physical CTA coordinate is outside the launch grid")
    if spec.name == "m_fast":
        return physical_x, physical_y
    if spec.name == "grouped_n":
        width = spec.width or 8
        return physical_x // width, physical_y * width + physical_x % width
    if grid.m_tiles % 8 or grid.n_tiles % 8:
        raise ValueError("morton8 requires complete 8x8 logical groups")
    linear = physical_y * grid.m_tiles + physical_x
    group = linear // 64
    lane = linear & 63
    group_m = group % (grid.m_tiles // 8)
    group_n = group // (grid.m_tiles // 8)
    local_m, local_n = _morton_decode8(lane)
    return group_m * 8 + local_m, group_n * 8 + local_n


def traversal(grid: TileGrid, spec: SwizzleSpec) -> list[tuple[int, int]]:
    """Enumerate logical tiles in physical launch order."""

    gx, gy = grid_shape(grid, spec)
    return [logical_tile(x, y, grid, spec) for y in range(gy) for x in range(gx)]


def locality_metrics(grid: TileGrid, spec: SwizzleSpec) -> dict:
    """Compute simple A/B tile reuse metrics for candidate comparison.

    A reuse is a consecutive run of equal M tile; B reuse is a consecutive run
    of equal N tile.  These metrics are diagnostic only and do not predict
    scheduler placement or L2 hit rate by themselves.
    """

    tiles = traversal(grid, spec)
    a_reuse = sum(a == b for (a, _), (b, _) in zip(tiles, tiles[1:]))
    b_reuse = sum(a == b for (_, a), (_, b) in zip(tiles, tiles[1:]))
    jumps = sum(abs(a - c) + abs(b - d) for (a, b), (c, d) in zip(tiles, tiles[1:]))
    return {"tiles": len(tiles), "a_reuse_edges": a_reuse,
            "b_reuse_edges": b_reuse, "tile_manhattan_jumps": jumps,
            "a_reuse_fraction": a_reuse / max(1, len(tiles) - 1),
            "b_reuse_fraction": b_reuse / max(1, len(tiles) - 1)}


def rewrite_grouped_n_prologue(source: str, m_tiles: int, n_tiles: int,
                               width: int = 8) -> tuple[str, dict]:
    """Rewrite a generated m-fast PTX prologue with grouped-N coordinates.

    The rewrite follows the public CUTLASS mapping:
    ``logical_m = blockIdx.x >> log_tile`` and
    ``logical_n = blockIdx.y << log_tile | (blockIdx.x & mask)``.
    The kernel must be launched with ``grid.x = m_tiles * width`` and
    ``grid.y = n_tiles / width``.  The experiment harness owns that launch.
    """

    grid = TileGrid(m_tiles, n_tiles)
    spec = SwizzleSpec("grouped_n", width)
    gx, gy = grid_shape(grid, spec)
    if "mov.u32 %r1,%ctaid.x; mov.u32 %r2,%ctaid.y;" not in source:
        raise ValueError("expected canonical CTA prologue")
    if source.count("mul.lo.u32 %r1,%r1,128;") != 1:
        raise ValueError("expected one logical M coordinate scaling site")
    log = width.bit_length() - 1
    mask = width - 1
    # Use OA's already-supported integer forms.  A literal SHL/OR pair is a
    # natural CUDA spelling, but the experiment must remain encodable by the
    # current OA backend so that it can be compared with ptxas later.
    mapping = [
        f"shr.u32 %r700,%r1,{log};",
        f"and.b32 %r701,%r1,{mask};",
        f"mul.lo.u32 %r702,%r2,{width};",
        "add.u32 %r2,%r702,%r701;",
        "mov.u32 %r1,%r700;",
    ]
    old = "mov.u32 %r1,%ctaid.x; mov.u32 %r2,%ctaid.y;"
    rewritten = source.replace(old, old + " " + " ".join(mapping), 1)
    return rewritten, {"cta_order": "grouped_n", "width": width,
                       "logical_tiles": [m_tiles, n_tiles],
                       "launch_grid": [gx, gy, 1],
                       "mapping": "logical_m=blockIdx.x>>log; logical_n=(blockIdx.y<<log)|(blockIdx.x&mask)"}


def candidate_specs(grid: TileGrid) -> list[SwizzleSpec]:
    """Return all valid traversal candidates for a logical grid."""

    rows = [SwizzleSpec("m_fast")]
    if grid.m_tiles % 8 == 0 and grid.n_tiles % 8 == 0:
        rows.append(SwizzleSpec("morton8"))
    for width in (8, 4, 2):
        if grid.n_tiles >= width and grid.n_tiles % width == 0:
            rows.append(SwizzleSpec("grouped_n", width))
            break
    return rows


"""Type annotations the pyfront subset understands. These are markers read
off a function's type hints at compile time, not runtime wrapper types --
calling a @kernel function in plain Python (no GPU) is intentionally not
supported, the same way you can't call PTX text as a Python function.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Scalar:
    ptx: str   # "s32" | "u32" | "s64" | "u64" | "f32"


@dataclass(frozen=True)
class ArrayType:
    elem: Scalar


i32 = Scalar("s32")
u32 = Scalar("u32")
i64 = Scalar("s64")
u64 = Scalar("u64")
f32 = Scalar("f32")


def Array(elem: Scalar) -> ArrayType:
    return ArrayType(Scalar("pred") if elem is bool else elem)


def element_bytes(kind: str) -> int:
    """Physical element width for an admitted logical scalar register kind."""
    return {"pred": 1, "f32": 4, "s32": 4, "u32": 4, "s64": 8, "u64": 8}[kind]


def logical_dtype(kind: str) -> str:
    return "bool" if kind == "pred" else kind

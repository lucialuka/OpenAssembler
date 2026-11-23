"""Type annotations the pyfront subset understands. These are markers read
off a function's type hints at compile time, not runtime wrapper types --
calling a @kernel function in plain Python (no GPU) is intentionally not
supported, the same way you can't call PTX text as a Python function.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Scalar:
    ptx: str   # "s32" | "u32" | "f32"


@dataclass(frozen=True)
class ArrayType:
    elem: Scalar


i32 = Scalar("s32")
u32 = Scalar("u32")
f32 = Scalar("f32")


def Array(elem: Scalar) -> ArrayType:
    return ArrayType(elem)

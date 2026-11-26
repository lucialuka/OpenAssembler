from typing import Final as Frozen
import typing as types
raise RuntimeError('static declarations are source-only')
GAIN: Frozen[int] = 3
BIAS: types.Final[int] = -(1 << 32) + 7
COEFFICIENTS: Frozen[tuple[int,int,int]] = (3,5,7)
GRID: Frozen[tuple[tuple[int,int],tuple[int,int]]] = ((11,13),(17,19))
SCALE: Frozen[float] = (1.0 / 8.0) * 4.0
NEGZERO: Frozen[float] = -0.0
SHIFT: Frozen[int] = (5 << 2) - 15 + (2**3 - 8)
COUNT: Frozen[int] = len(COEFFICIENTS)

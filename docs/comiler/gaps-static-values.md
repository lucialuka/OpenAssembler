# Explicit static numeric and tuple values

Status: implemented for the source contract below; broad G05 remains partial.

## Gap, solution and language boundary

Pure Python device helpers rejected useful module constants as captures and
constant tuple indices as memory access. The [initial experiment](../../compiler_exp/gap_closure/python_static_values/README.md)
reproduced scalar/tuple/FP32 failures. Its separate normalized source tree passed
functional checks but did not preserve original dependency and diagnostic identity.
It also exposed an annotated-rebinding counterexample, retained with the repair.
The [promotion cycle](../../compiler_exp/gap_closure/python_static_values/promotion/README.md)
now folds explicit static values inside the original source graph, before the
existing typed helper/entry checks. No user module is imported, evaluated with
`eval`, or executed during compilation.

```python
from typing import Final
from openassembler.pyfront import Array, i32, i64, device, kernel, tid_x

GAIN: Final[int] = 3
COEFFICIENTS: Final[tuple[int, int, int]] = (3, 5, 7)

@device
def affine(x: i64) -> i64:
    return x * GAIN + i64(COEFFICIENTS[-1])

@kernel
def apply(a: Array(i64), out: Array(i64), n: i32):
    i = tid_x()
    if i < n:
        out[i] = affine(a[i])
```

`typing.Final` and its direct/module aliases mark source values that the device
compiler specializes at compile time. The same marker from `typing_extensions`
is recognized statically. Numeric `int`, `float`, `bool`, finite tuples, nested
builtin tuple annotations and homogeneous `tuple[T, ...]` are supported. Optional
annotation arguments must exactly match the source literal types. An integer
annotated `Final[float]` does not silently become a float. Bare `Final` infers the
finite source value. Unmarked module variables remain unsupported captures.
Boolean values retain the existing device condition restrictions; this does not
add a new runtime Boolean storage or conversion type. Final declarations are a
typed device-source specialization contract; this does
not assert that arbitrary host code or import hooks preserve those bindings at
runtime. The original application host path remains responsible for host effects.

The finite evaluator supports numeric literals, previous/imported explicit Final
values, unary `+`, `-`, `~`, `not`, arithmetic `+ - * / // % **`, bit operations
and shifts, comparisons, finite tuples, static integer indexing/slicing and
`len(tuple)`. Evaluation follows bounded Python integer and binary64 arithmetic
before device literal conversion. Integer `/` produces binary64; floor and modulo
retain Python's signs. Each floating intermediate uses binary64; eventual FP32
uses the existing frontend conversion policy. Finite values and signed zero are
preserved; overflow/nonfinite values, invalid operations and mismatched declared
types reject. Powers require an integer exponent in [-2048,2048]. A result that
needs more than 2048 integer bits rejects. This adds no FP64 device arithmetic.

Tuple values are compile-time objects. A tuple must be resolved to a numeric
scalar through static indices/slices or `len` before entering device value
lowering. Negative tuple indices/slices use Python's tuple semantics. Runtime
indices, mutable lists/dicts, constructors/comprehensions, arbitrary calls,
strings, object attributes and runtime tuple storage reject. Array and loop index
types retain their existing device contracts; these static values do not add
runtime containers or dynamic Python evaluation.

## Scope, binding and identity

Constants resolve in their defining module, including local-package direct and
module aliases, relative imports and static re-exports through the
[helper source graph](gaps-python.md#static-local-package-device-helpers).
A local variable or parameter shadows a module constant. The compiler does not
replace a local read with the module value, infer definite initialization from a
later assignment, or override a locally shadowed `len`. Direct/module rebindings,
annotated rebindings, conditional assignment/deletion, tuple/subscript mutation,
import target mutation, cycles, invalid types and host calls reject with original
source paths and lines. A helper's constant specialization cache is keyed per
kernel so all dependent kernels retain their used constant records.

The evaluator bounds expression nesting to 32, tuple length to 256, integer
size to 2048 bits, and evaluation to 65,536 visited expression nodes per source
compilation. Existing graph limits remain 64 files/modules and 8 MiB of original
source. The versioned PTX graph permits at most 1,024 used static declarations.
Source hashes cover the original declaration and package files. Each used value
records its declaration path/name/line, exact tagged integer/Boolean/tuple value
or binary64 bits, and per-kernel use inventory. Ordinary local/remote builds,
source-change checks and native ownership carry that complete graph. A changed
constant invalidates the original graph; manifest/value tampering rejects.
Constant-only kernels also emit source identity, without inventing a helper call.

## Experiments and validation

The [baseline](../../compiler_exp/gap_closure/python_static_values/promotion/baseline.json)
retains **36 original source/entry rejections**. The
[offline gate](../../compiler_exp/gap_closure/python_static_values/promotion/candidate-offline.json)
passes **2,592 complete executions**: four original source trees × nine entries ×
four levels × two seeds × three lengths × PTX/normal/poisoned SASS. It covers
same-file/imported/re-export/entry-package declarations, helpers reused across
kernels, local shadowing, nested tuple indexing/slices/lengths, bounded powers,
64-bit wrapping, FP32 values and exact negative-zero bits. Independent full
output/input/guard/tail comparisons use arithmetic formulas outside the compiler.
The [generated expression gate](../../compiler_exp/gap_closure/python_static_values/promotion/property-probe.json)
retains 2,048 deterministic integer expression trees and independently constructed
arbitrary-precision results. These finite results do not prove general semantics.

The [negative controls](../../compiler_exp/gap_closure/python_static_values/promotion/negative-controls.json)
check 29 source/type/effect/binding/operation/size/cycle/index restrictions.
The [production regressions](../../tests/test_pyfront_static_values.py) pass
**118 focused tests**, including constant-only kernels, shared helper usage,
local `len` shadowing, exact FP32 rounding, original source changes and native
ownership/value tampering. A first focused test had an incorrect expected floor
reference: `(-17//3)*9 + (-17%3)` equals -53. Its
[failed log](../../compiler_exp/gap_closure/python_static_values/promotion/first-floor-reference-failed.log)
is retained; the corrected assertion changes no compiler result or tolerance.

The final original-source A100 report passes **1,440 launches**: original emitted
PTX controls plus four OA levels, four trees, nine kernels, two seeds, lengths
0/1/259/384 and three 128-thread CTAs. Archives retain original declarations,
full graph/PTX, cubins, manifests, disassembly and all before/after input/output
bytes. The production audit checks **9,338,880 complete bytes**, **144 exact native
word/register/resource outputs**, source/static/helper ownership and a corrupted
output rejection control. The earlier successful gate predates bounded-power and
local-`len` hardening and stays retained with exact old payloads and a separate
complete byte audit. Its results are not pooled as new coverage or timing.
Actual ptxas controls accept all four original source graphs / 36 kernels, with
thirty raw artifact identities. Observed IMAD.SHL aliases remain in listings;
the exact native word and numerical gates pass. There is no performance claim.

The full suites pass **3,656 supported tests** (Python 3.12.10/Torch 2.7.1,
zero skips) and **3,171 compiler-only tests with fifteen skips** (Python 3.14.5,
no Torch). The [frontend audit](../../compiler_exp/gap_closure/python_static_values/promotion/frontend-continuity.json)
checks seventeen prior files and 62 unchanged PTX kernels/rejections. The
[installed wheel](../../compiler_exp/gap_closure/python_static_values/promotion/wheel-audit.json)
checks 131 source/data identities and all 144 exact original-source PTX/native
outputs. The [public Modal check](../../compiler_exp/gap_closure/python_static_values/promotion/normal-remote-audit.json)
keeps the original static declarations through normal sender/worker/publication.
The [policy audit](../../compiler_exp/gap_closure/foundations_20261121/audit-20261124T022017Z.json)
retains all 49 policies and 98 original sessions. The
[complete earlier-native replay](../../compiler_exp/gap_closure/python_static_values/promotion/application-continuity.json)
checks **2,150 earlier kernel occurrences** with exact current words/registers,
including the complete previous package-helper corpus. The
[integration record](../../compiler_exp/gap_closure/python_static_values/promotion/validation.json)
binds final sources, complete independent numerical/ownership archives, suites,
retained failures and live/exported documentation. General
G05 objects, metaprogramming, host effects and unsupported runtime captures stay
open. Production imports no experiment module and uses no vendor fallback.

```sh
python compiler_exp/gap_closure/python_static_values/promotion/offline.py
python compiler_exp/gap_closure/python_static_values/promotion/negative_controls.py
python compiler_exp/gap_closure/python_static_values/promotion/property_probe.py
python compiler_exp/gap_closure/python_static_values/promotion/oracle_probe.py
modal run compiler_exp/gap_closure/python_static_values/promotion/run_modal.py
python -m pytest -q tests/test_pyfront_static_values.py
```

## Finite tuple iteration

The [tuple-loop extension](gaps-static-loops.md) now iterates finite static numeric
tuples in original source, preserving empty loops, selected targets, break,
continue, nested tuple/range loops and loop else. The separate normalized-source
prototype remains historical evidence. Original tuple/source/loop identity joins
native ownership; the final gate passes 2,400 A100 launches and 240 native outputs.
General runtime iteration/containers and arbitrary control semantics remain open.

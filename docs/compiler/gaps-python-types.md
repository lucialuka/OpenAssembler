# Explicit wide integer kernel types

Status: implemented

G03 previously exposed only i32/u32/f32 in Python kernels. The selected extension
adds explicit `i64` and `u64` annotations to scalar parameters, local values,
global arrays and scalar `@device` helpers. It preserves the typed device-language
boundary: integer arithmetic wraps at its declared width and no implicit mixed
integer/float or mixed-width promotion occurs. This narrows G03; it does not add
half/BF16/FP64 arithmetic, vector types or arbitrary Python numerical semantics.

```python
from openassembler.pyfront import Array, i32, i64, kernel, tid_x

@kernel
def add_offset(a: Array(i64), out: Array(i64), offset: i64, n: i32):
    i = tid_x()
    if i < n:
        out[i] = a[i] + offset
```

| Operation | Admitted contract |
|---|---|
| Add/subtract and `+=`/`-=` | Matching i64 or u64; low 64 bits retained, signed result interpreted in two's complement |
| `&`, `|`, `^`, `~` | Matching integer widths; bits preserved, including the high word |
| `<<`, `>>` | Constant counts 0..63 for wide integers, 0..31 for 32-bit integers; signed right shifts extend the sign, unsigned right shifts fill zeros; left shifts wrap |
| `<`, `<=`, `>`, `>=`, `==`, `!=`, truthiness | Exact signed/unsigned comparisons at the declared width, usable in branches and helper control flow |
| Unary minus | i64, wrapping `-INT64_MIN` to `INT64_MIN`; unsigned unary minus remains outside the contract |
| `i32(x)`, `u32(x)`, `i64(x)`, `u64(x)` | Explicit integer-only conversion: sign-extend signed sources when widening, zero-extend unsigned sources, truncate low destination bits when narrowing; same-width signedness conversion preserves bits |
| Typed integer cast of a literal | Same modulo-bit conversion, so `u64(-1)` is all ones; contextual literals beside typed operands must fit that type |
| Global arrays | 8-byte naturally aligned elements, with 8-byte address scaling and exact loads/stores |

Bare integer literals still default to i32. Wider literals need an explicit typed
cast or a wider contextual operand. Typed casts reject float and predicate inputs.
The existing `float()` and `int()` contracts remain unchanged and reject wide
integer operands. Wide multiply, division/remainder, min/max/abs, dynamic shift
counts in Python, wide shared arrays, wider range counters and wide array indices
reject with source diagnostics. Array bounds and negative runtime indices are
still unchecked; indices and range counters remain i32. Loop termination and
ordinary Python exceptions are not inferred.

The scalar ABI now distinguishes an unsigned64 scalar from an array pointer in
the frontend's source contracts. Wide kernels emit canonical
`openassembler-pyfront-types-v1` PTX comments with schema, entry name and ordered
parameter name/kind/dtype records. The typed verifier validates this metadata
against the actual PTX ABI and checks full element widths and high scalar bits.
The [continuity check](../../compiler_exp/gap_closure/python_wide/32bit-continuity.json)
compares 34 preexisting kernels with the exact pre-wide frontend snapshots and
retains byte-identical emitted PTX for all of them. Generic
verification uses bounded probe buffer extents; it does not infer application
shapes, all memory accesses or arbitrary launch contracts.

For saved runs, explicit launch specifications accept `i64`/`u64` buffer dtypes
and signed `i64`/unsigned `u64` scalar argument keys. The runner checks exact
initializer and argument ranges, packs all eight bytes, and converts signed
argument bits to the existing driver ABI. Use an explicit `--spec` for wide
workloads; the generic automatic runtime harness still guesses buffer bindings.
The [command reference](../COMMANDS.md) documents the specification format.

## Research and validation

The [candidate generator](../../compiler_exp/gap_closure/python_wide/make_candidate.py)
and [pre-extension source snapshots](../../compiler_exp/gap_closure/python_wide/before/)
retain the language gap and isolated candidate before production promotion.
The first native attempt encountered the existing subtraction encoder and
zero-count right-shift blockers, retained in
[candidate-native-blockers.json](../../compiler_exp/gap_closure/python_wide/candidate-native-blockers.json).
The separate backend study repaired and hardware-gated those operations,
comparisons and conversion snapshots before frontend integration. Production
never imports either experiment's overlay.

The independent [bit-reference probe](../../compiler_exp/gap_closure/python_wide/probe.py)
computes results using Python integer arithmetic, explicit modulo64 wrapping and
signed interpretation after each operation. It compares complete arrays against
PTX and safe/mined/sched/opt SASS execution. The
[production result](../../compiler_exp/gap_closure/python_wide/production-result.json)
retains source/PTX hashes, complete inputs/outputs and all native word/resource
identities. Signed and unsigned extrema, equal/unequal high and low words,
carry/borrow, shifts across bit 32, copied scalar arguments, helper loops,
conversion snapshots, immutable inputs and inactive tails are included.

The [final A100 report](../../compiler_exp/gap_closure/python_wide/hardware-20261121T132648Z.json)
and [artifact archive](../../compiler_exp/gap_closure/python_wide/hardware-20261121T132648Z.zip)
pass on NVIDIA A100-SXM4-40GB, Python 3.12.1 and CUDA toolkit 12.8.1. Ordinary Python
source compilation produced OA-owned cubins at all four optimization levels;
a separately named same-PTX ptxas executable supplies the matched control.
Seventy-five launches across changing inputs and active lengths 14, 0 and 16
passed 12,480 exact element comparisons plus allocation guards and full
input/output-tail checks. Five additional saved native explicit-spec cases
passed complete outputs. Two ordinary `build_source(...verify=True)` fixtures
also passed the typed automatic verifier, including signed and unsigned64 scalar
parameters. Current frontend and runner hashes match that hardware payload.
No performance comparison was measured.

The [first remote record](../../compiler_exp/gap_closure/python_wide/hardware-20261121T132455Z.json)
retains a reporting-harness failure: selecting a native spec cached executable
bytes into a manifest later passed to JSON serialization. Its partial results
remain archived; it is not a successful final hardware report. The corrected
harness uses a separate manifest copy for spec planning. Signed-immediate and
shift mnemonic aliases printed by nvdisasm remain in both logs; executable
ownership and bit comparisons passed in the final run.

The focused frontend, runner and typed-verifier selection passed 185 tests under
Python 3.12.10. The separate
[failed test selection](../../compiler_exp/gap_closure/python_wide/failed-test-selection.log)
retains an initial typo in the typed-verifier test filename; no tests ran there.

```sh
python3 compiler_exp/gap_closure/python_wide/probe.py --production
python3 -m pytest -q tests/test_pyfront_wide.py tests/test_runner_wide.py tests/test_pyfront.py tests/test_pyfront_static.py tests/test_pyfront_semantics.py tests/test_pyfront_helpers.py tests/test_pyfront_helper_control.py tests/test_runner.py tests/test_verify_typed.py
modal run compiler_exp/gap_closure/python_wide/run_modal.py
```

Implementation: [frontend](../../openassembler/pyfront/compile.py),
[type markers](../../openassembler/pyfront/types.py),
[typed helpers](../../openassembler/pyfront/helpers.py),
[explicit runner](../../openassembler/runner.py),
[wide frontend regressions](../../tests/test_pyfront_wide.py) and
[runner regressions](../../tests/test_runner_wide.py). These are bounded SM80
contracts; the remaining G03 types and numerical policies need separate work.

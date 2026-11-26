# Typed Python atomic primitives

Status: implemented

## Gap and contract

G08 lacked a source-level atomic route even after the native backend admitted
32-bit global atomics. Kernel source now exports `atomic_add`,
`atomic_exchange` and `atomic_compare_exchange` from `openassembler.pyfront`.
Each operation returns the previous element value and has relaxed GPU-scope
ordering. Addition accepts global `i32`/`u32` elements and wraps modulo `2^32`;
exchange and compare/exchange accept `u32` bits. CAS arguments are the comparison
bits followed by the replacement bits. These operations order their atomic
location; they do not publish unrelated non-atomic data.

```python
from openassembler.pyfront import Array, atomic_add, i32, kernel, tid_x

@kernel
def tickets(counter: Array(i32), previous: Array(i32)):
    previous[tid_x()] = atomic_add(counter, 0, 1)
```

The target must be a kernel `Array` or [checked `Buffer`](gaps-buffers.md)
parameter. Array indices are one `i32` expression and require caller-provided
bounds. Buffer coordinates follow its rank and checked layout: an out-of-range
operation returns zero and performs no update. Read-only buffers reject writes.
Index expressions and values evaluate from left to right, once each, with
snapshots preserving nested atomic effects. Results remain thread-dependent
even when all operands are uniform, so a barrier selected by an atomic return
fails the existing convergence check. Pure device helpers reject these effects.

The production path is [primitive lowering](../../openassembler/pyfront/primitives.py),
the ordinary frontend and buffer access path, and
[native atomic selection](gaps-atomics.md). It imports no experiment code.
Keywords, mismatched value/index types, FP/wide/shared atomics and arbitrary
target expressions reject with source diagnostics. These three primitives do
not close general warp collectives, async copies, tensor operations or G08.

## Experiments and validation

The [retained candidate experiment](../../compiler_exp/gap_closure/python_primitives/candidate-result.json)
records the original unsupported-call diagnostic, then independent integer
reference comparisons through Python → PTX → encoded SASS, including normal
and poisoned registers. The
[production replay](../../compiler_exp/gap_closure/python_primitives/production-result.json)
retains exact complete state and native identities. These sequential simulation
results establish values and effects within their domain, not concurrent order.

The [final A100 gate](../../compiler_exp/gap_closure/python_primitives/hardware-20261121T142508Z.json)
passes **65 launches** on an A100-SXM4-40GB with Python 3.12.1 and CUDA toolkit
12.8.1. Seven Python kernels run through all four OA optimization levels and a
separate same-PTX ptxas control. Contention checks include exact addition totals
and old-value tickets, one CAS winner, and a complete connected exchange trail;
they do not prescribe GPU thread order. Checked invalid coordinates include
both i32 extremes. Nested expression evaluation, discarded returns, inactive
tails, full input/state/output bytes and allocation guards pass.

The [archive](../../compiler_exp/gap_closure/python_primitives/hardware-20261121T142508Z.zip)
retains source, manifests, cubins and listings. The report records uploaded
production source hashes, harness/archive identities and complete outputs.
The [regressions](../../tests/test_pyfront_primitives.py) cover 30 positive and
negative cases; the [source test log](../../compiler_exp/gap_closure/python_primitives/final-source-tests-20261123.log)
records their latest rerun. There is no timing or general memory-ordering claim.

```sh
python compiler_exp/gap_closure/python_primitives/probe.py --production
python -m pytest -q tests/test_pyfront_primitives.py
modal run compiler_exp/gap_closure/python_primitives/run_modal.py
```

Fresh hardware runs write dated evidence and retain earlier failures/results.
Broader Python GPU primitives and stronger source-level ordering policies need
their own candidate, independent semantic tests and hardware validation.

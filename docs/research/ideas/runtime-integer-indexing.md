# Exact runtime integer indexing and F2I source collection

Status: implemented

The November 11 three-task report isolated a genuine large-grid native
`i % cols` correctness defect. The November 12 causal A100 experiment identifies
and repairs a missing conversion-source dependency rather than changing the
division formula or specializing all dimensions to masks/shifts.

`F2I` has a variable-latency result and asynchronously collects its source.
The production scheduler already assigned a write scoreboard to its result,
but omitted it from the source-reader protection list. In the failing safe
remainder sequence, `F2I ... R7,R5` was immediately followed by `MOV R5,0`;
the latter could overwrite the conversion input before collection. The repair
uses the existing read-scoreboard/WAR machinery for F2I. Source order,
arithmetic fragments, operands, allocation and numerical policy are preserved.

All four optimization levels reproduce failure without the protection.
F2I-only protection passes every isolated hardware case, whereas an I2F-only
control still fails. Two independent expanded A100-SXM4-40GB calls each pass
432 exact-output/canary checks and 156,477,312 checked integer results,
including exact-source ptxas controls. Signed and unsigned runtime quotient and
remainder tests include non-power-of-two divisors, large grids, quotient
boundaries, random values, signed extrema and negative signed divisors. The
contract excludes zero divisors and signed `INT_MIN/-1` quotient overflow.

The permanent [experiment and raw evidence](../../../exp/native_indexing/README.md)
retain failed controls, source/binary/compiler hashes and both passing sessions.
[Offline regression tests](../../../tests/test_runtime_indexing.py) check
conversion-source/result dependencies separately and exact arithmetic under
all four optimization levels. Functional emulation has no operand-collection
timing and remains separate from real-A100 validation.

The fix is in [the production scheduler](../../../openassembler/backend/compile.py).
It does not relax HMMA8/drain15, async-copy visibility, LDSM address-WAR or any
existing tensor dependency. All installed GEMM/rowwise plans require a retained
machine-text/register continuity audit before their compiler identities are
refreshed. No kernel performance improvement or complete-model numerical
admission follows from this correctness repair.

# Narrow scalar operations emitted by CUDA

Status: original-source scalar admission boundary established; no candidate.

After [narrow entry ABI admission](gaps-narrow-parameters.md), the exact original
CUDA Boolean controls from the [physical Boolean study](gaps-boolean-abi.md) now
parse their parameter declarations. They still reject at `setp.ne.s16` in the
scalar kernel and `setp.eq.s16` in the array kernel; the latter also contains
`selp.u16`. The [baseline report](../../compiler_exp/gap_closure/narrow_scalars/baseline.json)
checks every original nvcc/ptxas artifact identity and records the current earliest
scalar-contract diagnostics. It changes no original source or toolkit result.

G11 needs a faithful narrow register contract. Signed comparisons must interpret
the selected low-width sign bit; unsigned/bit comparisons must ignore unrelated
upper physical bits. Moves/selection need exact result width and alias/guard
snapshots. Conversions, arithmetic, shifts, modifiers and overflow behavior each
require independent definitions before admission. A 32-bit physical register does
not establish correct PTX16 behavior by itself. Existing unsupported forms remain
explicit and CPU interpreter agreement alone is insufficient.

The [experiment dossier](../../compiler_exp/gap_closure/narrow_scalars/README.md)
defines poisoned-upper-bit, signed-extreme, alias/guard/loop/immediate and original
CUDA controls, independent references and A100/source/native/resource/integration
gates. No numerical execution, performance or generic CUDA support is claimed by
this baseline. Boolean source launch/storage remains a separate logical contract,
and the full 51-gap program is incomplete.

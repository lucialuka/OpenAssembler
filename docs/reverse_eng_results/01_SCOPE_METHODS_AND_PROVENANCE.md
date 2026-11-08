# Scope, method and provenance

## What was reverse engineered

The work targets observable behavior needed to generate and execute supported
SM80 kernels:

1. PTX parsing and operation semantics for the supported subset.
2. SASS instruction words, operand fields and control bundles.
3. Register, predicate, memory and branch encodings.
4. ptxas instruction selection and arithmetic fragments.
5. scoreboards, stalls, reuse hints and selected dependency behavior.
6. CUDA ELF/cubin sections and `.nv.info` resource metadata.
7. A100 execution behavior for scalar, memory and MMA probe kernels.

It does not claim to recover NVIDIA's internal source code, proprietary cuBLAS
device modules, firmware, undocumented compiler heuristics in full, or a
complete cross-architecture binary ABI.

## Black-box workflow

The standard experiment has six stages:

1. Generate a small PTX input whose operands vary by lane or parameter so the
   compiler cannot fold away the behavior.
2. Assemble it with the pinned CUDA toolkit and capture the cubin and
   `nvdisasm -hex -c` listing.
3. Mutate one source operand, modifier or control field at a time and compare
   words, text, metadata and execution.
4. Re-encode the recovered representation with OpenAssembler and compare the
   exact words and disassembly where the form is supported.
5. Run the same input through the PTX interpreter, SASS emulator and real A100
   when the experiment needs hardware timing or synchronization evidence.
6. Retain hashes, toolkit/device identity, command line, raw output and the
   reason a candidate was accepted or rejected.

Printed SASS text is only one observation. Hidden bits can affect behavior even
when `nvdisasm` prints the same mnemonic, so text equivalence is never the only
gate.

## Evidence classes

| Evidence | Supports | Does not support |
|---|---|---|
| Word/field round trip | Encoding correctness for the tested form | Hardware timing or arbitrary modifiers |
| PTX interpreter vs SASS emulator | Functional agreement in the emulator model | Real GPU behavior or compiler equivalence |
| A100 output/canary check | Execution correctness for the launch and inputs | General shape or dtype coverage |
| Clocked `%clock64` probe | Timing of the exact probe sequence | Isolated latency of every instruction or peak throughput |
| Nsight counter | A measured stall/traffic/utilization signal | Causal proof without controlled paired variants |
| ptxas comparison | Same-source compiler comparison | cuBLAS superiority or library-level equivalence |
| cuBLAS/cuBLASLt observation | Public dispatch/control behavior | Extraction of proprietary kernel code |

## Provenance rules

The retained records identify CUDA toolkit, driver, GPU model, source hash and
candidate method. A historical experiment may be superseded; superseded data
remain available but are labeled. Raw records are not silently rewritten to
make a failed candidate look successful.


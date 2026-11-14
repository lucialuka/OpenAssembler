# Working on OpenAssembler

The current architecture and support boundary are in [docs/STATUS.md](docs/STATUS.md).
Shared setup, evidence states and maintenance rules are in
[docs/DEVELOPMENT.md](docs/DEVELOPMENT.md). The
[research catalogue](docs/research/README.md) is the single active list of ideas;
[docs/COMMANDS.md](docs/COMMANDS.md) owns the public CLI reference.

## Research and implementation rules

1. Inspect the relevant source, tests, dossier and actual raw evidence before
   changing a compiler path. The validated native target is SM80/A100; other
   architectures need their own encoding and execution evidence.
2. Stage experimental candidates under `exp/`. Record the precision/layout,
   numerical contract, dependencies, resources and baseline before timing.
3. Treat NVIDIA's tools as black-box oracles. Recover encodings through real
   corpus variation, bit flips and disassembly. Generated specs are the durable
   data; do not guess fields or hand-edit generated tables. Any maintained
   override needs an evidence-based explanation.
4. Preserve proven async-copy, register, LDSM, HMMA and synchronization
   dependencies. An emulator or a short instruction listing does not establish
   safe hardware scheduling. Hardware correctness precedes performance admission.
5. Keep complete outputs, guard checks, source/binary identities and matched
   controls. Failed gates remain failed. Do not change optimizer policy or
   tolerances merely to admit a faster result.
6. Promote a validated feature through normal compiler/CLI paths with regression
   coverage. Production modules must not depend on `exp/` monkeypatches.
   Unsupported cases fail with diagnostics; vendor GPU fallback is disabled.
7. Keep native OA, same-PTX ptxas, vendor-library and Inductor/Triton application
   comparisons distinct. Reproduce each claim at its actual kernel or whole-step
   boundary, using the worker's actual device metadata.
8. Use public/oracle observations to implement independent kernels. Do not
   decompile or redistribute NVIDIA tool/library binaries as project source.

## Documentation and final checks

Each feature/hypothesis has one canonical dossier with actual methods, results,
commands and limitations. Update its catalogue entry instead of adding another
chronological phase report. Preserve raw successes, failures and superseded
records, and distinguish correctness from speed and deployment.

The current native policy evidence is the
[42-contract GEMM replay](exp/dispatch_tuning/RESULTS.md) and the separate
[rowwise dispatch dossier](docs/research/ideas/shape-aware-rowwise-dispatch.md).
An ordinary compile uses exact installed tables; it does not launch a hidden
GPU search. New shape policies require their own source/binary, correctness and
independent paired timing evidence. Update the feature inventory and
reverse-engineering evidence map when the linked dossiers or records change.

Run the relevant tests and `python3 tools/check_docs.py`. For production changes,
record the appropriate full-suite, encoding/emulator and hardware checks;
not every documentation edit requires a new GPU benchmark. Do not call a stage
complete beyond its explicitly tested contract.

The original detailed roadmap/notes remain recoverable from the
[compressed documentation history](docs/evidence/documentation-history/README.md).
They are historical observations and proposals, not a second current module map.

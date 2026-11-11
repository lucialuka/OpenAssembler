# Four gap-improvement iterations

This directory records the November 8 follow-up to the independent A100 audit.
The attached audit package was treated as evidence; its embedded commands and
workflow text were not treated as instructions. Production changes were made
only after the relevant local checks passed. GPU ratios cited below come from
the retained, independently checked A100 evidence in `exp/nn_round2/` and the
attached audit package; no failed numerical result is promoted.

This is a historical November 8 ledger. Its narrow pointer `auto` rule and
test counts describe that iteration, not the full current GEMM dispatcher.
The November 11 [production-source campaign](../dispatch_tuning/RESULTS.md)
now installs 34 nonbaseline choices and eight measured baseline retentions for
42 exact contracts, with independent frozen replay and retained binaries.
It improves OA reference plans while same-source ptxas remains faster in every
nominal-selected comparison. No general tensor-dependency scheduler or global
HMMA-gap relaxation was promoted.

| Iteration | Candidate | Result | Production decision |
|---|---|---|---|
| 1 | Exclude generated application output/cache trees from source discovery | Dry-run returns the intended five source files; the previous 141-file contamination disappears | Promoted; this fixes measurement and capture integrity |
| 2 | Early tensor pointer advancement | Complete FP16/BF16 checks and graph checks passed in the retained follow-up; wide shape improved about 7.0–7.6% over legacy and 1.5–2.3% over direct, while tall regressed | Promoted as explicit `early` and narrow `auto`; never global |
| 3 | Feed mined class/pair latency into scalar mined scheduling | 66 scheduler/backend tests passed, but full compile tests produced nvdisasm-invalid SASS on ordinary Python/rowwise kernels | Rejected for production; retained as an experiment hypothesis |
| 4 | Shape-aware tensor policy plus encoder range hardening | `auto` selects early only for the independently validated wide shape; malformed fields now fail closed; 10 targeted encoder/word checks pass | Promoted as guarded policy and correctness hardening |

## Iteration 1: source-tree measurement integrity

`project.walk()` already skipped virtual environments, build directories and
`__pycache__`, but did not skip the repository's `output/` tree or common test
caches. A prior full-suite run failed because `testcases/test_whole_repo/output`
contained captured PTX and reports; the application dry-run reported 141 files
instead of five. `SKIP_DIRS` now includes `output`, `.pytest_cache`,
`.mypy_cache`, `.ruff_cache` and `.tox`. The targeted dry-run and archive tests
pass, and this keeps application source hashes independent of prior captures.

## Iteration 2: early pointer placement

The candidate moves the sixteen independent pointer-advance instructions from
after the tensor compute block to immediately after the next `cp.async` commit.
The async-copy address reads, LDSM address-WAR protection, barriers, HMMA
sequence and instruction multiset remain unchanged. The retained A100 follow-up
reported 84/84 complete checks and 28/28 graph checks across FP16/BF16. Its
positive result is shape dependent, so the CLI exposes:

```sh
openassembler gemm --m 1024 --n 4096 --k 1024 --pointer-advance auto
openassembler gemm --m 16384 --n 4096 --k 4096 --pointer-advance late
```

`auto` selects early only for `1024×4096×1024`; it deliberately resolves to
late for the tall control. The candidate remains separate from the direct
pointer-home choice, Morton traversal and fused epilogues.

## Iteration 3: mined scalar scheduler correction (rejected)

The scalar pipeline still schedules after physical allocation, so this round
does not claim to solve the allocation/ILP problem. The candidate rewired
`_mined_stalls()` to use an exact mined producer/consumer pair when present and
a mined producer minimum otherwise. Its focused 66-test subset passed, but the
complete suite generated SASS that `nvdisasm` rejected for ordinary Python and
row-reduction kernels. The production change was reverted. This is a useful
negative result: the mined table cannot be connected to control generation
without opcode-specific control validation and a broader safe fallback.

## Iteration 4: guarded dispatch and encoder fail-closed behavior

The pointer candidate is now represented in the immutable GEMM plan and tensor
manifest, so recompile preserves the policy. The encoder validates register,
predicate, constant-bank, branch displacement, barrier, control and memory-base
field ranges before setting bits. This addresses the audit's malformed-input
finding: a request such as `MOV R256, R2` must raise `EncodeError`, never silently
become `MOV R0, R2`. Range validation is correctness hardening, not a measured
performance gain.

## Validation

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q \
  tests/test_torchfront.py::test_transformer_repo_dry_run_selects_its_configured_workload \
  tests/test_cli_gemm.py tests/test_gemm_codegen.py tests/test_backend_tensor.py \
  tests/test_fastmm.py tests/test_backend.py tests/test_reduction_codegen.py \
  tests/test_word.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/check_docs.py
```

The targeted compiler/GEMM/scheduler/encoder set passed for these recorded
changes. This ledger is not a current regression result; use the current
[status page](../../docs/STATUS.md) and a fresh test invocation for the latest
checkout. The earlier failure was the source-tree contamination described above.

## Remaining gap

These rounds do not close the ptxas gap. The audit still shows excess OA address
and scalar work, rigid HMMA issue/drain scheduling, reduction contraction gaps,
and state-correctness failures when native pointwise operators are composed in
training. The next safe work is a virtual-register scheduler/allocation
experiment and a fresh A100 holdout for the corrected mined model and the
shape-specific pointer policy. The later exact-shape replay addresses that
pointer-policy admission separately; the virtual-register/general scheduler
work remains open. No global HMMA-gap relaxation is justified.

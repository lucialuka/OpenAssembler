# Explicit bounded decode specialization

Status: implemented

## Gap and selected contract

G39 requires every positive past length to have a precompiled exact variant.
The selected extension keeps that default and introduces explicit host-side
specialization, with a maximum context and a lifetime build-attempt budget.
It compiles an exact length on a declared miss; it does not add padding, empty
caches, a symbolic GPU shape or a vendor fallback.

```python
session = DecodeSession(
    prefill_runtime,
    {},
    specialization={
        "schema": 1,
        "factory": build_exact_variant,
        "artifact_key": "my-model-and-compiler-v1",
        "max_context": 128,
        "max_build_attempts": 32,
    },
)
```

The factory receives `(past_length, contract)` and returns a fresh, unused OA
native `Runtime`. The contract is an independent JSON-compatible copy containing
the artifact key/version, exact past/next lengths, context limit, SM80/math
policy, named non-cache/cache tensor schemas and captured parameter schemas.
It contains no session tensors. The factory owns its host compilation/artifact
lookup policy. It must use the declared model semantics and a stable artifact
namespace; tensor schema equality cannot prove two different models equivalent.
The ordinary runtime validates the returned native bundle and its ownership.

The session checks its producer epoch, weight identities/versions, cache
identities/versions, caller cache-binding prohibition, device/layout/dtype and
known non-cache input schema before invoking the builder. Context overflow
rejects before building or launching, including a precompiled variant. All
factory attempts consume budget, including failures. Reset retains compiled
variants, the fixed decode-input schema and consumed budget; it starts a new
cache sequence. Repeated lengths reuse the admitted runtime without a builder
call. Maximum context counts the complete cache after appending the next token.

After a miss, admission validates architecture/math, named parameter types and
shapes, actual cache non-sequence axes, exact inputs, three distinct outputs,
one-token FP32 cache append and a consistent prediction schema. Only then are
canonical prefill weights shared with the new runtime. The variant and new
cache become owned after successful execution and output metadata checks.
Factory/schema/forward failure preserves the prior session cache and variant
map; a fresh returned runtime is closed on failure. Close deduplicates owned
runtimes and is idempotent. A closed session cannot execute again.

The caller must serialize session use. The factory is trusted host code;
arbitrary effects through its closures, files, threads or external state cannot
be rolled back. The session does not provide general Python transactions,
concurrent request handling, GPU-fault recovery or durable factory-cache storage.
Runtime index-domain checks remain at the ordinary forward boundary; a new
unknown program's index rules cannot be inferred before its factory returns.

## Candidate and validation

The [policy experiment](../../compiler_exp/gap_closure/decode_specialization/candidate-results.json)
first checks exact unseen-length reuse and budget/context behavior independently
of production. The [staged implementation](../../compiler_exp/gap_closure/decode_specialization/candidate-decode.py)
and [baseline](../../compiler_exp/gap_closure/decode_specialization/baseline-decode.py)
retain the next research step before promotion. The
[CPU gate](../../compiler_exp/gap_closure/decode_specialization/candidate-cpu-results.json)
passes 85 specialization and existing model regressions under Python 3.12.10 /
Torch 2.7.1. CPU reference doubles test full cache transitions and failure state;
they do not establish CUDA ownership or attention numerical behavior.

The [initial A100 gate](../../compiler_exp/gap_closure/decode_specialization/candidate-hardware-20261123T084932Z.json)
passes 126 comparisons / 9,520 elements, four independent
storage-view/reuse policies and two changed-input sequences. It records 16
length-3..6 builds and reuse on the second sequence; 40 exact profiler audits
observe 440 native launches and no unknown compute. Builder failure, schema
mismatch, exhausted budget, maximum context and mutated caches reject. A final
run adds poisoned reused workspaces and exact prior-cache/token immutability.
The [final report](../../compiler_exp/gap_closure/decode_specialization/candidate-hardware-20261123T085245Z.json)
passes the same complete domains on A100-SXM4-40GB, Python 3.12.1,
Torch 2.7.1+cu126 and CUDA toolkit 12.8.1. Numerical comparisons retain
`rtol=1e-4, atol=1e-5`; input/cache immutability is exact. The
[archive](../../compiler_exp/gap_closure/decode_specialization/candidate-hardware-20261123T085245Z.zip)
retains full actual/reference tensors, JSON builder requests, all native plans,
PTX, cubins and profiler traces. No worker compute outside the 40 profiled
session invocations is attributed to their launch audits; deliberately invalid
caller mutations are setup for rejection controls.

The [promotion audit](../../compiler_exp/gap_closure/decode_specialization/promotion-audit.json)
checks all complete tensors, deliberately corrupted tensor rejection, source
and archive identities, every profiler launch inventory, and 711 current native
word/register comparisons from 25 bundles (258 unique kernel inputs). All four
backend modules and the uploaded runtime source hashes match. The exact
candidate decode bytes were then promoted to
[production](../../openassembler/modelfront/decode.py). Production imports no
experiment module. The [28 specialized regressions](../../tests/test_decode_specialization.py)
also retain default precompiled behavior through the existing model tests.

```sh
python compiler_exp/gap_closure/decode_specialization/cpu_gate.py
python -m pytest -q tests/test_decode_specialization.py tests/test_model_semantics.py
modal run compiler_exp/gap_closure/decode_specialization/run_modal.py
python compiler_exp/gap_closure/decode_specialization/audit.py compiler_exp/gap_closure/decode_specialization/candidate-hardware-20261123T085245Z.json /tmp/oa-decode-audit.json
```

G39 remains partial beyond explicit bounded specialization. Zero-length prefix,
padding/position rules, beam/speculative decode, paging/in-place cache append,
continuous batching, independent complete serving applications and performance
remain separate contracts under G22/G39–G41. No decoding speedup follows from
these correctness checks.

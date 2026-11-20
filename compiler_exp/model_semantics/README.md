# Model semantics experiment and promotion evidence

Canonical methods, supported contracts, results and limits are in the
[compiler dossier](../../docs/compiler/model-semantics.md). The pre-execution
[protocol](PROTOCOL.md) and [upstream source audit](sources.json) distinguish
implemented candidates from deferred optimization proposals.

## Reproduce

Run from the repository root with Python 3.12+, the CUDA 12.8 toolkit for local
native checks, Torch for CPU tests, and authenticated Modal for A100:

```sh
python -m pytest -q tests/test_model_semantics.py
modal run compiler_exp/model_semantics/run_modal.py
modal run compiler_exp/model_semantics/run_modal.py --production
modal run compiler_exp/model_semantics/run_modal.py --production --action cli
modal run compiler_exp/model_semantics/run_modal.py --production --action tests
python -m exp.native_indexing.audit_dispatch --update
python tools/check_docs.py
```

The default Modal command overlays only the frozen modified Python modules from
`overlay/` onto the current runtime payload. `--production` uses normal source
without an overlay or monkeypatch. Pinned images use Torch 2.7.1/CUDA12.8;
the actual device/version is recorded. Both CLI commands invoke the public
`python -m openassembler` entrypoint and save their artifacts. The fixed repo
fixture is byte-identical to `examples/native_language_model`.

`stage.py`, `stage_runtime.py`, `stage_repository.py`, `stage_fused.py` and
`stage_api.py` preserve the research construction sequence against baseline
`ed0b287`. They are historical staging helpers; use `overlay/` to reproduce the
final candidate, since subsequent validation repairs are included there.
The ignored `candidate/` directory is only a local working copy; it is not
needed by the distributed driver. `promotion.json` records baseline and promoted
file identities, with final validation repairs identified separately.

## Retained gates

| Record | Scope |
|---|---|
| [Split candidate](candidate-20261120T140713Z.json) | 653 A100 full-tensor checks, eight SGD/Adam/AdamW updates and resume, four storage modes and cache chain |
| [Fused candidate](candidate-20261120T141013Z.json) | 671 A100 checks plus paired split/fused optimizer timings |
| [Production hardware](production-hardware-20261120T142112Z.json) | 779 checks / 1,771,766 elements; adds large-offset CE and tiny/zero-gradient epsilon controls |
| [Final production CLI](production-cli-20261120T142424Z.json) | Public compile and saved run, 304 checks each, eight language-block updates, profiler/ownership and no vendor compute |
| [Targeted production CPU](production-cpu-final.log) | 214 local Torch tests before the additional cache-ownership test |
| [Pinned production regression](production-tests-20261120T143351Z.json) | 215 passed on Torch 2.7.1, four pytest mark-registration warnings |
| [Final contract tests](production-contract-tests.log) | 57 tests, including cache/weight/producer guards and atomic invalid-checkpoint rejection |
| [Full no-Torch suite](full-suite-repaired.log) | 1,075 passed, eight optional skips |
| [Binary continuity](../../exp/native_indexing/dispatch-continuity-20261120T142246Z.json) | 1,044 GEMM session comparisons, 40 rowwise comparisons, six fixed goldens; incumbent native text unchanged |

Each hardware JSON points to a SHA-256-bound ZIP. Successful hardware ZIPs retain
full actual/reference tensors as a Torch archive, graph/ledger metadata, PTX and
cubins. The final CLI ZIP retains the saved source/weights/native bundle and reports;
earlier CLI records retained reports only.
Load tensor archives only as data (e.g. `torch.load(...,weights_only=True)`).

Initial CPU package-selection and overlay-fixture failures remain in their logs.
The [failed production expansion](production-hardware-20261120T141927Z.json)
was a harness lookup error (`bias` versus canonical `parameter:bias`), with its
emitted artifacts retained. The original no-Torch collection failure is retained
in `full-suite.log`; the new Torch tests now use the repository's optional-Torch
skip convention. These are not successful correctness gates or numerical
counterexamples. Any new failed mathematical check is saved with complete
actual/reference tensors; changing a tolerance to admit it is prohibited.

The [archive/source audit](audit.json) checks every result archive and compares
current production source with the hardware-frozen overlay after removing
docstrings only. Run `python compiler_exp/model_semantics/audit.py`. The first
pinned transport missed a test helper; the next attempt timed out before worker
execution. The final pinned result above passes; raw failed attempts remain.

The [exported-tree check](exported-docs.json) passes without ignored workspace
files. Its tree identity precedes adding this final validation record, so the
record does not recursively claim to hash itself.

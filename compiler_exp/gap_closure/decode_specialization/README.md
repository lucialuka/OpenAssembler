# Bounded exact-length decode specialization

The [canonical dossier](../../../docs/compiler/gaps-decode.md) defines G39,
factory/schema/budget/context contracts, failure state and remaining limits.
The independent policy candidate precedes the staged implementation; CPU and
A100 gates precede production promotion. Production imports no experiment code.

The [final A100 report](candidate-hardware-20261123T085245Z.json) passes 126
full prediction/cache comparisons (9,520 elements), exact input/old-cache
immutability, poisoned workspace reuse, 16 explicit builds, reuse after reset,
and failure/budget/context/mutation controls. Four view/reuse policies run two
changed-input sequences. Forty exact trace audits observe 440 OA native launches.
The device is A100-SXM4-40GB with Torch 2.7.1+cu126 / Python 3.12.1 / toolkit
12.8.1. These are correctness and ownership checks, with no latency result.

The [read-only audit](promotion-audit.json) checks all archived tensors and
traces, current uploaded runtime source hashes, 25 native bundles and 711 exact
word/register comparisons. An intentionally corrupted tensor rejects. The
[first report](candidate-hardware-20261123T084932Z.json) and its harness remain
retained; the final gate adds poisoning and exact input immutability.

The [supported suite](supported-suite.log) and
[compiler-only suite](compiler-only-suite.log) run after promotion. Their
[validation record](validation.json) identifies environments, source hashes,
docs checks and the separately retained foundation integration. CPU reference
doubles independently check cache/adoption control flow and rejection cases;
they do not substitute for native hardware execution.

```sh
python compiler_exp/gap_closure/decode_specialization/cpu_gate.py
python -m pytest -q tests/test_decode_specialization.py tests/test_model_semantics.py
modal run compiler_exp/gap_closure/decode_specialization/run_modal.py
python compiler_exp/gap_closure/decode_specialization/audit.py compiler_exp/gap_closure/decode_specialization/candidate-hardware-20261123T085245Z.json /tmp/oa-decode-audit.json
```

The Modal driver stages exact retained candidate bytes. They match the promoted
decode source and are checked by the CPU/hardware/audit records. New runs write
dated records and preserve earlier evidence. G39 remains partial beyond this
explicit policy; G40 cache paging and G41 serving remain separate open work.

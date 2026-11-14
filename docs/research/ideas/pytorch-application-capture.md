# PyTorch application capture and saved replay

Status: tested-not-selected

This dossier preserves the historical PyTorch capture/replay experiment.
The current CLI uses the [OA-native model bridge](native-model-bridge.md);
Inductor/Triton execution is disabled. Unused vendor instrumentation and the
Modal adapter were removed on November 14; historical hashes stay in evidence.
The historical implementation captured configured PyTorch workloads and
replayed validated bundles. Its GPU graph lowering uses AOTAutograd,
TorchInductor and Triton; it is not the native OpenAssembler PTX-to-SASS path.

## Idea in plain language

Instead of compiling only a hand-written kernel, observe a bounded model run,
compile the observed forward and backward computation graphs, and save the
result with the source, arguments and runtime cache. A saved application can
then replay those already-captured specializations. This reduces repeated
graph compilation and can fuse multiple operations, while Python still runs
the program and external CUDA libraries still provide some operators.

`openassembler.toml` selects the entry point, arguments, seed, runtime,
backend and numerical gate. Capture executes that workload on a GPU. It does
not statically translate every possible branch of a repository into one binary.

## Contract and observed result

The worker requires PyTorch 2.7.1 and SM80. The pinned Modal image uses CUDA 12.8
and recorded Python 3.11; a compatible local capture records its own exact
versions. Replay must match the saved PyTorch, CUDA, Python and capability.
Native
OpenAssembler compilation on the host separately requires Python 3.12+.
Compilation can run locally with a compatible GPU/runtime or in the pinned
Modal image, or over SSH on an existing compatible Linux A100 host. The same
`--backend user@host` syntax works for compile, run and cnr; `run REPO` selects
its saved application. The [workflow guide](../../REPOSITORY_WORKFLOW.md)
documents required remote runtimes and source/build selection. Dependencies are constrained so they cannot silently replace the
supported PyTorch runtime.

With verification enabled, eager and compiled execution run in separate seeded
processes. The worker observes initial outputs, final parameters, retained
gradients, optimizer tensors and step counts. Output/gradient tolerances and
post-update-state tolerances are separate. Unobserved shapes and branches are
not proven by this comparison. A failed gate prevents artifact publication.

The actual CLI proof completed capture and replay, checked **77 tensors**,
including **15 final gradients**, and captured **35 cubins** labelled
`torch-inductor-triton`. Replay performed five optimizer steps with **zero new
GPU code generation**. Its three timing samples are an integration proof,
not the main whole-transformer performance study.

Replay checks source/cache/cubin identities and requires captured caches.
Uncaptured specializations fail and require a new capture. Archives have size
and file-count bounds; distributed training and unrestricted dynamic workloads
are outside the current contract.
Source/cache archives are limited to 256 MiB and 20,000 files each; tensor
verification snapshots are limited to 128 MiB. These are explicit packaging
limits, not evidence that larger workloads are correct or fast.

## Reproduce

Install the CLI and authenticate the optional Modal client. The retained proof
script creates its isolated application/configuration and calls the real CLI:

```sh
python3.13 exp/whole_stack/cli_proof/prove.py
```

Equivalent explicit command set for that generated application:

```sh
openassembler compile exp/whole_stack/cli_proof/application --frontend torch \
  -o proof_model --verify --no-sass --out-dir output_application \
  --build-backend modal --build-budget 600
openassembler run output_application --application proof_model --backend modal \
  --verify-application --budget 240 --job-timeout 180
```

The configuration is part of the experiment. Do not silently replace its
arguments with a different workload and call the result a reproduction.
For a read-only setup check use `compile ... --verify --dry-run` first.

```sh
.venv/bin/python -m pytest -q tests/test_torchfront.py tests/test_integration.py
python3.13 -m pytest -q tests/test_torchfront_numerics.py
```

The last command needs PyTorch; it can use a separate host environment. Full
options and mutually exclusive application/kernel selectors are listed in
[COMMANDS.md](../../COMMANDS.md).

## Implementation and evidence

- [Frontend/bundle handling](../../../openassembler/torchfront/frontend.py),
  [runtime worker](../../../openassembler/torchfront/worker.py),
  [archive validation](../../../openassembler/torchfront/archive.py).
- [Actual CLI proof driver](../../../exp/whole_stack/cli_proof/prove.py),
  [proof result](../../../exp/whole_stack/cli_proof/summary.json),
  [application configuration](../../../exp/whole_stack/cli_proof/application/openassembler.toml).
- [Whole-transformer evaluation](whole-transformer-evaluation.md) and
  [complete verification diagnostics](state-verification-diagnostics.md).

## Limits and next questions

The optimizer remains shared/eager in the measured application route, cold
compilation is material for short runs, and native `gemm`/`rowwise` kernels are
not automatically inserted into these graphs. Benefits of this route must be
attributed to graph compilation and its libraries, not to a newly optimized
OpenAssembler MMA instruction stream.

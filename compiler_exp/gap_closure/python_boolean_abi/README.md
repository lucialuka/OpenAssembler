# Boolean scalar/array/helper ABI: next G03/G10 cycle

Status: physical parameter/storage ABI remains research; logical helper signatures have a separately validated production extension.

[Boolean values](../../../docs/compiler/gaps-boolean-values.md) currently live in
predicate registers and convert explicitly to the existing numeric types. That
does not provide Boolean scalar parameters, Boolean arrays or annotated Boolean
helper signatures. The [original source](source.py) has three such forms and a
top-level exception, proving discovery must remain static.

The [baseline/toolkit gate](baseline-toolkit.json) records all three original
rejections with file/line diagnostics and compiles an independent original CUDA
control with actual nvcc/ptxas. This toolkit emits `.param.u8` for its Boolean
scalar and `ld.global.u8`/`st.global.u8` for Boolean array storage. Full original
CUDA/PTX/cubin/disassembly/tool logs and hashes remain retained. No GPU execution
or complete ABI conformance is claimed.

The [narrow entry](narrow.ptx) checks the prerequisite parameter boundary. Its
[parser probe](narrow-baseline.json) confirms explicit rejection before native
lowering: current entry ABI admits scalar32/64 parameters. A neighboring-byte
extraction hypothesis therefore is **not an admitted miscompile**. The initial
probe incorrectly assumed narrow-parameter admission; its source/log and the
subsequent parser-admission failure remain retained. The corrected observer
records the earliest actual boundary without claiming numerical execution.

A production solution needs one complete logical/physical contract: Boolean
register values versus byte storage, exact parameter sizes/alignment/offsets,
zero-extension and canonical Boolean representation, typed source metadata,
array/shared/buffer behavior, annotated helper/caller copies and return types,
native ownership and public launch bindings. It must use original toolkit and
independent raw-byte controls with adjacent parameters/guards/tails, then pass
A100, installed/remote and full-suite gates. Existing parser/admission/ownership
checks must not be bypassed by treating a byte as an unvalidated full word.
Frozen pre-ABI modules are retained as `baseline-*.py` for later replay.

```sh
python compiler_exp/gap_closure/python_boolean_abi/replay_baseline.py
python compiler_exp/gap_closure/python_boolean_abi/replay_parameter_boundary.py
```

The subsequent [logical helper cycle](helpers/README.md) admits pure Boolean
helper parameter/return annotations through existing predicate registers and
static expansion. That subcycle has no physical Boolean parameter/storage ABI.
The [frozen original replay](baseline-replay.json) keeps all three historical
rejections reproducible without rewriting `baseline-toolkit.json`; a retained
first replay error omitted the helper snapshot and is fixed by restoring it too.

The [mixed-width layout probe](parameter_layout/README.md) now retains actual
seven-parameter toolkit metadata, an independent natural-layout audit, and explicit
parser/six narrow-form boundaries. It establishes exact byte slot evidence for
the next physical ABI solution; no candidate or GPU execution is claimed.

# Boolean parameter and storage ABI research

Status: physical parameter/storage research remains open; logical helper signatures have a separate validated implementation.

The [Boolean-value extension](gaps-boolean-values.md) admits logical predicate
locals and explicit conversion to existing numeric scalars. G03/G10 still need
Boolean scalar parameters and Boolean array storage. The original study also
rejected annotated Boolean helper signatures; the subsequent [logical helper
extension](gaps-boolean-helpers.md) now implements those without a physical ABI. The [three original rejections](../../compiler_exp/gap_closure/python_boolean_abi/baseline-toolkit.json)
retain their source file/line diagnostics and nonexecuted user module.

An independent [CUDA control](../../compiler_exp/gap_closure/python_boolean_abi/source.cu)
compiled by actual nvcc/ptxas emits a `.param.u8` Boolean scalar and byte global
loads/stores for Boolean arrays. The complete source/PTX/cubin/disassembly/tool
logs and exact identities remain retained. These are this toolkit's observed
forms, without GPU execution or universal ABI claims.

The [narrow-parameter probe](../../compiler_exp/gap_closure/python_boolean_abi/narrow-baseline.json)
confirms that OpenAssembler rejects the byte entry in its parameter parser before
interpretation or native lowering. An initial neighboring-byte extraction
hypothesis assumed admission incorrectly; its source/log remain retained. This
is an unsupported ABI boundary, not an admitted numerical miscompile. The
corrected observer records the actual first gate without weakening it.

The next solution must define logical predicates versus physical byte storage,
parameter sizes/alignment/offsets and extension, representation/canonicalization,
source and launch metadata, arrays/shared/buffers, helper argument/return copies
and complete native ownership. Adjacent parameters, guards/tails, original toolkit
controls and independent raw-byte/A100/installed/remote/full-suite evidence remain
required. Frozen pre-ABI module snapshots support later baseline replay. No
candidate imports or source/admission bypass enter production.

The [experiment dossier](../../compiler_exp/gap_closure/python_boolean_abi/README.md)
retains the complete current boundary and remaining gates. The full 51-gap program
remains incomplete.

The [frozen replay](../../compiler_exp/gap_closure/python_boolean_abi/baseline-replay.json)
reproduces the three original rejections after helper promotion without rewriting
any historical toolkit report. Its first attempt omitted the matching helper
snapshot; the retained failure is corrected by loading all three frozen frontend
modules together. The original byte parser boundary is unchanged.

The subsequent [mixed-width layout probe](../../compiler_exp/gap_closure/python_boolean_abi/parameter_layout/README.md)
retains seven original adjacent scalar parameters and actual ptxas metadata. Its
independent audit matches offsets 0/1/2/4/8/16/24 and a 26-byte bank. Original entry
and six narrow load-form rejections remain explicit; there is no OA candidate,
admitted numerical failure or GPU claim. The next lowering must coordinate both
selectors, sign extension in the PTX interpreter, verification/launch packing,
metadata/source ownership and independently guarded numerical/hardware gates.

The subsequent [narrow-entry implementation](gaps-narrow-parameters.md) now admits
scalar b/u/s8/16 physical entry slots after independent original-PTX/A100 validation.
The [frozen boundary replay](../../compiler_exp/gap_closure/python_boolean_abi/parameter-boundary-replay.json)
keeps both original parser and six load-form rejections reproducible without
rewriting toolkit artifacts. Python `bool` kernel parameters and byte-backed
Array/Buffer/shared Boolean storage still need logical/physical metadata,
canonical representation and their own original-source numerical/launch gates.

The [logical scalar study](../../compiler_exp/gap_closure/python_boolean_abi/scalar/README.md)
now retains three original bool-annotated kernel rejections after raw u8 slots
are admitted. The exact independent CUDA controls instead reach the
[setp.s16 scalar boundary](gaps-narrow-scalars.md). Logical source/host contracts,
raw scalar operation coverage and Boolean storage are distinct remaining gates.

The subsequent [logical Boolean parameter implementation](gaps-boolean-parameters.md)
now joins source bool annotations, predicate decoding, canonical Boolean host
values and exact source/manifest/native type custody. Original-source A100 and
actual public Boolean run-spec gates pass. Boolean array/shared/Buffer byte storage
and actual CUDA narrow register operations remain separate explicit contracts.

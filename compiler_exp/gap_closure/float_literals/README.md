# G11 decimal PTX floating constants

Status: implemented for bounded decimal and raw scalar literal contexts.

The pressure-scheduling FP32 control exposed decimal literals parsed as variable
names. NVIDIA's [PTX floating constant specification](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-constants)
defines their binary64 representation followed by conversion at the instruction
use. The existing parser admits explicit `0f`/`0d` raw-bit forms but recognizes
neither a decimal point nor exponent notation as a floating constant.

`baseline.py` retains the original parser/selector hashes, a same-source ptxas
binary/disassembly and the OA failure. This is a syntax/code-generation
counterexample, with no GPU numerical claim. The pressure-scheduling experiment
uses admitted raw-bit constants and keeps its original failure separately.

`oracle-matrix.json` records 21 toolkit context/spelling controls, with complete
tool diagnostics and accepted binary/disassembly hashes. Fifteen are accepted.
Decimal/exponent, leading/trailing decimal point, signed zero, finite FP32
overflow/subnormal and raw64-to-f32 forms pass. Decimal `mov.f64` and `mov.b64`
pass, while decimal b32/u32/u64 and integer-source `mov.f32` reject. Exponents
that underflow/overflow binary64 also reject. The initial matrix captured only
generic exception text for failures; its script/report/log remain retained,
and the final run captures actual tool stdout/stderr.

`rounding-probe.json` compares an independent rational integer IEEE rounding
algorithm with CPU execution of fifteen accepted ptxas listings, including
exact guard checks. The literal `1.0000000596046448` is a useful counterexample:
direct rational-to-binary32 rounding yields `0x3f800001`, while specified
binary64-then-binary32 conversion and the retained ptxas listing yield
`0x3f800000`. This is CPU/disassembly evidence; it is not a native GPU or
production implementation claim.

The canonical methods, production boundaries and current results are in
[wide scalar and literal semantics](../../../docs/compiler/gaps-wide.md#decimal-and-raw-floating-literal-contexts).
This directory retains the research cycle:

1. `baseline.json` and frozen parser/selector sources reproduce valid decimal
   PTX rejected by OA; ptxas binaries and disassembly remain alongside them.
2. Six oracle matrices retain 110 accepted/rejected context, width, decimal
   range and raw NaN controls with complete tool outputs. Decimal binary64
   subnormals/underflow/overflow reject under the observed CUDA 12.8 toolchain;
   f32 subnormal/overflow results from normal binary64 values remain admitted.
   `oracle-evidence-audit.json` independently checks 368 retained artifacts,
   every source/binary/disassembly/diagnostic hash and current parser acceptance;
   the local ptxas identifies V12.8.61.
3. `literal_helpers.py` and `candidate.py` generate `candidate-parse.py`.
   It parses binary64 first and normalizes source operands contextually. Raw32
   in f64 moves/stores zero-extends the exact source bits. Converted raw64 NaNs
   preserve sign/high payload and become quiet. Wrong integer/bit/destination/
   predicate/pack contexts reject. No selector, encoder or generated spec changes.
4. `candidate-offline.json` checks 5,016 independent rational decimal controls,
   5,008 raw64 controls, seven raw32 controls and the complete oracle matrices.
   `execution-offline.json` passes 360 complete PTX/normal/poisoned-SASS checks
   across five fixtures and four compiler levels. The first execution report
   accidentally reported 54 rather than 18 executions per configuration; its
   source/report/log remain retained and the final inventory is explicitly 360.
5. `candidate-a100-20261123T121453Z.json` passes 320 launches, five fixtures,
   all four OA levels and same-PTX controls, 3×128 threads, lengths 0/1/259/384
   and two seeds. Complete raw outputs/inputs/guards/tails, binaries, manifests
   and disassembly remain in the companion archive. The actual worker is
   A100-SXM4-40GB, driver 13050, Python 3.12.1, CUDA toolkit 12.8.1.
6. `audit.py` captures its source hash at module load, then rechecks every
   retained byte against the independent reference and every executable against
   current native generation. Production audit passes 28,737,536 bytes and
   twenty native comparisons with ownership/frame/register checks. Production
   parser bytes equal the hardware candidate; it imports no experiment code.
7. Normal compiler tests, all 49 policy/98-session continuity and 1,478 earlier
   application native occurrences remain separately recorded. `validation.json`
   binds the final full suites, sources and documentation checks.

The first full supported/compiler-only suites each found the same stale
raw32-to-f64 store rejection; their complete logs and original test source
remain in `initial-suite-failure.json`. The obsolete row now covers unsupported
vector64, while positive literal stores and invalid raw32/b64 width controls
remain explicit. The focused literal/memory rerun passes 184 tests.

The first continuity reader stopped when it encountered the preceding
scheduler's frozen-prototype reports. `first-continuity.py` and its failed log
remain retained. The final reader explicitly verifies that stage's completed
production-audit hashes, final-source admission, exact recorded source repair
and all unchanged backend inputs before comparing current native words. It
does not rewrite numerical reports or treat missing history as a pass.

`expression-baseline.json` starts the next original G11 boundary: typed floating
constant expressions accepted by ptxas still reject in OA. Those observations
are kept separate from the implemented single-literal grammar and are not an
expression implementation or GPU numerical claim.
`expression-semantics.json` independently checks binary64 intermediate rounding
against fourteen normal/poisoned CPU executions of the seven accepted listings.
The cancellation case rounds to zero after binary64 addition, whereas treating
the entire expression as exact arithmetic would incorrectly retain one.

Final integration passes 2,974 supported-environment tests and 2,489 tests with
15 explicit skips in the separate compiler-only environment. Live/exported
documentation and all source/evidence identities are bound by `validation.json`.

The arithmetic execution corpus is exactly representable and proves literal
positions and state preservation; general IEEE operation conformance, floating
constant expressions, wider types/modifiers and complete PTX operand validation
remain original G11 work. No speed or all-gaps-complete claim is made.

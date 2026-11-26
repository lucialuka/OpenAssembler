"""The actual Python-subset -> PTX compiler: ast.parse the kernel's source, walk the tree,
emit PTX text. One Python variable name gets exactly one fixed PTX register for its whole
scope (reused across loop iterations and if/else branches, same discipline as the
hand-written PTX elsewhere in this repo) -- no SSA, no phi nodes, by design: this is a
small enough subset that a classic one-register-per-variable allocator is both correct and
trivial to verify.

STATIC COMPILATION. compile_file(path) never imports or executes the user's file: it
ast-parses it, finds the top-level functions decorated @kernel / @pyfront.kernel /
@openassembler.pyfront.kernel, and resolves every name (annotations, intrinsics, `math.`)
from the file's import statements alone. A repo-wide compile therefore cannot run
arbitrary top-level code (tests/test_pyfront_semantics.py compiles a file whose top level
is `raise SystemExit`). find_kernel() is static too; compile_kernel(fn) accepts either a
live function (it re-parses the function's whole source file, so it sees the same imports)
or find_kernel's result, and emits byte-identical PTX to compile_file for the same kernel.

SEMANTICS CONTRACT. The goal is EXACT Python semantics wherever the GPU allows; every
place it can't is listed here (and nothing else is silently different):

  Integers are i32 (two's complement, wrapping). Python ints are unbounded: any i32
  overflow (x * y past 2**31, abs(-2**31), -(-2**31), -2**31 // -1) wraps on the GPU where
  Python would just grow. Integer literals outside i32 are rejected at compile time.
  i64/u64 explicitly select wrapping 64-bit integers, including scalar parameters,
  local variables, global Array elements and pure device helpers. Add/subtract,
  bitwise and/or/xor/inversion, comparisons and constant shifts 0..63 are exact at
  the declared width. Signed right shifts extend the sign; unsigned shifts fill
  zero. Multiplication keeps the low 64 bits for both types. Unary minus supports
  i64 and wraps at INT64_MIN. Division/remainder and float conversions remain rejected.
  Explicit i32/u32/i64/u64 casts preserve/truncate the low destination bits and
  sign-extend a signed source when widening; integer literals undergo the same
  modulo conversion. Mixed typed operands require explicit casts. Range counters
  and array indices remain i32. These contracts do not adopt Python's unbounded
  integer arithmetic, negative indexing or runtime exceptions.

  Floats are f32, not Python's float64: every operation rounds to f32. A single + - * /
  sqrt on f32 inputs is bit-identical to Python computing in float64 and rounding the
  result to f32 (float64 has > 2*24+2 bits, so the double rounding is innocuous), but a
  CHAIN of operations rounds after every step, which Python does not do.
  Float literals are rounded to the nearest f32.

  a / b   true division, always f32 (i32 operands are converted with cvt.rn.f32 first,
          exact for |x| <= 2**24), emitted as div.rn.f32 -- IEEE correctly rounded, so it
          equals Python's a / b rounded to f32. Division by a constant power of two is
          emitted as the (exactly equal) multiply by its reciprocal. Python raises
          ZeroDivisionError for x / 0; the GPU returns +-inf / NaN (IEEE). A literal
          0 or 0.0 divisor is rejected at compile time.
  a // b, a % b on i32: Python FLOOR semantics (-7 // 2 == -4, -7 % 2 == 1,
          7 % -2 == -1, sign of % follows the divisor). Emitted as PTX's truncating
          div.s32 (+ a remainder) and a branch-free floor correction; a constant divisor
          stays an immediate so the backend's divide-by-constant path applies, and a
          positive power-of-two divisor becomes shr.s32 / and.b32 (which ARE floor
          semantics in two's complement). On u32 floor == truncation, so div.u32/rem.u32
          as-is. Division by zero is UNDEFINED on the GPU (Python raises); -2**31 // -1
          wraps. A literal 0 divisor is rejected at compile time.
  a // b, a % b on f32: rejected. Python's float // is floor((a - a % b) / b) computed
          with an exact fmod and special sign/zero/inf rules (e.g. 1 // 0.1 == 9.0 even
          though 1 / 0.1 == 10.0); PTX has no matching instruction and composing one from
          floor(a / b) would be wrong exactly on those edge cases.
  float(x) on i32: cvt.rn.f32.s32 (round to nearest; exact for |x| <= 2**24, Python's
          float() would be exact up to 2**53). int(x) on f32: cvt.rzi.s32.f32, truncation
          toward zero like Python. Python raises for int(nan) / int(inf) and grows past
          2**31; the GPU saturates (NaN -> 0, out-of-range -> -2**31 / 2**31-1).
          int(u32) is rejected (values >= 2**31 don't fit i32).
  min(a, b), max(a, b): two arguments of the same type. i32/u32 -> min/max PTX
          (identical to Python on ints). f32 -> setp + selp mirroring CPython's exact
          algorithm (min returns b only if b < a, else a), so NaN and -0.0/+0.0 order
          behave exactly like Python -- PTX's min.f32 differs on both (it drops NaNs
          and orders -0 < +0). abs(x): i32 as max(x, -x) (abs(-2**31) wraps), f32 as
          abs.f32 (clears the sign bit; NaN payload may be canonicalized).
  sqrt(x) / math.sqrt(x): sqrt.rn.f32, correctly rounded -- exact. sqrt of a negative
          number returns NaN where Python's math.sqrt raises ValueError; sqrt(-0.0) is -0.0
          in both. Denormal inputs are handled exactly (no flush-to-zero).
  exp, log, tanh (bare, or as math.exp / math.log / math.tanh after `import math`) and
          rsqrt are APPROXIMATE, like CUDA's __expf/__logf/tanhf fast paths:
            exp(x)   = ex2.approx.f32(x * log2(e))   (the x*log2(e) rounding grows the
                       error with |x|: CUDA documents 2 + floor(|1.173 x|) ulp
                       for __expf, which is this exact sequence)
            log(x)   = lg2.approx.f32(x) * ln(2)     (CUDA's __logf: abs error <= 2**-21.41 for x in
                       [0.5, 2], <= 3 ulp elsewhere; log(0) = -inf, log(x<0) = NaN where
                       Python raises)
            tanh(x)  = tanh.approx.f32(x)            (relative error ~2**-11, sm_75+)
            rsqrt(x) = rsqrt.approx.f32(x)           (a few ulp; pyfront-only, math has none)
          These are deliberate: they are what training kernels use (softmax, GELU,
          layernorm); an exactly rounded expf is a long software sequence.
  Comparisons: < <= > >= == on f32 are ordered (false on NaN, as in Python); != is
          true on NaN, as in Python (setp.neu). Chained comparisons are rejected.
  Conditions of if/while: comparisons, and/or (short-circuited into branches), `not`
          (folded into the branch polarity -- no predicate is materialized), a bare
          int/float (Python truthiness: != 0, NaN is true), True/False. `not` anywhere
          else (assigned, nested in arithmetic) is rejected; so are and/or used as values.
  Loops: bounded static numeric tuples (including break/continue/nested/else),
          for v in range(start, stop[, step]) with a constant non-zero step, while cond:,
          break and continue in both, nested to any depth. Bounds are evaluated once
          (Python semantics), the loop variable keeps its last value after the loop, and
          assigning to it inside the body does not change the iteration count (both via
          a hidden counter, used only when the variable is assigned in the body or
          read outside the loop). An i32 counter that would overflow (range near 2**31
          with a large step) wraps on the GPU. for/while ... else are rejected.
  Arrays: Array(f32|i32|u32|i64|u64) parameters (global memory), and shared memory declared at
          kernel top level as  s = shared_array(f32, 256)  (i32/u32/i64/u64 too; constant
          positive size; at most 48 KB in total), read/written as s[i]. Indices are i32.
          Out-of-bounds indices are not checked (Python raises IndexError; the GPU
          reads/writes whatever is there) and NEGATIVE indices do not count from the end
          (a constant negative index is rejected; a runtime one is out of bounds).
          Shared memory starts uninitialized (Python code can't observe that only
          because pyfront kernels can't run as Python).
  syncthreads() -> bar.sync 0. Every thread of the block must reach it, so it is rejected
          at compile time when it is (lexically) under control flow that depends on the
          thread index -- an if/while on tid_x(), a for with a tid-dependent bound, or
          after a tid-dependent return/break -- which would deadlock or be undefined.
          The analysis is conservative: values loaded from arrays at a tid-dependent index
          count as thread-dependent too.
  Device helpers: same-file or statically resolved local-package @device functions
          with explicitly annotated scalar
          parameters and return values are inlined at each call site. Their bodies
          contain scalar assignments, if/elif/else, while/for-range with break/continue,
          and one terminal return expression. Nested pure helpers and numeric
          intrinsics are allowed. Locals must be initialized on every incoming path;
          loops conservatively may execute zero times. Arguments are copied, so helper
          parameter/local assignments cannot change caller variables. No memory effects,
          mutable captures, recursion, early returns or runtime calls are admitted. Explicit
          typing.Final numeric/tuple source values have bounded compile-time semantics. Local-package
          imports use an explicit source root (or the enclosing package), retain exact
          source/dependency/call identities, and execute no user module code.
  Not supported (NotSupported with the line): any other call, lists/tuples/dicts,
          strings, attribute access other than math./pyfront. intrinsics, augmented or
          plain assignment to more than one target, f16/f64, atomics, recursion, nested
          functions, lambdas, try/with, global/nonlocal. Integer bitwise operators are
          supported; ** is not implemented.
"""

from __future__ import annotations

import ast
import inspect
import json
import math
import struct
import textwrap
from dataclasses import dataclass, field
from pathlib import Path

from openassembler.pyfront.types import ArrayType, Scalar

HEADER = ".version 8.7\n.target {arch}\n.address_size 64\n\n"

_SREGS = {
    "tid_x": "%tid.x", "tid_y": "%tid.y", "tid_z": "%tid.z",
    "ctaid_x": "%ctaid.x", "ctaid_y": "%ctaid.y", "ctaid_z": "%ctaid.z",
    "ntid_x": "%ntid.x", "ntid_y": "%ntid.y", "ntid_z": "%ntid.z",
}
_INTRINSICS = _SREGS  # back-compat name
_THREAD_VARYING = {"tid_x", "tid_y", "tid_z"}       # differ between threads of one block
_MATH1 = {"sqrt", "rsqrt", "exp", "log", "tanh"}
_MATH_MODULE = {"sqrt", "exp", "log", "tanh"}       # also accepted as math.<name>
_PF_NAMES = set(_SREGS) | _MATH1 | {"shared_array", "syncthreads", "kernel", "device", "Array", "Buffer", "atomic_add", "atomic_exchange", "atomic_compare_exchange",
                                    "f32", "i32", "u32", "i64", "u64"}
_INT_TYPES = {"i32": "s32", "u32": "u32", "i64": "s64", "u64": "u64"}
_BUILTINS = {"min", "max", "abs", "float", "int", "range", "bool"}

_LOG2E_F32 = 0x3FB8AA3B     # f32(log2(e)) = 1.44269502, the constant __expf multiplies by
_LN2_F32 = 0x3F317218       # f32(ln(2))   = 0.693147182
_SMEM_LIMIT = 48 * 1024     # static .shared limit per block (sm_80, without opt-in)
_I32_MIN, _I32_MAX = -(1 << 31), (1 << 31) - 1


def _f32_bits(v: float) -> int:
    try:
        return struct.unpack("<I", struct.pack("<f", v))[0]
    except OverflowError:   # |v| beyond f32 range rounds to inf, as a C float literal would
        return 0x7F800000 | (0x80000000 if v < 0 else 0)


class NotSupported(Exception):
    """Raised for any Python construct outside the compiled subset, with the source line
    (and, from compile_file, the file) attached -- we never guess at semantics we haven't
    modeled."""

    def __init__(self, msg: str, node: ast.AST | None = None, *, path: str | None = None,
                 lineno: int | None = None):
        self.msg = msg
        self.lineno = lineno if lineno is not None else getattr(node, "lineno", None)
        self.path = path or getattr(node, "_oa_source_path", None)
        super().__init__(self._render())

    def _render(self) -> str:
        if self.lineno is None and self.path is None:
            return self.msg
        parts = [str(self.path)] if self.path is not None else []
        if self.lineno is not None:
            parts.append(f"line {self.lineno}")
        return f"{self.msg} ({', '.join(parts)})"

    def with_path(self, path) -> "NotSupported":
        if self.path is None:
            self.path = str(path)
            self.args = (self._render(),)
        return self


# ---- static name resolution ------------------------------------------------------------------

class _Names:
    """What names in the user's file refer to, read off its import statements only (never
    by executing it). resolve() maps an expression to a canonical string:
      "pf.X"   -- openassembler.pyfront's X (from any of: bare X never rebound by an import,
                  `from openassembler.pyfront import X [as Y]`, `pf.X` after
                  `import openassembler.pyfront as pf`, `openassembler.pyfront.X`, ...)
      "math.X" -- the math module's X (needs an `import math [as m]` / `from math import X`)
      "X"      -- a bare builtin name (min, max, abs, float, int, range) not shadowed
      None     -- anything else."""

    _PF_MODULES = ("openassembler.pyfront", "openassembler.pyfront.types",
                   "openassembler.pyfront.compile")

    def __init__(self, tree: ast.AST):
        self.source_tree = tree
        self.alias: dict[str, str] = {}        # local name -> dotted origin
        self.defined: set[str] = set()         # module-level names bound by def/class/assign
        for n in tree.body if isinstance(tree, ast.Module) else ():
            if isinstance(n, ast.Import):
                for a in n.names:
                    if a.asname:
                        self.alias[a.asname] = a.name
                    else:                       # `import a.b.c` binds `a`
                        root = a.name.split(".")[0]
                        self.alias[root] = root
            elif isinstance(n, ast.ImportFrom) and n.module and not n.level:
                for a in n.names:
                    if a.name != "*":
                        self.alias[a.asname or a.name] = f"{n.module}.{a.name}"
        if isinstance(tree, ast.Module):
            for n in tree.body:
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    self.defined.add(n.name)
                elif isinstance(n, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                    for t in (n.targets if isinstance(n, ast.Assign) else [n.target]):
                        for sub in ast.walk(t):
                            if isinstance(sub, ast.Name):
                                self.defined.add(sub.id)
        from openassembler.pyfront.helpers import DeviceFunctions, _ModuleBindings
        bindings = _ModuleBindings()
        bindings.visit(tree)
        self.defined.update(bindings.counts)
        self.alias = {name: origin for name, origin in self.alias.items() if bindings.counts[name] == 1}
        self.devices = DeviceFunctions(tree, self)

    def _dotted(self, e: ast.expr) -> str | None:
        if isinstance(e, ast.Name):
            if e.id in self.alias:
                return self.alias[e.id]
            return None if e.id in self.defined else "@" + e.id      # "@": unbound bare name
        if isinstance(e, ast.Attribute):
            base = self._dotted(e.value)
            if base is None:
                return None
            return f"{base.lstrip('@')}.{e.attr}"
        return None

    def resolve(self, e: ast.expr) -> str | None:
        d = self._dotted(e)
        if d is None:
            return None
        if d.startswith("@"):                  # bare, not imported, not defined at top level
            name = d[1:]
            if name in _PF_NAMES:              # tid_x() etc. are usable without an import
                return "pf." + name
            return name if name in _BUILTINS else None
        mod, _, attr = d.rpartition(".")
        if mod in self._PF_MODULES or mod in ("pyfront", "openassembler.pyfront.types"):
            return "pf." + attr if attr in _PF_NAMES else None
        if mod == "math" and attr in _MATH_MODULE:
            return "math." + attr
        return None


def _resolve_type(ann: ast.expr, names: _Names, *, allow_boolean: bool = False) -> Scalar | ArrayType:
    """Resolve a parameter annotation (i32, f32, Array(f32), pf.Array(pf.i32), or the same
    as a string under `from __future__ import annotations`) from the AST -- no eval."""
    from openassembler.pyfront import types as T

    if isinstance(ann, ast.Constant) and isinstance(ann.value, str):
        try:
            ann = ast.parse(ann.value, mode="eval").body
        except SyntaxError as e:
            raise NotSupported(f"unparseable annotation {ann.value!r}", ann) from e
    r = names.resolve(ann.func if isinstance(ann, ast.Call) else ann)
    if isinstance(ann, ast.Call) and r == "pf.Buffer":
        from openassembler.pyfront.buffers import resolve_annotation
        return resolve_annotation(ann,names)
    if isinstance(ann, ast.Call) and r == "pf.Array":
        if len(ann.args) != 1 or ann.keywords:
            raise NotSupported("Array(...) takes exactly one element type", ann)
        elem = _resolve_type(ann.args[0], names)
        if not isinstance(elem, Scalar):
            raise NotSupported("Array element type must be f32, i32, u32, i64 or u64", ann)
        return T.Array(elem)
    if allow_boolean and not isinstance(ann, ast.Call) and r == "bool":
        return T.Scalar("pred")
    if not isinstance(ann, ast.Call) and r in ("pf.f32", "pf.i32", "pf.u32", "pf.i64", "pf.u64"):
        return getattr(T, r[3:])
    raise NotSupported(f"unsupported type annotation {ast.unparse(ann)!r} "
                       "(use i32, u32, i64, u64, f32 or Array of those types)", ann)


# ---- barrier uniformity ----------------------------------------------------------------------

class _BarrierCheck:
    """Rejects syncthreads() under thread-dependent control flow (see the module docstring).
    `varying` = names whose value may differ between threads of one block; it is grown to a
    fixpoint because loops carry values backwards. Conservative by construction: anything
    touched by a tid, by an array read at a varying index, or assigned under varying control
    flow is varying."""

    def __init__(self, fdef: ast.FunctionDef, names: _Names):
        self.names, self.fdef = names, fdef
        self.varying: set[str] = set()

    def run(self) -> None:
        while True:
            before = set(self.varying)
            self.errors: list[ast.AST] = []
            self._block(self.fdef.body, False)
            if self.varying == before:
                break
        if self.errors:
            raise NotSupported("syncthreads() is reached under thread-dependent control flow "
                               "(an if/while/for bound or early return/break that depends on "
                               "tid_x()); every thread of the block must reach the barrier, "
                               "otherwise it deadlocks or is undefined", self.errors[0])

    def _vexpr(self, e: ast.AST | None) -> bool:
        if e is None:
            return False
        for n in ast.walk(e):
            if isinstance(n, ast.Name) and n.id in self.varying:
                return True
            if isinstance(n, ast.Call) and self.names.resolve(n.func) in {f"pf.{t}" for t in _THREAD_VARYING | {"atomic_add", "atomic_exchange", "atomic_compare_exchange"}}:
                return True
        return False

    def _assign(self, target: ast.AST, varying: bool) -> None:
        if varying and isinstance(target, ast.Name):
            self.varying.add(target.id)

    def _block(self, body: list[ast.stmt], div: bool) -> tuple[bool, bool]:
        """Returns (divergent exit that leaves the innermost loop: break/continue,
        divergent return). `div` = control flow here already depends on the thread."""
        loop_exit = ret_exit = False
        for st in body:
            d = div or loop_exit or ret_exit
            le, re_ = self._stmt(st, d)
            loop_exit |= le
            ret_exit |= re_
        return loop_exit, ret_exit

    def _stmt(self, st: ast.stmt, div: bool) -> tuple[bool, bool]:
        if isinstance(st, ast.Expr) and isinstance(st.value, ast.Call) \
                and self.names.resolve(st.value.func) == "pf.syncthreads":
            if div:
                self.errors.append(st)
            return False, False
        if isinstance(st, ast.Assign):
            for t in st.targets:
                self._assign(t, div or self._vexpr(st.value))
            return False, False
        if isinstance(st, ast.AugAssign):
            self._assign(st.target, div or self._vexpr(st.value) or self._vexpr(st.target))
            return False, False
        if isinstance(st, ast.If):
            d = div or self._vexpr(st.test)
            a = self._block(st.body, d)
            b = self._block(st.orelse, d)
            return a[0] or b[0], a[1] or b[1]
        if isinstance(st, (ast.While, ast.For)):
            hdr = st.test if isinstance(st, ast.While) else st.iter
            d = div or self._vexpr(hdr)
            if isinstance(st, ast.For):
                self._assign(st.target, d)
            le, re_ = self._block(st.body, d)
            if le or re_:              # some threads left early: later iterations diverge
                self._block(st.body, True)
            if st.orelse:
                a, b = self._block(st.orelse, d or le or re_)
                ret_exit = re_ or b
                return a, ret_exit
            # all threads that entered the loop meet again after it (break/continue don't
            # leak out); a divergent return does: the rest of the function is divergent.
            return False, re_
        if isinstance(st, (ast.Break, ast.Continue)):
            return div, False
        if isinstance(st, ast.Return):
            return False, div
        return False, False


# ---- codegen state ----------------------------------------------------------------------------

@dataclass
class _Reg:
    name: str      # "%r3", "%f2", "%p1", "%rd1"
    kind: str      # "s32" | "u32" | "s64" | "u64" | "f32" | "pred" | "ptr" (ptr = 64-bit global address)


@dataclass
class _Arr:
    space: str            # "global" | "shared"
    elem: str             # "f32" | "s32" | "u32" | "s64" | "u64"
    base: _Reg            # %rd (global, generic->global converted) or %r (shared window offset)
    sym: str | None = None
    length: int | None = None
    view: object = None


@dataclass
class _Scope:
    """One kernel's codegen state: register allocation, labels, output."""

    arch: str
    vars: dict[str, _Reg] = field(default_factory=dict)
    arrays: dict[str, _Arr] = field(default_factory=dict)
    params: list[tuple[str, str]] = field(default_factory=list)  # (ptx_type, name)
    shared: list[tuple[str, str, int]] = field(default_factory=list)  # (sym, ptx type, count)
    smem_bytes: int = 0
    lines: list[str] = field(default_factory=list)
    n_r: int = 0
    n_f: int = 0
    n_p: int = 0
    n_rd: int = 0
    label_n: int = 0
    exit_label: str = "L_exit"
    device_call_sites: int = 0
    wide_integer: bool = False

    def emit(self, text: str) -> None:
        self.lines.append(text)

    def new_label(self, tag: str) -> str:
        self.label_n += 1
        return f"L_{tag}_{self.label_n}"

    def new_reg(self, kind: str) -> _Reg:
        if kind in ("s32", "u32"):
            self.n_r += 1
            return _Reg(f"%r{self.n_r}", kind)
        if kind == "f32":
            self.n_f += 1
            return _Reg(f"%f{self.n_f}", kind)
        if kind == "pred":
            self.n_p += 1
            return _Reg(f"%p{self.n_p}", kind)
        if kind in ("s64", "u64"):
            self.wide_integer = True
        if kind in ("ptr", "s64", "u64"):
            self.n_rd += 1
            return _Reg(f"%rd{self.n_rd}", kind)
        raise ValueError(kind)

    def var_reg(self, name: str, kind: str, node: ast.AST) -> _Reg:
        """First assignment to `name` allocates its register; later ones must agree on type
        -- reassigning a scalar's type is not supported (it would need a second physical
        register, defeating the point of reusing one name across branches/iterations)."""
        if name in self.arrays:
            raise NotSupported(f"{name!r} is an array; it can't be reassigned", node)
        if name in self.vars:
            r = self.vars[name]
            if r.kind != kind:
                raise NotSupported(f"variable {name!r} reassigned with a different type "
                                   f"({r.kind} -> {kind})", node)
            return r
        r = self.new_reg(kind)
        self.vars[name] = r
        return r


def _const_value(node: ast.expr):
    """int/float value of a literal, folding unary +/- (so `-2` is a constant); else None."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) \
            and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        v = _const_value(node.operand)
        if v is not None:
            return -v if isinstance(node.op, ast.USub) else v
    return None


def _is_pow2(v: int) -> bool:
    return v > 0 and v & (v - 1) == 0


class _Emitter:
    """One pass over the kernel body; every statement either emits PTX or raises
    NotSupported. Expressions return a _Reg holding the value (a fresh scratch register
    unless the expression is a bare variable read, which returns its home register)."""

    def __init__(self, scope: _Scope, names: _Names, fdef: ast.FunctionDef):
        self.s, self.names, self.fdef = scope, names, fdef
        self.loops: list[tuple[str, str]] = []      # (continue label, break label)
        self.depth = 0                              # >0: inside if/loop (not kernel top level)
        self.local_names = {a.arg for a in fdef.args.args} | {
            n.id for n in ast.walk(fdef)
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}

    # ---- statements ----

    def body(self, stmts: list[ast.stmt]) -> None:
        for st in stmts:
            self.stmt(st)

    def stmt(self, node: ast.stmt) -> None:
        fn = getattr(self, f"visit_{type(node).__name__}", None)
        if fn is None:
            what = {"FunctionDef": "nested function definitions", "AsyncFunctionDef":
                    "nested function definitions", "Lambda": "lambdas"}.get(type(node).__name__)
            raise NotSupported(f"{what} are not supported" if what else
                               f"unsupported statement: {type(node).__name__}", node)
        fn(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        if len(node.targets) != 1:
            raise NotSupported("multiple assignment targets", node)
        target = node.targets[0]
        if isinstance(target, ast.Subscript):
            self._store(target, *self._eval(node.value))
            return
        if not isinstance(target, ast.Name):
            raise NotSupported("assignment target must be a name or a[i]", node)
        if isinstance(node.value, ast.Call) and self.names.resolve(node.value.func) == "pf.shared_array":
            self._declare_shared(target, node.value, node)
            return
        val, kind = self._eval(node.value)
        dst = self.s.var_reg(target.id, kind, node)
        self._mov(dst, val, kind)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        if isinstance(node.target, ast.Subscript):
            if isinstance(node.target.value,ast.Name) and getattr(self.s.arrays.get(node.target.value.id),"view",None) is not None:
                raise NotSupported("Buffer augmented stores require an explicit checked load and assignment",node)
            arr, addr = self._address(node.target)
            cur = self._load(arr, addr)
            res, kind = self._arith(node.op, (cur, arr.elem), node.value, node)
            self._store_at(arr, addr, res, kind, node.target)
            return
        if not isinstance(node.target, ast.Name):
            raise NotSupported("augmented assignment target must be a name or a[i]", node)
        cur = self.s.vars.get(node.target.id)
        if cur is None:
            raise NotSupported(f"{node.target.id!r} used before assignment", node)
        res, kind = self._arith(node.op, (cur, cur.kind), node.value, node, into=cur)
        if kind != cur.kind:
            raise NotSupported(f"variable {node.target.id!r} reassigned with a different type "
                               f"({cur.kind} -> {kind})", node)
        self._mov(cur, res, kind)

    def visit_If(self, node: ast.If) -> None:
        # Conditions are compiled straight into branches (and/or short-circuit, `not` flips
        # the polarity): no combined predicate is ever materialized, because our backend's
        # selector has no and.pred/or.pred/not.pred (openassembler.backend.ra.Selector).
        end_label = self.s.new_label("endif")
        else_label = self.s.new_label("else") if node.orelse else end_label
        self._jump(node.test, else_label, when=False)
        self.depth += 1
        self.body(node.body)
        if node.orelse:
            self.s.emit(f" bra {end_label};")
            self.s.emit(f"{else_label}:")
            self.body(node.orelse)
        self.depth -= 1
        self.s.emit(f"{end_label}:")

    def visit_While(self, node: ast.While) -> None:
        top, end = self.s.new_label("while"), self.s.new_label("wend")
        exhausted = self.s.new_label("whileelse") if node.orelse else end
        self.s.emit(f"{top}:")
        self._jump(node.test, exhausted, when=False)
        self.loops.append((top, end))
        self.depth += 1
        self.body(node.body)
        self.depth -= 1
        self.loops.pop()
        self.s.emit(f" bra {top};")
        if node.orelse:
            self.s.emit(f"{exhausted}:")
            self.body(node.orelse)
        self.s.emit(f"{end}:")

    def _static_tuple_loop(self, node: ast.For) -> None:
        """Finite scalar tuple iteration with real loop exits and target homes."""
        if not isinstance(node.target, ast.Name):
            raise NotSupported('static tuple loop target must be a scalar name', node.target)
        values = node._oa_tuple_values
        if not values:
            # No target binding or body effects occur for an empty tuple.
            self.body(node.orelse)
            return
        self.s.static_tuple_values = getattr(self.s, 'static_tuple_values', 0) + len(values)
        if self.s.static_tuple_values > 1024:
            raise NotSupported('static tuple loops exceed 1024 selected values per kernel', node)
        if any(type(value) not in (int, float) for value in values):
            raise NotSupported('static tuple iteration requires numeric scalar elements', node.iter)
        prior = self.s.vars.get(node.target.id)
        if prior is not None:
            kind = prior.kind
        elif any(type(value) is float for value in values):
            kind = 'f32'
        elif all(_I32_MIN <= value <= _I32_MAX for value in values):
            kind = 's32'
        elif all(-(1 << 63) <= value < (1 << 63) for value in values):
            kind = 's64'
        elif all(0 <= value < (1 << 64) for value in values):
            kind = 'u64'
        else:
            raise NotSupported('static tuple elements do not fit a single device integer type', node.iter)
        if kind not in ('s32', 'u32', 's64', 'u64', 'f32'):
            raise NotSupported('static tuple target requires a scalar numeric type', node.target)
        if kind != 'f32' and any(type(value) is not int for value in values):
            raise NotSupported('static tuple element does not fit the existing target type', node.iter)
        target = self.s.var_reg(node.target.id, kind, node.target)
        ctr = self.s.new_reg('s32')
        self.s.emit(f' mov.u32 {ctr.name}, 0;')
        top, body, cont, exhausted, end = [self.s.new_label(tag) for tag in
                                          ('tuple', 'tuplebody', 'tuplenext', 'tupleelse', 'tupleend')]
        self.s.emit(f'{top}:')
        pred = self.s.new_reg('pred')
        self.s.emit(f' setp.ge.s32 {pred.name}, {ctr.name}, {len(values)};')
        self.s.emit(f' @{pred.name} bra {exhausted};')
        for index, value in enumerate(values):
            next_choice = self.s.new_label('tuplechoice') if index + 1 < len(values) else None
            if next_choice:
                check = self.s.new_reg('pred')
                self.s.emit(f' setp.ne.s32 {check.name}, {ctr.name}, {index};')
                self.s.emit(f' @{check.name} bra {next_choice};')
            literal, actual_kind = self._lit_as(value, kind, node.iter)
            self._mov(target, literal, actual_kind)
            if next_choice:
                self.s.emit(f' bra {body};')
                self.s.emit(f'{next_choice}:')
        self.s.emit(f'{body}:')
        self.loops.append((cont, end))
        self.depth += 1
        self.body(node.body)
        self.depth -= 1
        self.loops.pop()
        self.s.emit(f'{cont}:')
        self.s.emit(f' add.s32 {ctr.name}, {ctr.name}, 1;')
        self.s.emit(f' bra {top};')
        self.s.emit(f'{exhausted}:')
        self.body(node.orelse)
        self.s.emit(f'{end}:')

    def visit_For(self, node: ast.For) -> None:
        if hasattr(node, '_oa_tuple_values'):
            return self._static_tuple_loop(node)
        # Existing runtime range-loop contract.
        if not (isinstance(node.target, ast.Name) and isinstance(node.iter, ast.Call)
                and self.names.resolve(node.iter.func) == "range" and not node.iter.keywords):
            raise NotSupported("only 'for i in range(...)' loops are supported", node)
        args = node.iter.args
        if len(args) == 1:
            start_e, stop_e, step = ast.Constant(value=0), args[0], 1
        elif len(args) in (2, 3):
            start_e, stop_e, step = args[0], args[1], 1
            if len(args) == 3:
                step = _const_value(args[2])
                if not isinstance(step, int):
                    raise NotSupported("range() step must be a constant int", node)
        else:
            raise NotSupported("range() takes 1-3 args", node)
        step = int(step)  # Python permits a compile-time True step as integer one.
        if step == 0:
            raise NotSupported("range() step must not be 0", node)
        if not _I32_MIN <= step <= _I32_MAX:
            raise NotSupported("range() step does not fit in i32", node)
        name = node.target.id

        # Python evaluates range()'s bounds once, then binds the variable on each iteration.
        start_v, start_k = self._eval(start_e)
        stop_v, stop_k = self._eval(stop_e)
        if start_k != "s32" or stop_k != "s32":
            raise NotSupported("range() bounds must be int32 expressions", node.iter)
        var = self.s.var_reg(name, "s32", node)
        # The variable doubles as the counter unless that would be observable: if the body
        # assigns it (Python: next iteration rebinds it anyway, the trip count is unchanged)
        # or it is read outside the loop (Python: it keeps the LAST value taken, not `stop`).
        inside = {id(n) for part in [node.target, *node.body] for n in ast.walk(part)}
        stored_in_body = any(isinstance(n, ast.Name) and n.id == name and isinstance(n.ctx, ast.Store)
                             for part in node.body for n in ast.walk(part))
        used_outside = any(isinstance(n, ast.Name) and n.id == name and id(n) not in inside
                           for n in ast.walk(self.fdef))
        ctr = self.s.new_reg("s32") if (stored_in_body or used_outside) else var
        self._mov(ctr, start_v, "s32")
        stop_reg = self.s.new_reg("s32")
        self.s.emit(f" mov.u32 {stop_reg.name}, {stop_v.name};")   # snapshot, never aliased

        top, cont, end = self.s.new_label("loop"), self.s.new_label("next"), self.s.new_label("loopend")
        exhausted = self.s.new_label("forelse") if node.orelse else end
        self.s.emit(f"{top}:")
        pred = self.s.new_reg("pred")
        self.s.emit(f" setp.{'ge' if step > 0 else 'le'}.s32 {pred.name}, {ctr.name}, {stop_reg.name};")
        self.s.emit(f" @{pred.name} bra {exhausted};")
        if ctr is not var:
            self.s.emit(f" mov.u32 {var.name}, {ctr.name};")
        self.loops.append((cont, end))
        self.depth += 1
        self.body(node.body)
        self.depth -= 1
        self.loops.pop()
        self.s.emit(f"{cont}:")
        if abs(step) > 1:
            # With i32 start/stop, a next value outside i32 has crossed stop.
            # Exhaust before the increment can wrap, preserving the last target.
            overflow = self.s.new_reg("pred")
            relation, limit = ("gt", _I32_MAX - step) if step > 0 else ("lt", _I32_MIN - step)
            self.s.emit(f" setp.{relation}.s32 {overflow.name}, {ctr.name}, {limit};")
            self.s.emit(f" @{overflow.name} bra {exhausted};")
        if step > 0:
            self.s.emit(f" add.s32 {ctr.name}, {ctr.name}, {step};")
        else:
            self.s.emit(f" sub.s32 {ctr.name}, {ctr.name}, {-step};")
        self.s.emit(f" bra {top};")
        if node.orelse:
            self.s.emit(f"{exhausted}:")
            self.body(node.orelse)
        self.s.emit(f"{end}:")

    def visit_Break(self, node: ast.Break) -> None:
        if not self.loops:
            raise NotSupported("'break' outside a loop", node)
        self.s.emit(f" bra {self.loops[-1][1]};")

    def visit_Continue(self, node: ast.Continue) -> None:
        if not self.loops:
            raise NotSupported("'continue' outside a loop", node)
        self.s.emit(f" bra {self.loops[-1][0]};")

    def visit_Expr(self, node: ast.Expr) -> None:
        v = node.value
        if isinstance(v, ast.Constant) and isinstance(v.value, str):
            return                                   # docstring / bare string: no effect
        if isinstance(v, ast.Call) and self.names.resolve(v.func) == "pf.syncthreads":
            if v.args or v.keywords:
                raise NotSupported("syncthreads() takes no arguments", node)
            self.s.emit(" bar.sync 0;")
            return
        if isinstance(v,ast.Call) and self.names.resolve(v.func) in {"pf.atomic_add", "pf.atomic_exchange", "pf.atomic_compare_exchange"}:
            self._call(v)
            return
        raise NotSupported("expression statement with no supported effect", node)

    def visit_Pass(self, node: ast.Pass) -> None:
        pass

    def visit_Return(self, node: ast.Return) -> None:
        context = getattr(self, 'helper_return', None)
        if context is not None:
            target, wanted, end = context
            if node.value is None:
                raise NotSupported('scalar helper return requires a value', node)
            value, kind = self._eval(node.value)
            if kind != wanted:
                raise NotSupported(f'device helper returns {kind}, annotated {wanted}', node)
            self._mov(target, value, kind)
            self.s.emit(f' bra {end};')
            return
        if node.value is not None:
            raise NotSupported("kernels don't return a value (write to an output array)", node)
        # A bare `return` is the standard CUDA early-out (`if i >= n: return`).
        self.s.emit(f" bra {self.s.exit_label};")

    def _declare_shared(self, target: ast.Name, call: ast.Call, node: ast.AST) -> None:
        if self.depth or self.loops:
            raise NotSupported("shared_array() must be assigned at the kernel's top level, "
                               "not inside an if or a loop", node)
        name = target.id
        if name in self.s.arrays or name in self.s.vars:
            raise NotSupported(f"{name!r} is already defined; a shared_array needs its own name", node)
        if len(call.args) != 2 or call.keywords:
            raise NotSupported("shared_array(type, count) takes exactly two arguments", node)
        elem = _resolve_type(call.args[0], self.names)
        count = _const_value(call.args[1])
        if not isinstance(elem, Scalar):
            raise NotSupported("shared_array element type must be f32, i32, u32, i64 or u64", node)
        if type(count) is not int or count <= 0:
            raise NotSupported("shared_array count must be a positive integer literal", node)
        element_bytes = 8 if elem.ptx in ("s64", "u64") else 4
        # Match declaration order and natural alignment, including inter-array
        # padding, when enforcing the existing per-block static storage limit.
        self.s.smem_bytes = ((self.s.smem_bytes + element_bytes - 1)
                             // element_bytes * element_bytes + element_bytes * count)
        if self.s.smem_bytes > _SMEM_LIMIT:
            raise NotSupported(f"shared_array total {self.s.smem_bytes} bytes exceeds the "
                               f"{_SMEM_LIMIT}-byte static shared-memory limit", node)
        sym = f"__smem_{name}"
        reg = self.s.new_reg("u32")
        self.s.emit(f" mov.u32 {reg.name}, {sym};")         # 32-bit shared-window address
        self.s.shared.append((sym, elem.ptx, count))
        self.s.arrays[name] = _Arr("shared", elem.ptx, reg, sym, count)

    def _jump(self, test: ast.expr, label: str, when: bool) -> None:
        """Branch to `label` iff bool(test) == when; otherwise fall through."""
        if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
            return self._jump(test.operand, label, not when)
        if isinstance(test, ast.BoolOp):
            if isinstance(test.op, ast.And) != when:
                # (and, jump if false) / (or, jump if true): the first deciding operand jumps
                for v in test.values:
                    self._jump(v, label, when)
            else:
                # (and, jump if true) / (or, jump if false): every operand is needed
                skip = self.s.new_label("sc")
                for v in test.values[:-1]:
                    self._jump(v, skip, not when)
                self._jump(test.values[-1], label, when)
                self.s.emit(f"{skip}:")
            return
        if isinstance(test, ast.Constant) and isinstance(test.value, (bool, int, float)):
            if bool(test.value) == when:
                self.s.emit(f" bra {label};")
            return
        if isinstance(test, ast.Compare) and len(test.ops) > 1:
            return self._chain_jump(test, label, when)
        if isinstance(test, ast.Compare):
            pred, inverted = self._compare(test, branch=True)
            when = when != inverted
        else:
            v, k = self._eval(test)
            if k == "pred":
                pred = v
            else:                              # Python truthiness: x != 0 (NaN is truthy)
                pred = self.s.new_reg("pred")
                if k in ("s64", "u64"):
                    zero, _ = self._lit_as(0, k, test)
                    pred = self._wide_compare(v, zero, k, "eq")
                else:
                    zero = "0f00000000" if k == "f32" else "0"
                    self.s.emit(f" setp.eq.{k} {pred.name}, {v.name}, {zero};")
                when = not when
        self.s.emit(f" @{'' if when else '!'}{pred.name} bra {label};")

    def _chain_jump(self, test: ast.Compare, label: str, when: bool) -> None:
        """Evaluate each operand once and later operands only on a live path."""
        graph = getattr(self.names, 'graph', None)
        if graph is not None:
            graph.note_comparison(self.names, self.fdef, test)
        failed = self.s.new_label("chainfalse") if when else label
        left = test.left
        for operator, right in zip(test.ops, test.comparators):
            a, ka, b, kb = self._pair(left, right, test)
            saved = self.s.new_reg(kb)
            self._mov(saved, b, kb)
            pred, inverted = self._compare_values(a, ka, saved, kb, operator, test, branch=True)
            self.s.emit(f" @{'' if inverted else '!'}{pred.name} bra {failed};")
            # A pure literal keeps the existing adjacent comparison's contextual
            # type rule. An expression keeps its one evaluated scalar snapshot.
            left = right if _const_value(right) is not None else (saved, kb)
        if when:
            self.s.emit(f" bra {label};")
            self.s.emit(f"{failed}:")

    def _boolean_value(self, value, kind):
        if kind == "pred":
            return value
        if kind in ("s64", "u64"):
            zero, _ = self._lit_as(0, kind, self.fdef)
            equal = self._wide_compare(value, zero, kind, "eq")
        else:
            equal = self.s.new_reg("pred")
            zero = "0f00000000" if kind == "f32" else "0"
            self.s.emit(f" setp.eq.{kind} {equal.name}, {value.name}, {zero};")
        result = self.s.new_reg("pred")
        self.s.emit(f" not.pred {result.name}, {equal.name};")
        return result

    def _boolean_expression(self, node):
        result, done = self.s.new_reg("s32"), self.s.new_label("boolean")
        self.s.emit(f" mov.s32 {result.name}, 0;")
        self._jump(node, done, when=False)
        self.s.emit(f" mov.s32 {result.name}, 1;")
        self.s.emit(f"{done}:")
        return self._boolean_value(result, "s32")

    def _boolean_operands(self, node):
        # Compile expression fragments to learn their fixed kinds without running
        # user code. Emit each fragment only on its original live runtime path.
        operands, kinds = [], set()
        for operand in node.values:
            literal = _const_value(operand)
            if literal is not None:
                operands.append((operand, literal, None, None))
                continue
            enclosing = self.s.lines
            self.s.lines = []
            try:
                value, kind = self._eval(operand)
                fragment = self.s.lines
            finally:
                self.s.lines = enclosing
            kinds.add(kind)
            operands.append((operand, None, (value, kind), fragment))
        if len(kinds) > 1 or (kinds == {"pred"} and any(item[1] is not None for item in operands)):
            raise NotSupported("and/or operands require one fixed scalar kind", node)
        kind = next(iter(kinds)) if kinds else ("f32" if any(type(item[1]) is float for item in operands) else "s32")
        result, done = self.s.new_reg(kind), self.s.new_label("booloperands")
        for index, (operand, literal, compiled, fragment) in enumerate(operands):
            if index:
                truth = self._boolean_value(result, kind)
                self.s.emit(f" @{'' if isinstance(node.op, ast.Or) else '!'}{truth.name} bra {done};")
            if compiled is None:
                value, _ = self._lit_as(literal, kind, operand)
            else:
                value, _ = compiled
                self.s.lines.extend(fragment)
            self._mov(result, value, kind)
        self.s.emit(f"{done}:")
        self._note_value(node, kind)
        return result, kind

    def _note_value(self, node, kind):
        graph = getattr(self.names, 'graph', None)
        if graph is not None:
            graph.note_boolean(self.names, self.fdef, node, kind)

    def _mov(self, dst: _Reg, src: _Reg, kind: str) -> None:
        if dst.name == src.name:
            return
        op = {"f32": "mov.f32", "pred": "mov.pred", "s64": "mov.b64", "u64": "mov.b64"}.get(kind, "mov.u32")
        self.s.emit(f" {op} {dst.name}, {src.name};")

    # ---- expressions ----

    def _ev(self, x) -> tuple[_Reg, str]:
        return x if isinstance(x, tuple) else self._eval(x)

    def _eval(self, node: ast.expr) -> tuple[_Reg, str]:
        cv = _const_value(node)
        if cv is not None:
            return self._lit_as(cv, "f32" if isinstance(cv, float) else "s32", node)
        if isinstance(node, ast.Name):
            if node.id in self.s.arrays:
                raise NotSupported(f"array {node.id!r} used as a value (index it: {node.id}[i])", node)
            if node.id not in self.s.vars:
                raise NotSupported(f"{node.id!r} used before assignment", node)
            r = self.s.vars[node.id]
            return r, r.kind
        if isinstance(node, ast.BinOp):
            return self._arith(node.op, node.left, node.right, node)
        if isinstance(node, ast.Compare) and len(node.ops) > 1:
            result = self._boolean_expression(node)
            self._note_value(node, "pred")
            return result, "pred"
        if isinstance(node, ast.Compare):
            pred, _ = self._compare(node, branch=False)
            return pred, "pred"
        if isinstance(node, ast.BoolOp):
            return self._boolean_operands(node)
        if isinstance(node, ast.UnaryOp):
            return self._unaryop(node)
        if isinstance(node, ast.Call):
            return self._call(node)
        if isinstance(node, ast.Subscript):
            if isinstance(node.value,ast.Name) and getattr(self.s.arrays.get(node.value.id),"view",None) is not None:
                from openassembler.pyfront.buffers import access
                return access(self,node)
            arr, addr = self._address(node)
            return self._load(arr, addr), arr.elem
        if isinstance(node, ast.Constant) and isinstance(node.value, bool):
            value, kind = self._lit_as(int(node.value), "s32", node)
            self._note_value(node, "pred")
            return self._boolean_value(value, kind), "pred"
        raise NotSupported(f"unsupported expression: {type(node).__name__}", node)

    def _lit_as(self, v, want: str, node: ast.AST) -> tuple[_Reg, str]:
        """Materialize literal v. An int literal next to an f32 operand becomes an f32
        constant (Python would promote it), but only when that conversion is exact."""
        if want in ("s64", "u64"):
            low, high = (-(1 << 63), (1 << 63) - 1) if want == "s64" else (0, (1 << 64) - 1)
            if type(v) is not int or not low <= v <= high:
                raise NotSupported(f"literal {v!r} does not fit in {want}", node)
            r = self.s.new_reg(want)
            self.s.emit(f" mov.{want} {r.name}, {v};")
            return r, want
        if isinstance(v, int):
            if want == "f32":
                if _f32_bits(float(v)) != _f32_bits(float(_f32(float(v)))) or _f32(float(v)) != v:
                    raise NotSupported(f"int literal {v} is not exactly representable as f32", node)
                v = float(v)
            elif want == "u32":
                if not 0 <= v < (1 << 32):
                    raise NotSupported(f"literal {v} does not fit in u32", node)
                r = self.s.new_reg("u32")
                self.s.emit(f" mov.u32 {r.name}, {v};")
                return r, "u32"
            else:
                if not _I32_MIN <= v <= _I32_MAX:
                    raise NotSupported(f"int literal {v} does not fit in i32 (kernels compute in "
                                       "32-bit integers)", node)
                r = self.s.new_reg("s32")
                self.s.emit(f" mov.s32 {r.name}, {v};")
                return r, "s32"
        r = self.s.new_reg("f32")
        self.s.emit(f" mov.f32 {r.name}, 0f{_f32_bits(v):08X};")
        return r, "f32"

    def _pair(self, left, right, node):
        """Evaluate two operands; a literal takes the other side's type (see _lit_as)."""
        lit = lambda x: None if isinstance(x, tuple) else _const_value(x)  # noqa: E731
        lv, rv = lit(left), lit(right)
        if lv is not None and rv is not None:
            k = "f32" if isinstance(lv, float) or isinstance(rv, float) else "s32"
            return (*self._lit_as(lv, k, left), *self._lit_as(rv, k, right))
        if lv is not None:
            b, kb = self._ev(right)
            return (*self._lit_as(lv, kb, left), b, kb)
        a, ka = self._ev(left)
        if rv is not None:
            return (a, ka, *self._lit_as(rv, ka, right))
        return (a, ka, *self._ev(right))

    def _mixed(self, ka, kb, node):
        raise NotSupported(f"mixed-type operands ({ka} vs {kb}); convert explicitly with "
                           "float(x), int(x), or typed i32/u32/i64/u64 casts", node)

    def _arith(self, op: ast.operator, left, right, node: ast.AST,
               into: _Reg | None = None) -> tuple[_Reg, str]:
        """left/right: AST nodes or already-evaluated (reg, kind). `into`: write a one-
        instruction result straight there (`s += x` stays a single add, as before)."""
        if isinstance(op, (ast.BitAnd, ast.BitOr, ast.BitXor, ast.LShift, ast.RShift)):
            return self._integer_bits(op, left, right, node, into=into)
        if isinstance(op, ast.Div):
            return self._truediv(left, right, node)
        if isinstance(op, (ast.FloorDiv, ast.Mod)):
            return self._floordiv(op, left, right, node)
        mnem = {ast.Add: "add", ast.Sub: "sub", ast.Mult: "mul"}.get(type(op))
        if mnem is None:
            raise NotSupported(f"operator {type(op).__name__} is not supported", node)
        a, ka, b, kb = self._pair(left, right, node)
        if ka != kb:
            self._mixed(ka, kb, node)
        if ka == "pred":
            raise NotSupported("arithmetic on a comparison result", node)
        dst = into if into is not None and into.kind == ka else self.s.new_reg(ka)
        if ka == "f32":
            self.s.emit(f" {mnem}.f32 {dst.name}, {a.name}, {b.name};")
        else:
            self.s.emit(f" {mnem}{'.lo' if mnem == 'mul' else ''}.{ka if ka in ('s64','u64') else 's32'} {dst.name}, {a.name}, {b.name};")
        return dst, ka

    def _integer_bits(self, op, left, right, node, into=None):
        if isinstance(op, (ast.LShift, ast.RShift)):
            value, kind = self._ev(left)
            bits = 64 if kind in ("s64", "u64") else 32
            count = None if isinstance(right, tuple) else _const_value(right)
            if kind not in ("s32", "u32", "s64", "u64") or type(count) is not int or not 0 <= count < bits:
                raise NotSupported(f"integer shifts require a constant count in [0, {bits-1}]", node)
            result = into if into is not None and into.kind == kind else self.s.new_reg(kind)
            mnemonic = "shl.b" + str(bits) if isinstance(op, ast.LShift) else "shr." + kind
            self.s.emit(f" {mnemonic} {result.name}, {value.name}, {count};")
            return result, kind
        a, ka, b, kb = self._pair(left, right, node)
        if ka != kb or ka not in ("s32", "u32", "s64", "u64"):
            raise NotSupported("bitwise operands require matching integer types", node)
        result = into if into is not None and into.kind == ka else self.s.new_reg(ka)
        mnemonic = {ast.BitAnd:"and", ast.BitOr:"or", ast.BitXor:"xor"}[type(op)]
        self.s.emit(f" {mnemonic}.b{64 if ka.endswith('64') else 32} {result.name}, {a.name}, {b.name};")
        return result, ka

    def _integer_cast(self, want, arg, node):
        literal = _const_value(arg)
        if literal is not None:
            if type(literal) is not int:
                raise NotSupported("typed integer casts require integer values, not floats", node)
            bits = 64 if want.endswith("64") else 32
            literal &= (1 << bits) - 1
            if want.startswith("s") and literal >= (1 << (bits - 1)):
                literal -= 1 << bits
            return self._lit_as(literal, want, arg)
        value, kind = self._eval(arg)
        if kind == "pred":
            converted = self.s.new_reg("s32")
            self.s.emit(f" selp.s32 {converted.name}, 1, 0, {value.name};")
            value, kind = converted, "s32"
        if kind not in ("s32", "u32", "s64", "u64"):
            raise NotSupported("typed integer casts require integer or Boolean values", node)
        result = self.s.new_reg(want)
        self.s.emit(f" cvt.{want}.{kind} {result.name}, {value.name};")
        return result, want

    def _wide_compare(self, a, b, kind, comparison):
        result = self.s.new_reg("pred")
        self.s.emit(f" setp.{comparison}.{kind} {result.name}, {a.name}, {b.name};")
        return result

    def _to_f32(self, x, node) -> _Reg:
        v = None if isinstance(x, tuple) else _const_value(x)
        if v is not None:
            return self._lit_as(v, "f32", node)[0]
        r, k = self._ev(x)
        if k == "f32":
            return r
        if k not in ("s32", "u32"):
            raise NotSupported(f"can't convert {k} to f32", node)
        d = self.s.new_reg("f32")
        self.s.emit(f" cvt.rn.f32.{k} {d.name}, {r.name};")
        return d

    def _truediv(self, left, right, node) -> tuple[_Reg, str]:
        a = self._to_f32(left, node)
        rv = None if isinstance(right, tuple) else _const_value(right)
        dst = self.s.new_reg("f32")
        if rv is not None:
            c = _f32(float(rv))
            if c == 0:
                raise NotSupported("division by a zero literal (Python raises ZeroDivisionError)", node)
            m, e = math.frexp(abs(c))
            if m == 0.5 and -127 <= e - 1 <= 127:
                # x / 2**k == x * 2**-k exactly (same real number, same single rounding)
                self.s.emit(f" mul.f32 {dst.name}, {a.name}, 0f{_f32_bits(1.0 / c):08X};")
                return dst, "f32"
        b = self._to_f32(right, node)
        self.s.emit(f" div.rn.f32 {dst.name}, {a.name}, {b.name};")
        return dst, "f32"

    def _floordiv(self, op, left, right, node) -> tuple[_Reg, str]:
        is_mod = isinstance(op, ast.Mod)
        sym = "%" if is_mod else "//"
        rv = None if isinstance(right, tuple) else _const_value(right)
        if rv is not None and not isinstance(left, tuple) and _const_value(left) is None:
            a, ka = self._eval(left)                   # keep a constant divisor an immediate
            if ka in ("s64", "u64"):
                raise NotSupported("64-bit floor division and modulo are not supported", node)
            if ka == "f32" or isinstance(rv, float):
                self._float_floordiv(sym, node)
            if rv == 0:
                raise NotSupported(f"{sym} by a zero literal (Python raises ZeroDivisionError)", node)
            if ka == "u32" and rv < 0 or not _I32_MIN <= rv <= (_I32_MAX if ka == "s32" else (1 << 32) - 1):
                raise NotSupported(f"divisor {rv} does not fit the {ka} operand", node)
            if ka not in ("s32", "u32"):
                self._mixed(ka, "s32", node)
            return self._divconst(a, ka, rv, is_mod), ka
        a, ka, b, kb = self._pair(left, right, node)
        if ka in ("s64", "u64") or kb in ("s64", "u64"):
            raise NotSupported("64-bit floor division and modulo are not supported", node)
        if "f32" in (ka, kb):
            self._float_floordiv(sym, node)
        if ka != kb or ka not in ("s32", "u32"):
            self._mixed(ka, kb, node)
        return self._divreg(a, ka, b, is_mod), ka

    @staticmethod
    def _float_floordiv(sym, node):
        raise NotSupported(f"'{sym}' on f32 is not supported: Python's float floor division and "
                           "modulo use an exact fmod with special sign/zero/inf rules (1 // 0.1 is "
                           "9.0 although 1 / 0.1 is 10.0) that no PTX instruction reproduces; use "
                           f"int operands, or int(x) {sym} int(y) if that is what you mean", node)

    def _divconst(self, a: _Reg, k: str, d: int, is_mod: bool) -> _Reg:
        """a // d or a % d for a literal d != 0, Python floor semantics."""
        e = self.s.emit
        new = lambda: self.s.new_reg(k)  # noqa: E731
        out = new()
        if k == "u32":                                # floor == truncation for unsigned
            if _is_pow2(d):
                sh = d.bit_length() - 1
                e(f" and.b32 {out.name}, {a.name}, {d - 1};" if is_mod else
                  f" shr.u32 {out.name}, {a.name}, {sh};")
            else:
                e(f" {'rem' if is_mod else 'div'}.u32 {out.name}, {a.name}, {d};")
            return out
        if d == 1 or d == -1:
            if is_mod:
                e(f" mov.s32 {out.name}, 0;")
            elif d == 1:
                e(f" mov.u32 {out.name}, {a.name};")
            else:
                e(f" neg.s32 {out.name}, {a.name};")    # -2**31 // -1 wraps (Python: 2**31)
            return out
        if _is_pow2(d):
            # two's-complement arithmetic shift IS floor division; the low bits ARE the
            # floor remainder (in [0, d)). No correction needed.
            e(f" and.b32 {out.name}, {a.name}, {d - 1};" if is_mod else
              f" shr.s32 {out.name}, {a.name}, {d.bit_length() - 1};")
            return out
        if d == _I32_MIN:                             # |d| not an i32: use the general path
            dr = new()
            e(f" mov.s32 {dr.name}, {d};")
            return self._divreg(a, k, dr, is_mod)
        q, p, r, m = new(), new(), new(), new()
        if d > 0:
            # trunc q, r = a - q*d (sign of a). Floor differs iff r < 0: then q -= 1, r += d.
            # m = r >> 31 is -1 exactly then, so q + m and r + (m & d) -- branch-free.
            e(f" div.s32 {q.name}, {a.name}, {d};")
            e(f" mul.lo.s32 {p.name}, {q.name}, {d};")
            e(f" sub.s32 {r.name}, {a.name}, {p.name};")
            e(f" shr.s32 {m.name}, {r.name}, 31;")
        else:
            # d < 0: divide by |d| (backend's divide-by-constant only takes positives) and
            # negate. Floor differs iff r > 0: m = (-r) >> 31 is -1 exactly then.
            t, nr = new(), new()
            e(f" div.s32 {t.name}, {a.name}, {-d};")
            e(f" neg.s32 {q.name}, {t.name};")
            e(f" mul.lo.s32 {p.name}, {t.name}, {-d};")
            e(f" sub.s32 {r.name}, {a.name}, {p.name};")
            e(f" neg.s32 {nr.name}, {r.name};")
            e(f" shr.s32 {m.name}, {nr.name}, 31;")
        if is_mod:
            md = new()
            e(f" and.b32 {md.name}, {m.name}, {d};")
            e(f" add.s32 {out.name}, {r.name}, {md.name};")
        else:
            e(f" add.s32 {out.name}, {q.name}, {m.name};")
        return out

    def _divreg(self, a: _Reg, k: str, b: _Reg, is_mod: bool) -> _Reg:
        """a // b or a % b, b a register: truncating div + branch-free floor correction."""
        e = self.s.emit
        new = lambda: self.s.new_reg(k)  # noqa: E731
        out = new()
        if k == "u32":
            e(f" {'rem' if is_mod else 'div'}.u32 {out.name}, {a.name}, {b.name};")
            return out
        q, p, r, x, nr, nz, sg, m = (new() for _ in range(8))
        e(f" div.s32 {q.name}, {a.name}, {b.name};")
        e(f" mul.lo.s32 {p.name}, {q.name}, {b.name};")
        e(f" sub.s32 {r.name}, {a.name}, {p.name};")             # trunc remainder, sign of a
        # floor != trunc iff r != 0 and sign(r) != sign(b):  bit 31 of (r ^ b) & (r | -r)
        e(f" xor.b32 {x.name}, {r.name}, {b.name};")
        e(f" neg.s32 {nr.name}, {r.name};")
        e(f" or.b32 {nz.name}, {r.name}, {nr.name};")
        e(f" and.b32 {sg.name}, {x.name}, {nz.name};")
        e(f" shr.s32 {m.name}, {sg.name}, 31;")                  # -1 if correction, else 0
        if is_mod:
            md = new()
            e(f" and.b32 {md.name}, {m.name}, {b.name};")
            e(f" add.s32 {out.name}, {r.name}, {md.name};")
        else:
            e(f" add.s32 {out.name}, {q.name}, {m.name};")
        return out

    def _compare(self, node: ast.Compare, branch: bool) -> tuple[_Reg, bool]:
        """Returns (pred, inverted): pred == (comparison) xor inverted. In a branch, f32 !=
        is emitted as eq + inverted polarity (our backend has no setp.neu); as a value it
        must be setp.neu.f32 -- ordered ne would be wrong (false) on NaN."""
        if len(node.ops) != 1 or len(node.comparators) != 1:
            raise NotSupported("chained comparisons (a < b < c) are not supported", node)
        a, ka, b, kb = self._pair(node.left, node.comparators[0], node)
        return self._compare_values(a, ka, b, kb, node.ops[0], node, branch)

    def _compare_values(self, a, ka, b, kb, operator, node, branch):
        """One typed relation, shared by ordinary and chained comparisons."""
        if ka != kb:
            raise NotSupported(f"comparing mismatched types ({ka} vs {kb})", node)
        if ka == "pred":
            raise NotSupported("comparing comparison results", node)
        op = type(operator)
        cmp = {ast.Lt: "lt", ast.LtE: "le", ast.Gt: "gt", ast.GtE: "ge", ast.Eq: "eq",
               ast.NotEq: "ne"}.get(op)
        if cmp is None:
            raise NotSupported(f"comparison {op.__name__} not supported", node)
        inverted = False
        if ka in ("s64", "u64"):
            return self._wide_compare(a, b, ka, cmp), False
        if ka == "f32" and cmp == "ne":
            cmp, inverted = ("eq", True) if branch else ("neu", False)
        dst = self.s.new_reg("pred")
        self.s.emit(f" setp.{cmp}.{ka} {dst.name}, {a.name}, {b.name};")
        return dst, inverted

    def _unaryop(self, node: ast.UnaryOp) -> tuple[_Reg, str]:
        if isinstance(node.op, ast.Invert):
            value, kind = self._eval(node.operand)
            if kind not in ("s32", "u32", "s64", "u64"):
                raise NotSupported("bitwise inversion requires an integer", node)
            result = self.s.new_reg(kind)
            self.s.emit(f" xor.b{64 if kind.endswith('64') else 32} {result.name}, {value.name}, {hex((1 << (64 if kind.endswith('64') else 32)) - 1)};")
            return result, kind
        if isinstance(node.op, ast.Not):
            result = self._boolean_expression(node)
            self._note_value(node, "pred")
            return result, "pred"
        if isinstance(node.op, ast.UAdd):
            return self._eval(node.operand)
        if not isinstance(node.op, ast.USub):
            raise NotSupported(f"unary operator {type(node.op).__name__} not supported", node)
        v, k = self._eval(node.operand)
        if k == "s64":
            dst = self.s.new_reg(k)
            self.s.emit(f" sub.s64 {dst.name}, 0, {v.name};")
            return dst, k
        if k not in ("f32", "s32"):
            raise NotSupported(f"unary - on {k}", node)
        dst = self.s.new_reg(k)
        self.s.emit(f" neg.{k} {dst.name}, {v.name};")
        return dst, k

    def _call(self, node: ast.Call) -> tuple[_Reg, str]:
        root = node.func
        while isinstance(root, ast.Attribute):
            root = root.value
        if isinstance(root, ast.Name) and root.id in self.local_names:
            raise NotSupported(f"call target {root.id!r} is a local value", node)
        if isinstance(node.func, ast.Name):
            if node.func.id in self.names.devices.definitions:
                return self.names.devices.inline(self, node)
        fname = self.names.resolve(node.func)
        if fname is None:
            graph = getattr(self.names, "graph", None)
            target = graph.resolve(self.names, node.func) if graph is not None else None
            if target is not None:
                target_names, target_name = target
                return target_names.devices.inline(self, node, name=target_name)
            raise NotSupported(f"call to {ast.unparse(node.func)!r} is not supported", node)
        if node.keywords:
            raise NotSupported(f"{fname}() keyword arguments are not supported", node)
        args, short = node.args, fname.split(".")[-1]
        if fname == "bool":
            if len(args) != 1:
                raise NotSupported("bool() takes exactly one scalar argument", node)
            result = self._boolean_value(*self._eval(args[0]))
            self._note_value(node, "pred")
            return result, "pred"
        if fname.startswith("pf.") and short in {"atomic_add", "atomic_exchange", "atomic_compare_exchange"}:
            from openassembler.pyfront.primitives import atomic
            return atomic(self,node,short)
        if fname.startswith("pf.") and short in _SREGS:
            if args:
                raise NotSupported(f"{short}() takes no arguments", node)
            r = self.s.new_reg("s32")
            self.s.emit(f" mov.u32 {r.name}, {_SREGS[short]};")
            return r, "s32"
        if fname.startswith("pf.") and short in _INT_TYPES:
            if len(args) != 1:
                raise NotSupported(f"{short}() takes exactly one integer argument", node)
            return self._integer_cast(_INT_TYPES[short], args[0], node)
        if fname in ("min", "max"):
            if len(args) < 2:
                raise NotSupported(f"{fname}() needs at least two arguments (no iterables)", node)
            # no side effects in this subset, so evaluating non-literals first is safe; a
            # literal takes the type of the first non-literal argument (as in _pair)
            ev = {i: self._eval(x) for i, x in enumerate(args) if _const_value(x) is None}
            lits = [_const_value(x) for x in args if _const_value(x) is not None]
            k = (next(iter(ev.values()))[1] if ev else
                 "f32" if any(isinstance(v, float) for v in lits) else "s32")
            vals = [ev[i] if i in ev else self._lit_as(_const_value(x), k, x)
                    for i, x in enumerate(args)]
            cur, k = vals[0]
            for b, kb in vals[1:]:                   # CPython folds left to right
                cur = self._minmax(fname, cur, k, b, kb, node)
            return cur, k
        if fname in ("abs", "float", "int") or fname.startswith(("pf.", "math.")) and short in _MATH1:
            if len(args) != 1:
                raise NotSupported(f"{short}() takes exactly one argument", node)
        if fname == "abs":
            v, k = self._eval(args[0])
            if k == "u32":
                return v, k
            dst = self.s.new_reg(k)
            if k == "f32":
                self.s.emit(f" abs.f32 {dst.name}, {v.name};")
            elif k == "s32":       # max(x, -x): abs.s32 exactly, incl. wrapping at -2**31
                n = self.s.new_reg("s32")
                self.s.emit(f" neg.s32 {n.name}, {v.name};")
                self.s.emit(f" max.s32 {dst.name}, {v.name}, {n.name};")
            else:
                raise NotSupported(f"abs() of {k}", node)
            return dst, k
        if fname == "float":
            cv = _const_value(args[0])
            return (self._lit_as(float(cv), "f32", node) if cv is not None
                    else (self._to_f32(args[0], node), "f32"))
        if fname == "int":
            cv = _const_value(args[0])
            if cv is not None:
                return self._lit_as(int(cv), "s32", node)
            v, k = self._eval(args[0])
            if k == "s32":
                return v, k
            if k != "f32":
                raise NotSupported(f"int() of {k} is not supported; use an explicit i32/u32/i64/u64 cast for integer bit conversion", node)
            dst = self.s.new_reg("s32")
            self.s.emit(f" cvt.rzi.s32.f32 {dst.name}, {v.name};")    # truncation, like int()
            return dst, "s32"
        if short in _MATH1 and (fname.startswith("pf.") or fname.startswith("math.")):
            return self._math(short, args[0], node), "f32"
        if fname == "pf.syncthreads":
            raise NotSupported("syncthreads() is a statement, not a value", node)
        if fname == "pf.shared_array":
            raise NotSupported("shared_array() must be assigned directly to a name at the "
                               "kernel's top level: s = shared_array(f32, 256)", node)
        raise NotSupported(f"call to {ast.unparse(node.func)!r} is not supported", node)

    def _minmax(self, fname, a, ka, b, kb, node) -> _Reg:
        if ka != kb:
            raise NotSupported(f"{fname}() of mixed types ({ka} vs {kb})", node)
        dst = self.s.new_reg(ka)
        if ka in ("s32", "u32"):
            self.s.emit(f" {fname}.{ka} {dst.name}, {a.name}, {b.name};")
        elif ka == "f32":
            # CPython: min(a, b) is `b if b < a else a` (max: b > a). PTX min.f32 would
            # return the non-NaN operand and order -0 < +0 -- both differ from Python.
            p = self.s.new_reg("pred")
            self.s.emit(f" setp.{'lt' if fname == 'min' else 'gt'}.f32 {p.name}, {b.name}, {a.name};")
            self.s.emit(f" selp.f32 {dst.name}, {b.name}, {a.name}, {p.name};")
        else:
            raise NotSupported(f"{fname}() of {ka}", node)
        return dst

    def _math(self, fn: str, arg, node) -> _Reg:
        x = self._to_f32(arg, node)
        dst = self.s.new_reg("f32")
        e = self.s.emit
        if fn == "sqrt":
            e(f" sqrt.rn.f32 {dst.name}, {x.name};")
        elif fn == "rsqrt":
            e(f" rsqrt.approx.f32 {dst.name}, {x.name};")
        elif fn == "exp":
            t = self.s.new_reg("f32")
            e(f" mul.f32 {t.name}, {x.name}, 0f{_LOG2E_F32:08X};")
            e(f" ex2.approx.f32 {dst.name}, {t.name};")
        elif fn == "log":
            t = self.s.new_reg("f32")
            e(f" lg2.approx.f32 {t.name}, {x.name};")
            e(f" mul.f32 {dst.name}, {t.name}, 0f{_LN2_F32:08X};")
        elif fn == "tanh":
            if _sm(self.s.arch) < 75:
                raise NotSupported(f"tanh() needs tanh.approx.f32 (sm_75+), target is {self.s.arch}", node)
            e(f" tanh.approx.f32 {dst.name}, {x.name};")
        return dst

    # ---- memory ----

    def _address(self, sub: ast.Subscript) -> tuple[_Arr, str]:
        """PTX address operand for arr[i] (with the address arithmetic emitted)."""
        if not isinstance(sub.value, ast.Name) or sub.value.id not in self.s.arrays:
            raise NotSupported("subscript target must be a declared array", sub)
        arr = self.s.arrays[sub.value.id]
        element_bytes = 8 if arr.elem in ("s64", "u64") else 4
        ci = _const_value(sub.slice)
        if isinstance(ci, int):
            if ci < 0:
                raise NotSupported("negative index: GPU arrays don't count from the end "
                                   "like Python lists", sub)
            if arr.length is not None and ci >= arr.length:
                raise NotSupported(f"index {ci} out of range for shared_array of {arr.length}", sub)
            if element_bytes * ci < (1 << 23):                    # fits the 24-bit signed address offset
                return arr, (f"[{arr.sym}+{element_bytes * ci}]" if arr.space == "shared"
                             else f"[{arr.base.name}+{element_bytes * ci}]")
        idx, idx_kind = self._eval(sub.slice)
        if idx_kind != "s32":
            raise NotSupported(f"array index must be an int32 expression, not {idx_kind}", sub)
        if arr.space == "shared":
            off, addr = self.s.new_reg("u32"), self.s.new_reg("u32")
            self.s.emit(f" shl.b32 {off.name}, {idx.name}, {3 if element_bytes == 8 else 2};")
            self.s.emit(f" add.s32 {addr.name}, {off.name}, {arr.base.name};")
            return arr, f"[{addr.name}]"
        addr = self.s.new_reg("ptr")
        # mul.wide.u32 (not .s32): our backend fuses mul.wide+add.s64 of a cbank-held
        # pointer into one IMAD.WIDE, and the encoder only has a template for the unsigned
        # form of that fusion (see openassembler/isa/encode.py). Negative indices are out
        # of bounds anyway (see the module docstring).
        self.s.emit(f" mul.wide.u32 {addr.name}, {idx.name}, {element_bytes};")
        self.s.emit(f" add.s64 {addr.name}, {arr.base.name}, {addr.name};")
        return arr, f"[{addr.name}]"

    def _load(self, arr: _Arr, addr: str) -> _Reg:
        dst = self.s.new_reg(arr.elem)
        self.s.emit(f" ld.{arr.space}.{arr.elem if arr.elem in ('f32','s64','u64') else 'u32'} {dst.name}, {addr};")
        return dst

    def _store(self, target: ast.Subscript, val: _Reg, kind: str) -> None:
        if isinstance(target.value,ast.Name) and getattr(self.s.arrays.get(target.value.id),"view",None) is not None:
            from openassembler.pyfront.buffers import access
            access(self,target,val,kind)
            return
        arr, addr = self._address(target)
        self._store_at(arr, addr, val, kind, target)

    def _store_at(self, arr: _Arr, addr: str, val: _Reg, kind: str, node) -> None:
        if kind != arr.elem:
            # `out[i] = 1` into an Array(f32) would store the integer's raw bits (1.4e-45,
            # not 1.0) and --verify can't see it: interpreter and SASS agree on the wrong value.
            raise NotSupported(f"storing a {kind} value into an Array of {arr.elem} "
                               "(write 1.0, not 1, for a float array; or convert with float()/int())", node)
        self.s.emit(f" st.{arr.space}.{arr.elem if arr.elem in ('f32','s64','u64') else 'u32'} {addr}, {val.name};")


# ---- entry points ----------------------------------------------------------------------------

def _f32(v: float) -> float:
    try:
        return struct.unpack("<f", struct.pack("<f", v))[0]
    except OverflowError:
        return math.copysign(math.inf, v)


def _sm(arch: str) -> int:
    digits = "".join(ch for ch in arch.split("_", 1)[-1] if ch.isdigit())
    return int(digits) if digits else 0


def _ident(name: str, what: str, node: ast.AST) -> str:
    if not (name.isascii() and name.isidentifier()) or name.startswith("__smem_"):
        raise NotSupported(f"{what} name {name!r} is not a valid PTX identifier (ASCII only, "
                           "no __smem_ prefix)", node)
    return name


def _check_shared_producers(fdef, names):
    """Reject shared reads with no possible store anywhere in the kernel.

    This proves a narrow missing-producer bug, not temporal initialization or
    race freedom. An unknown write index can cover any slot and is not rejected.
    """
    arrays, writes = set(), {}
    for node in ast.walk(fdef):
        if (isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)
                and names.resolve(node.value.func) == 'pf.shared_array'):
            arrays.update(t.id for t in node.targets if isinstance(t, ast.Name))
    for node in ast.walk(fdef):
        if (isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Store)
                and isinstance(node.value, ast.Name) and node.value.id in arrays):
            index = _const_value(node.slice)
            writes.setdefault(node.value.id, []).append(index if type(index) is int else None)
    for node in ast.walk(fdef):
        if (isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Load)
                and isinstance(node.value, ast.Name) and node.value.id in arrays):
            name = node.value.id
            possible = writes.get(name, [])
            index = _const_value(node.slice)
            if not possible or (type(index) is int and None not in possible and index not in possible):
                raise NotSupported(f"shared_array {name!r} read has no possible writer"
                                   + (f" for index {index}" if type(index) is int else ''), node)


def _compile_entry(fdef: ast.FunctionDef, names: _Names, arch: str) -> str:
    """One `.visible .entry` (no module header) for a kernel FunctionDef."""
    graph = getattr(names, 'graph', None)
    if graph is not None:
        fdef = graph.static_values.fold(names, fdef)
    _check_shared_producers(fdef, names)
    if isinstance(fdef, ast.AsyncFunctionDef):
        raise NotSupported("async kernels are not supported", fdef)
    a = fdef.args
    if a.vararg or a.kwarg or a.kwonlyargs or a.posonlyargs:
        raise NotSupported("kernel parameters must be plain positional parameters "
                           "(no *args, **kwargs, keyword-only or positional-only)", fdef)
    if a.defaults:
        raise NotSupported("kernel parameters can't have default values", fdef)
    _ident(fdef.name, "kernel", fdef)
    scope = _Scope(arch=arch)
    for arg in a.args:
        if arg.annotation is None:
            raise NotSupported(f"parameter {arg.arg!r} has no type annotation", fdef)
        t = _resolve_type(arg.annotation, names)
        _ident(arg.arg, "parameter", arg)
        if isinstance(t, ArrayType):
            scope.params.append(("u64", arg.arg))
            scope.arrays[arg.arg] = _Arr("global", t.elem.ptx, scope.new_reg("ptr"))
            from openassembler.pyfront.buffers import BufferType
            if isinstance(t,BufferType):scope.arrays[arg.arg].view=t
        else:
            scope.params.append((t.ptx, arg.arg))
            scope.vars[arg.arg] = scope.new_reg(t.ptx)
    _BarrierCheck(fdef, names).run()
    _Emitter(scope, names, fdef).body(fdef.body)
    return _render(fdef.name, scope)


def _render(name: str, s: _Scope) -> str:
    param_list = ", ".join(f".param .{t} {n}" for t, n in s.params)
    out = [f".visible .entry {name}({param_list})\n{{"]
    if s.wide_integer or any(a.view is not None for a in s.arrays.values()):
        contract = {"schema": 1, "kernel": name, "parameters": [
            {"name": pname, "kind": "array" if pname in s.arrays else "scalar",
             "dtype": s.arrays[pname].elem if pname in s.arrays else ptx_t}
            for ptx_t, pname in s.params]}
        out.insert(0, "// openassembler-pyfront-types-v1 " + json.dumps(contract, sort_keys=True, separators=(",", ":")))
    views=[(pname,s.arrays[pname].view) for _,pname in s.params if pname in s.arrays and s.arrays[pname].view is not None]
    if views:
        from openassembler.pyfront.buffers import metadata
        out.insert(0,"// openassembler-pyfront-buffer-v1 "+json.dumps({"schema":1,"kernel":name,"buffers":[metadata(pname,view) for pname,view in views]},sort_keys=True,separators=(",",":")))
    for sym, ty, count in s.shared:
        element_bytes = 8 if ty in ("s64", "u64") else 4
        storage_type = "b64" if element_bytes == 8 else "f32" if ty == "f32" else "b32"
        out.append(f" .shared .align {element_bytes} .{storage_type} {sym}[{count}];")
    if s.n_p:
        out.append(f" .reg .pred %p<{s.n_p + 1}>;")
    if s.n_r:
        out.append(f" .reg .b32 %r<{s.n_r + 1}>;")
    if s.n_f:
        out.append(f" .reg .f32 %f<{s.n_f + 1}>;")
    if s.n_rd:
        out.append(f" .reg .b64 %rd<{s.n_rd + 1}>;")
    for ptx_t, pname in s.params:
        if pname in s.arrays:
            r = s.arrays[pname].base
            out.append(f" ld.param.u64 {r.name}, [{pname}];")
            out.append(f" cvta.to.global.u64 {r.name}, {r.name};")
        else:
            r = s.vars[pname]
            out.append(f" ld.param.{ptx_t if ptx_t in ('f32','s64','u64') else 'u32'} {r.name}, [{pname}];")
    out.extend(s.lines)
    out.append(f"{s.exit_label}:")
    out.append(" ret;\n}")
    return "\n".join(out) + "\n"


def _parse_file(path: Path) -> ast.Module:
    try:
        src = Path(path).read_text(encoding="utf-8")
    except UnicodeDecodeError as e:
        raise NotSupported(f"{path} is not valid UTF-8: {e}", path=str(path)) from e
    try:
        return ast.parse(src, filename=str(path))
    except SyntaxError as e:
        raise NotSupported(f"syntax error in {path}:{e.lineno}: {e.msg}", path=str(path),
                           lineno=e.lineno) from e
    except ValueError as e:                     # e.g. NUL bytes in the source
        raise NotSupported(f"can't parse {path}: {e}", path=str(path)) from e


def _is_kernel_decorator(d: ast.expr, names: _Names) -> bool:
    if isinstance(d, ast.Call) and names.resolve(d.func) == "pf.kernel":
        raise NotSupported("@kernel takes no arguments (write @kernel, not @kernel())", d)
    return names.resolve(d) == "pf.kernel"


def _file_kernels(tree: ast.Module, names: _Names, path) -> list[ast.FunctionDef]:
    """Top-level @kernel functions, in file order; duplicate names are an error (Python
    would silently keep the last one)."""
    found: dict[str, ast.FunctionDef] = {}
    out = []
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                and any(_is_kernel_decorator(d, names) for d in n.decorator_list):
            if len(n.decorator_list) != 1:
                raise NotSupported(f"kernel {n.name!r} has other decorators besides @kernel", n)
            if n.name in found:
                raise NotSupported(f"duplicate @kernel name {n.name!r} (first defined on line "
                                   f"{found[n.name].lineno})", n)
            found[n.name] = n
            out.append(n)
    return out


def compile_file(path: Path, arch: str = "sm_80", *, source_root=None) -> tuple[str, list[str]]:
    """Compile every top-level @kernel in `path` into ONE PTX module (one header, one
    .visible .entry per kernel, in file order) WITHOUT importing or executing the file.
    Returns (ptx, kernel names); ("", []) if the file has no @kernel. Errors are
    NotSupported naming the file and line."""
    path = Path(path)
    tree = _parse_file(path)
    try:
        names = _Names(tree)
        kernels = _file_kernels(tree, names, path)
        if not kernels:
            return "", []
        from .modules import SourceGraph
        graph = SourceGraph(path, tree, names, source_root)
        entries = []
        for k in kernels:
            graph.kernel = k.name
            entries.append(_compile_entry(k, names, arch))
        prefix = graph.prefix()
    except NotSupported as e:
        raise e.with_path(path) from None
    return prefix + HEADER.format(arch=arch) + "\n".join(entries), [k.name for k in kernels]


class KernelRef:
    """A @kernel found statically in a file by find_kernel (the file was never executed).
    Accepted by compile_kernel exactly like the decorated function."""

    _is_openassembler_kernel = True

    def __init__(self, path: Path, fdef: ast.FunctionDef, names: _Names):
        self.path, self.fdef, self.names = path, fdef, names
        self.source_root = None
        self.__name__ = fdef.name

    def __repr__(self) -> str:
        return f"KernelRef({self.__name__!r} in {self.path}, line {self.fdef.lineno})"

    def __call__(self, *_a, **_kw):
        raise NotSupported(f"{self.__name__} is a @kernel found in {self.path}; compile it with "
                           "openassembler.pyfront.compile_kernel(...)")


def find_kernel(path, entry: str | None, *, source_root=None) -> KernelRef:
    """Find the @kernel named `entry` in a .py file (or the only one, if entry is None)
    WITHOUT importing it -- same static scan as compile_file. Raises NotSupported if none,
    or several when entry is None (same "don't guess" policy as the rest of pyfront)."""
    path = Path(path)
    tree = _parse_file(path)
    try:
        names = _Names(tree)
        kernels = {k.name: k for k in _file_kernels(tree, names, path)}
    except NotSupported as e:
        raise e.with_path(path) from None
    if entry:
        if entry not in kernels:
            raise NotSupported(f"no @kernel function named {entry!r} in {path}; have {list(kernels)}")
        fdef = kernels[entry]
    elif len(kernels) != 1:
        raise NotSupported(f"{path} has {len(kernels)} @kernel functions, pass --kernel NAME; "
                           f"have {list(kernels)}")
    else:
        (fdef,) = kernels.values()
    reference = KernelRef(path, fdef, names)
    reference.source_root = source_root
    return reference


def _live_source(fn) -> tuple[ast.FunctionDef, _Names, str | None]:
    """AST of a live function: parse its WHOLE source file (so the names/imports it sees
    are exactly those compile_file would see) and pick the def by line; fall back to the
    function's own source text (e.g. defined in a REPL)."""
    path = inspect.getsourcefile(fn)
    first = fn.__code__.co_firstlineno
    if path and Path(path).is_file():
        try:
            tree = ast.parse(Path(path).read_text(encoding="utf-8"), filename=path)
        except (SyntaxError, ValueError, UnicodeDecodeError):
            tree = None
        if tree is not None:
            for n in ast.walk(tree):
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == fn.__name__ \
                        and first in {n.lineno, *(d.lineno for d in n.decorator_list)}:
                    try:
                        return n, _Names(tree), path
                    except NotSupported as e:
                        raise e.with_path(path) from None
    lines, first = inspect.getsourcelines(fn)
    mod = ast.parse(textwrap.dedent("".join(lines)))
    ast.increment_lineno(mod, first - 1)     # report real file line numbers, not "line 2"
    (fdef,) = [n for n in mod.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    return fdef, _Names(mod), None


def compile_kernel(fn, arch: str = "sm_80", *, source_root=None) -> str:
    """PTX (header + one .visible .entry) for one kernel: a @kernel-decorated function, a
    plain function, or a find_kernel() result. Byte-identical to the corresponding entry
    of compile_file(). Raises NotSupported for anything outside the module docstring's
    subset."""
    if isinstance(fn, KernelRef):
        reference = find_kernel(fn.path, fn.__name__, source_root=source_root or fn.source_root)
        fdef, names, path = reference.fdef, reference.names, reference.path
        source_root = reference.source_root
    else:
        fdef, names, path = _live_source(getattr(fn, "__wrapped__", fn))
    try:
        if path is not None:
            from .modules import SourceGraph
            graph = SourceGraph(path, names.source_tree, names, source_root)
            graph.kernel = fdef.name
            entry = _compile_entry(fdef, names, arch)
            return graph.prefix() + HEADER.format(arch=arch) + entry
        return HEADER.format(arch=arch) + _compile_entry(fdef, names, arch)
    except NotSupported as e:
        raise (e.with_path(path) if path else e) from None


def kernel(fn):
    """Marks a function as pyfront-compilable. Calling it directly in plain Python raises --
    use openassembler.pyfront.compile_kernel(fn) to get PTX."""

    def _no_call(*_a, **_kw):
        raise NotSupported(f"{fn.__name__} is a @kernel function; call "
                           f"openassembler.pyfront.compile_kernel({fn.__name__}) to get PTX, "
                           "not the function itself")

    _no_call.__wrapped__ = fn
    _no_call.__name__ = fn.__name__
    _no_call.__qualname__ = getattr(fn, "__qualname__", fn.__name__)
    _no_call._is_openassembler_kernel = True
    return _no_call


def device(fn):
    """Mark a pure typed scalar helper for static inlining inside a @kernel.

    Device helpers have no Python execution or separate device-call ABI. Static
    compilation resolves their same-file source definitions without running them.
    """
    def _no_call(*_a, **_kw):
        raise NotSupported(f"{fn.__name__} is a @device helper; call it inside a compiled @kernel")

    _no_call.__wrapped__ = fn
    _no_call.__name__ = fn.__name__
    _no_call.__qualname__ = getattr(fn, "__qualname__", fn.__name__)
    return _no_call

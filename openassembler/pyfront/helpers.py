"""Static typed scalar helpers, inlined without a device-call ABI or host execution."""
from __future__ import annotations

import ast
from collections import Counter

from openassembler.pyfront import compile as c
from openassembler.pyfront.types import Scalar


class _ModuleBindings(ast.NodeVisitor):
    """Bindings in module control flow, without descending into function/class scopes."""

    def __init__(self):
        self.counts = Counter()

    def visit_Name(self, node):
        if isinstance(node.ctx, (ast.Store, ast.Del)):
            self.counts[node.id] += 1

    def visit_FunctionDef(self, node):
        self.counts[node.name] += 1

    visit_AsyncFunctionDef = visit_FunctionDef
    visit_ClassDef = visit_FunctionDef

    def visit_Import(self, node):
        for alias in node.names:
            self.counts[alias.asname or alias.name.split('.')[0]] += 1

    def visit_ImportFrom(self, node):
        for alias in node.names:
            self.counts[alias.asname or alias.name] += 1

    def visit_ExceptHandler(self, node):
        if node.name:
            self.counts[node.name] += 1
        self.generic_visit(node)


class _HelperFlow:
    """Conservative definite initialization; loops may execute zero iterations."""

    def __init__(self, parameters):
        self.parameters = set(parameters)

    def expr(self, expr, initialized):
        if isinstance(expr, ast.Name):
            if expr.id not in initialized:
                raise c.NotSupported(f'device helper local {expr.id!r} may be uninitialized or captured', expr)
        elif isinstance(expr, ast.Call):
            # Calls have statically resolved targets; only values are device reads.
            for arg in expr.args:
                self.expr(arg, initialized)
            for keyword in expr.keywords:
                self.expr(keyword.value, initialized)
        else:
            for child in ast.iter_child_nodes(expr):
                self.expr(child, initialized)

    def block(self, body, initialized, loop_depth=0):
        initialized = set(initialized)
        for st in body:
            if isinstance(st, ast.Assign) and len(st.targets) == 1 and isinstance(st.targets[0], ast.Name):
                self.expr(st.value, initialized)
                initialized.add(st.targets[0].id)
            elif isinstance(st, ast.AugAssign) and isinstance(st.target, ast.Name):
                self.expr(st.target, initialized)
                self.expr(st.value, initialized)
            elif isinstance(st, ast.If):
                self.expr(st.test, initialized)
                initialized = self.block(st.body, initialized, loop_depth) & self.block(st.orelse, initialized, loop_depth)
            elif isinstance(st, ast.For) and hasattr(st, '_oa_tuple_values'):
                if not isinstance(st.target, ast.Name):
                    raise c.NotSupported('device helper for target must be a scalar name', st.target)
                inner = set(initialized)
                inner.add(st.target.id)
                self.block(st.body, inner, loop_depth + 1)
                if st._oa_tuple_values:
                    initialized.add(st.target.id)
                after_else = self.block(st.orelse, initialized, loop_depth)
                # A break can bypass else. Nested-loop breaks do not target us.
                def breaks(body):
                    for part in body:
                        if isinstance(part, ast.Break):
                            return True
                        if isinstance(part, ast.If) and (breaks(part.body) or breaks(part.orelse)):
                            return True
                    return False
                if not st._oa_tuple_values or not breaks(st.body):
                    initialized = after_else
            elif isinstance(st, (ast.While, ast.For)):
                if st.orelse:
                    raise c.NotSupported('device helper loop else clauses are not supported', st.orelse[0])
                inner = set(initialized)
                if isinstance(st, ast.For):
                    if not isinstance(st.target, ast.Name):
                        raise c.NotSupported('device helper for target must be a scalar name', st.target)
                    self.expr(st.iter, initialized)
                    inner.add(st.target.id)
                else:
                    self.expr(st.test, initialized)
                self.block(st.body, inner, loop_depth + 1)
            elif isinstance(st, (ast.Break, ast.Continue)):
                if not loop_depth:
                    raise c.NotSupported('device helper break/continue outside a loop', st)
            elif isinstance(st, ast.Pass):
                pass
            else:
                raise c.NotSupported('device helpers allow scalar assignments, structured control flow and one terminal return', st)
        return initialized


class DeviceFunctions:
    """Static helper definitions with one hygienic scalar scope per call site."""

    def __init__(self, tree, names):
        self.names = names
        self.definitions = {}
        self.active = []
        self.signatures = {}
        bindings = _ModuleBindings()
        bindings.visit(tree)
        for node in tree.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            decorators = [names.resolve(d.func if isinstance(d, ast.Call) else d)
                          for d in node.decorator_list]
            if 'pf.device' not in decorators:
                continue
            if len(node.decorator_list) != 1 or isinstance(node.decorator_list[0], ast.Call):
                raise c.NotSupported('@device takes no arguments or additional decorators', node)
            if bindings.counts[node.name] != 1:
                raise c.NotSupported(f'device helper {node.name!r} is rebound at module scope', node)
            self.definitions[node.name] = node

    def _signature(self, name):
        graph = getattr(self.names, 'graph', None)
        signature_key = (name, graph.kernel) if graph is not None else name
        if signature_key in self.signatures:
            return self.signatures[signature_key]
        fn = self.definitions[name]
        graph = getattr(self.names, 'graph', None)
        if graph is not None:
            fn = graph.static_values.fold(self.names, fn)
        a = fn.args
        if isinstance(fn, ast.AsyncFunctionDef):
            raise c.NotSupported('async device helpers are not supported', fn)
        if a.vararg or a.kwarg or a.kwonlyargs or a.posonlyargs or a.defaults:
            raise c.NotSupported('device helpers require plain positional parameters without defaults', fn)
        if len({arg.arg for arg in a.args}) != len(a.args):
            raise c.NotSupported('duplicate device helper parameter names', fn)
        kinds = []
        for arg in a.args:
            if arg.annotation is None:
                raise c.NotSupported(f'device helper parameter {arg.arg!r} needs a scalar annotation', arg)
            ty = c._resolve_type(arg.annotation, self.names, allow_boolean=True)
            if not isinstance(ty, Scalar):
                raise c.NotSupported('device helpers accept scalar parameters only', arg)
            kinds.append(ty.ptx)
        if fn.returns is None:
            raise c.NotSupported('device helpers need an explicit scalar return annotation', fn)
        result = c._resolve_type(fn.returns, self.names, allow_boolean=True)
        if not isinstance(result, Scalar):
            raise c.NotSupported('device helpers return scalars only', fn)
        body = fn.body
        if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                and isinstance(body[0].value.value, str):
            body = body[1:]
        if not body:
            raise c.NotSupported('device helpers require scalar return paths', fn)
        locals_ = {arg.arg for arg in a.args} | {
            node.id for node in ast.walk(fn)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)}
        for st in body:
            for node in ast.walk(st):
                if isinstance(node, ast.Subscript):
                    raise c.NotSupported('device helpers cannot access memory', node)
                if isinstance(node, ast.Call):
                    root = node.func
                    while isinstance(root, ast.Attribute):
                        root = root.value
                    if isinstance(root, ast.Name) and root.id in locals_:
                        raise c.NotSupported(f'call target {root.id!r} is a local value', node)
                    target = self.names.resolve(node.func)
                    if target and target.startswith('pf.') and target[3:] not in c._MATH1 | set(c._INT_TYPES):
                        raise c.NotSupported('device helpers allow only pure scalar calls', node)
        from .return_flow import ReturnFlow
        flow = ReturnFlow(arg.arg for arg in a.args)
        paths = flow.analyze(body, flow.parameters)
        if paths.fall is not None or paths.breaks or paths.continues or not paths.returns:
            raise c.NotSupported('device helper requires a terminal return or scalar return on every path', fn)
        if graph is not None and (not isinstance(body[-1], ast.Return) or
                sum(isinstance(node, ast.Return) for statement in body
                    for node in ast.walk(statement)) != 1):
            graph.early_returns = True
        self.signatures[signature_key] = (fn, kinds, result.ptx, body)
        return self.signatures[signature_key]

    def inline(self, emitter, call, name=None):
        name = name or call.func.id
        graph = getattr(self.names, "graph", None)
        active = graph.active if graph is not None else self.active
        key = str(self.names.module_path)+":"+name if graph is not None else name
        if call.keywords:
            raise c.NotSupported('device helper keyword arguments are not supported', call)
        if key in active:
            raise c.NotSupported('recursive device helper call: ' + ' -> '.join(item.rsplit(':', 1)[-1] for item in [*active, key]), call)
        if len(active) >= 32:
            raise c.NotSupported('device helper nesting exceeds 32 calls', call)
        emitter.s.device_call_sites += 1
        if emitter.s.device_call_sites > 1024:
            raise c.NotSupported('device helper expansion exceeds 1024 call sites per kernel', call)
        try:
            fn, kinds, result_kind, body = self._signature(name)
        except c.NotSupported as error:
            if graph is not None:
                raise error.with_path(self.names.module_path) from None
            raise
        if len(call.args) != len(kinds):
            raise c.NotSupported(f'device helper {name!r} expects {len(kinds)} arguments, got {len(call.args)}', call)
        values = []
        for arg, want in zip(call.args, kinds):
            literal = c._const_value(arg)
            value, kind = (emitter._lit_as(literal, want, arg) if literal is not None
                           else emitter._eval(arg))
            if kind != want:
                raise c.NotSupported(f'device helper {name!r} expected {want}, got {kind}', arg)
            copy = emitter.s.new_reg(kind)
            emitter._mov(copy, value, kind)
            values.append(copy)
        saved_vars, saved_arrays = emitter.s.vars, emitter.s.arrays
        emitter.s.vars = {arg.arg: reg for arg, reg in zip(fn.args.args, values)}
        emitter.s.arrays = {}
        active.append(key)
        if graph is not None:
            graph.note_call(self.names, name, self.definitions[name], kinds, result_kind)
        try:
            child = c._Emitter(emitter.s, self.names, fn)
            # Preserve the established lowering for one terminal return.
            returns = [node for statement in body for node in ast.walk(statement)
                       if isinstance(node, ast.Return)]
            if len(returns) == 1 and isinstance(body[-1], ast.Return):
                child.body(body[:-1])
                value, kind = child._eval(body[-1].value)
                if kind != result_kind:
                    raise c.NotSupported(f'device helper {name!r} returns {kind}, annotated {result_kind}', body[-1])
                return value, kind
            returned = emitter.s.new_reg(result_kind)
            end = emitter.s.new_label('helperreturn')
            child.helper_return = (returned, result_kind, end)
            child.body(body)
            emitter.s.emit(f'{end}:')
            return returned, result_kind
        finally:
            active.pop()
            emitter.s.vars, emitter.s.arrays = saved_vars, saved_arrays

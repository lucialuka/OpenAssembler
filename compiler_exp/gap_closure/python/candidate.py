"""G04 candidate: typed, pure, straight-line scalar helper specialization.

This retained prototype imports compiler internals only to test the candidate;
production must never import this experiment.
"""
from __future__ import annotations

import ast
from collections import Counter

from openassembler.pyfront import compile as c
from openassembler.pyfront.types import Scalar


class DeviceFunctions:
    """Static helper definitions with one hygienic scalar scope per call site."""

    def __init__(self, tree, names):
        self.names = names
        self.definitions = {}
        self.active = []
        self.signatures = {}
        bindings = Counter()
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                bindings[node.name] += 1
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    bindings[alias.asname or alias.name.split('.')[0]] += 1
            elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    bindings.update(n.id for n in ast.walk(target) if isinstance(n, ast.Name))
        for node in tree.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            decorators = [names._dotted(d.func if isinstance(d, ast.Call) else d)
                          for d in node.decorator_list]
            matches = {'@device', 'openassembler.pyfront.device', 'pyfront.device'}
            if not any(d in matches for d in decorators):
                continue
            if len(node.decorator_list) != 1 or isinstance(node.decorator_list[0], ast.Call):
                raise c.NotSupported('@device takes no arguments or additional decorators', node)
            if bindings[node.name] != 1:
                raise c.NotSupported(f'device helper {node.name!r} is rebound at module scope', node)
            self.definitions[node.name] = node

    def _signature(self, name):
        if name in self.signatures:
            return self.signatures[name]
        fn = self.definitions[name]
        a = fn.args
        if isinstance(fn, ast.AsyncFunctionDef):
            raise c.NotSupported('async device helpers are not supported', fn)
        if a.vararg or a.kwarg or a.kwonlyargs or a.posonlyargs or a.defaults:
            raise c.NotSupported('device helpers require plain positional parameters without defaults', fn)
        kinds = []
        for arg in a.args:
            if arg.annotation is None:
                raise c.NotSupported(f'device helper parameter {arg.arg!r} needs a scalar annotation', arg)
            ty = c._resolve_type(arg.annotation, self.names)
            if not isinstance(ty, Scalar):
                raise c.NotSupported('device helpers accept scalar parameters only', arg)
            kinds.append(ty.ptx)
        if fn.returns is None:
            raise c.NotSupported('device helpers need an explicit scalar return annotation', fn)
        result = c._resolve_type(fn.returns, self.names)
        if not isinstance(result, Scalar):
            raise c.NotSupported('device helpers return scalars only', fn)
        body = fn.body
        if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                and isinstance(body[0].value.value, str):
            body = body[1:]
        if not body or not isinstance(body[-1], ast.Return) or body[-1].value is None:
            raise c.NotSupported('device helpers require a terminal return expression', fn)
        for st in body[:-1]:
            if isinstance(st, ast.Assign) and len(st.targets) == 1 and isinstance(st.targets[0], ast.Name):
                continue
            if isinstance(st, ast.AugAssign) and isinstance(st.target, ast.Name):
                continue
            raise c.NotSupported('device helpers allow only scalar assignments and a terminal return', st)
        for st in body:
            for node in ast.walk(st):
                if isinstance(node, ast.Subscript):
                    raise c.NotSupported('device helpers cannot access memory', node)
                if isinstance(node, ast.Call):
                    target = self.names.resolve(node.func)
                    if target and target.startswith('pf.') and target[3:] not in c._MATH1:
                        raise c.NotSupported('device helpers allow only pure scalar calls', node)
        self.signatures[name] = (fn, kinds, result.ptx, body)
        return self.signatures[name]

    def inline(self, emitter, call, emitter_type):
        name = call.func.id
        if call.keywords:
            raise c.NotSupported('device helper keyword arguments are not supported', call)
        if name in self.active:
            raise c.NotSupported('recursive device helper call: ' + ' -> '.join([*self.active, name]), call)
        if len(self.active) >= 32:
            raise c.NotSupported('device helper nesting exceeds 32 calls', call)
        fn, kinds, result_kind, body = self._signature(name)
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
        self.active.append(name)
        try:
            child = emitter_type(emitter.s, self.names, fn)
            child.body(body[:-1])
            value, kind = child._eval(body[-1].value)
            if kind != result_kind:
                raise c.NotSupported(f'device helper {name!r} returns {kind}, annotated {result_kind}', body[-1])
            return value, kind
        finally:
            self.active.pop()
            emitter.s.vars, emitter.s.arrays = saved_vars, saved_arrays


class CandidateEmitter(c._Emitter):
    def _call(self, node):
        if isinstance(node.func, ast.Name):
            local = {a.arg for a in self.fdef.args.args} | {
                n.id for n in ast.walk(self.fdef) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
            if node.func.id in local:
                raise c.NotSupported(f'call target {node.func.id!r} is a local value', node)
            if node.func.id in self.names.devices.definitions:
                return self.names.devices.inline(self, node, CandidateEmitter)
        return super()._call(node)


def compile_candidate(path):
    tree = c._parse_file(path)
    names = c._Names(tree)
    names.devices = DeviceFunctions(tree, names)
    kernels = c._file_kernels(tree, names, path)
    entries = []
    for fn in kernels:
        scope = c._Scope(arch='sm_80')
        for arg in fn.args.args:
            ty = c._resolve_type(arg.annotation, names)
            if isinstance(ty, Scalar):
                scope.params.append((ty.ptx, arg.arg))
                scope.vars[arg.arg] = scope.new_reg(ty.ptx)
            else:
                scope.params.append(('u64', arg.arg))
                scope.arrays[arg.arg] = c._Arr('global', ty.elem.ptx, scope.new_reg('ptr'))
        c._check_shared_producers(fn, names)
        c._BarrierCheck(fn, names).run()
        CandidateEmitter(scope, names, fn).body(fn.body)
        entries.append(c._render(fn.name, scope))
    return c.HEADER.format(arch='sm_80') + '\n'.join(entries)

"""Structured pure scalar helper candidate; experiment code is never imported by production."""
from __future__ import annotations

import ast
from unittest.mock import patch
from pathlib import Path
import types

from openassembler.pyfront import compile as c
from openassembler.pyfront import helpers


class HelperFlow:
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


_frozen = types.ModuleType('oa_frozen_scalar_helpers')
_frozen.__file__ = str(Path(__file__).with_name('helpers-before.py'))
exec(compile(Path(_frozen.__file__).read_bytes(), _frozen.__file__, 'exec'), _frozen.__dict__)


class CandidateFunctions(_frozen.DeviceFunctions):
    def _signature(self, name):
        if name in self.signatures:
            return self.signatures[name]
        # Retain all original type, signature, memory and purity checks while
        # independently validating the structured body rather than flattening it.
        fn = self.definitions[name]
        original = fn.body
        body = original[1:] if original and isinstance(original[0], ast.Expr) and isinstance(original[0].value, ast.Constant) and isinstance(original[0].value.value, str) else original
        if not body or not isinstance(body[-1], ast.Return) or body[-1].value is None:
            raise c.NotSupported('device helpers require a terminal return expression', fn)
        flow = HelperFlow(a.arg for a in fn.args.args)
        initialized = flow.block(body[:-1], flow.parameters)
        flow.expr(body[-1].value, initialized)
        # Parent signature checker can validate signatures/purity on a surrogate
        # body with only a terminal return; then inspect all actual expressions.
        try:
            fn.body = [body[-1]]
            result = super()._signature(name)
        finally:
            fn.body = original
        for st in body:
            for node in ast.walk(st):
                if isinstance(node, ast.Subscript):
                    raise c.NotSupported('device helpers cannot access memory', node)
                if isinstance(node, ast.Call):
                    target = self.names.resolve(node.func)
                    if target and target.startswith('pf.') and target[3:] not in c._MATH1:
                        raise c.NotSupported('device helpers allow only pure scalar calls', node)
        result = (*result[:3], body)
        self.signatures[name] = result
        return result


def compile_candidate(path):
    with patch.object(helpers, 'DeviceFunctions', CandidateFunctions):
        return c.compile_file(path)[0]

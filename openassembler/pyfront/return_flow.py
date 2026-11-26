"""Definite initialization and scalar returns across structured pure helper paths."""
from __future__ import annotations

import ast
from dataclasses import dataclass, field

from openassembler.pyfront import compile as c
from openassembler.pyfront.helpers import _HelperFlow


@dataclass
class Paths:
    fall: set[str] | None
    breaks: list[set[str]] = field(default_factory=list)
    continues: list[set[str]] = field(default_factory=list)
    returns: int = 0


def meet(values):
    """A terminated path imposes no initialization requirement at the join."""
    active = [set(value) for value in values if value is not None]
    return set.intersection(*active) if active else None


class ReturnFlow(_HelperFlow):
    """Conservative loop analysis; only finite nonempty tuples prove a first trip."""

    def analyze(self, body, initialized, loop_depth=0):
        state = Paths(set(initialized))
        for node in body:
            if state.fall is None:
                break
            current = state.fall
            if isinstance(node, ast.Return):
                if node.value is None:
                    raise c.NotSupported('scalar helper return requires a value', node)
                self.expr(node.value, current)
                branch = Paths(None, returns=1)
            elif isinstance(node, ast.If):
                self.expr(node.test, current)
                left = self.analyze(node.body, current, loop_depth)
                right = self.analyze(node.orelse, current, loop_depth)
                branch = Paths(meet([left.fall, right.fall]),
                               left.breaks + right.breaks,
                               left.continues + right.continues,
                               left.returns + right.returns)
            elif isinstance(node, (ast.For, ast.While)):
                static_tuple = isinstance(node, ast.For) and hasattr(node, '_oa_tuple_values')
                inner = set(current)
                nonempty = False
                if isinstance(node, ast.For):
                    if not isinstance(node.target, ast.Name):
                        raise c.NotSupported('helper loop target must be a scalar name', node.target)
                    if static_tuple:
                        nonempty = bool(node._oa_tuple_values)
                    else:
                        self.expr(node.iter, current)
                    inner.add(node.target.id)
                else:
                    self.expr(node.test, current)
                body_paths = self.analyze(node.body, inner, loop_depth + 1)
                exhaustion = set(current)
                if nonempty:
                    exhaustion.add(node.target.id)
                # Every first-iteration path returns; this tuple cannot exhaust.
                if (nonempty and body_paths.fall is None and body_paths.returns
                        and not body_paths.breaks and not body_paths.continues):
                    branch = Paths(None, returns=body_paths.returns)
                else:
                    # Else belongs to the enclosing loop context. This loop's
                    # own break paths bypass it; nested breaks are consumed by
                    # their own analysis and do not appear in body_paths.
                    alternate = self.analyze(node.orelse, exhaustion, loop_depth)
                    branch = Paths(meet([alternate.fall, *body_paths.breaks]),
                                   alternate.breaks, alternate.continues,
                                   body_paths.returns + alternate.returns)
            elif isinstance(node, ast.Break):
                if not loop_depth:
                    raise c.NotSupported('helper break outside a loop', node)
                branch = Paths(None, breaks=[set(current)])
            elif isinstance(node, ast.Continue):
                if not loop_depth:
                    raise c.NotSupported('helper continue outside a loop', node)
                branch = Paths(None, continues=[set(current)])
            else:
                branch = Paths(super().block([node], current, loop_depth))
            state.fall = branch.fall
            state.breaks.extend(branch.breaks)
            state.continues.extend(branch.continues)
            state.returns += branch.returns
        return state

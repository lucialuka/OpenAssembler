"""Isolated path-sensitive initialization and all-path scalar return analysis."""
from dataclasses import dataclass,field
import ast
from openassembler.pyfront import compile as c
from openassembler.pyfront.helpers import _HelperFlow
@dataclass
class Paths:
 fall: set|None
 breaks: list=field(default_factory=list)
 continues: list=field(default_factory=list)
 returns: int=0

def meet(values):
 active=[value for value in values if value is not None]
 return set.intersection(*map(set,active))if active else None

class ReturnFlow(_HelperFlow):
 def analyze(self,body,initialized,loop_depth=0):
  state=Paths(set(initialized))
  for node in body:
   if state.fall is None:break
   current=state.fall
   if isinstance(node,ast.Return):
    if node.value is None:raise c.NotSupported('scalar helper return requires a value',node)
    self.expr(node.value,current);branch=Paths(None,returns=1)
   elif isinstance(node,ast.If):
    self.expr(node.test,current);left=self.analyze(node.body,current,loop_depth);right=self.analyze(node.orelse,current,loop_depth)
    branch=Paths(meet([left.fall,right.fall]),left.breaks+right.breaks,left.continues+right.continues,left.returns+right.returns)
   elif isinstance(node,(ast.For,ast.While)):
    if node.orelse and not (isinstance(node,ast.For) and hasattr(node,'_oa_tuple_values')):
     raise c.NotSupported('device helper loop else clauses are not supported',node.orelse[0])
    inner=set(current)
    if isinstance(node,ast.For):
     if not isinstance(node.target,ast.Name):raise c.NotSupported('helper loop target must be a scalar name',node.target)
     if hasattr(node,'_oa_tuple_values'):
      nonempty=bool(node._oa_tuple_values)
     else:self.expr(node.iter,current);nonempty=False
     inner.add(node.target.id)
    else:self.expr(node.test,current);nonempty=False
    body_paths=self.analyze(node.body,inner,loop_depth+1)
    exhaustion=set(current)
    if nonempty:exhaustion.add(node.target.id)
    # A finite tuple whose every first-iteration path returns cannot exhaust.
    if nonempty and body_paths.fall is None and not body_paths.breaks and not body_paths.continues:
     branch=Paths(None,returns=body_paths.returns)
    else:
     alternate=self.analyze(node.orelse,exhaustion,loop_depth)
     fall=meet([alternate.fall,*body_paths.breaks])
     branch=Paths(fall,alternate.breaks,alternate.continues,body_paths.returns+alternate.returns)
   elif isinstance(node,ast.Break):
    if not loop_depth:raise c.NotSupported('helper break outside a loop',node)
    branch=Paths(None,breaks=[set(current)])
   elif isinstance(node,ast.Continue):
    if not loop_depth:raise c.NotSupported('helper continue outside a loop',node)
    branch=Paths(None,continues=[set(current)])
   else:branch=Paths(super().block([node],current,loop_depth))
   state.fall=branch.fall;state.breaks.extend(branch.breaks);state.continues.extend(branch.continues);state.returns+=branch.returns
  return state

"""Stage original helper return labels; production remains unchanged."""
from pathlib import Path
HERE=Path(__file__).resolve().parent

def once(text,old,new):
 assert text.count(old)==1,old
 return text.replace(old,new,1)
if __name__=='__main__':
 text=(HERE/'baseline-helpers.py').read_text()
 old="""        if not body or not isinstance(body[-1], ast.Return) or body[-1].value is None:
            raise c.NotSupported('device helpers require a terminal return expression', fn)"""
 new="""        if not body:
            raise c.NotSupported('device helpers require scalar return paths', fn)"""
 text=once(text,old,new)
 old="""        flow = _HelperFlow(arg.arg for arg in a.args)
        initialized = flow.block(body[:-1], flow.parameters)
        flow.expr(body[-1].value, initialized)"""
 new="""        from .return_flow import ReturnFlow
        flow = ReturnFlow(arg.arg for arg in a.args)
        paths = flow.analyze(body, flow.parameters)
        if paths.fall is not None or paths.breaks or paths.continues or not paths.returns:
            raise c.NotSupported('device helper requires a terminal return or scalar return on every path', fn)
        if graph is not None and (not isinstance(body[-1], ast.Return) or
                sum(isinstance(node, ast.Return) for statement in body
                    for node in ast.walk(statement)) != 1):
            graph.early_returns = True"""
 text=once(text,old,new)
 old="""            child.body(body[:-1])
            value, kind = child._eval(body[-1].value)
            if kind != result_kind:
                raise c.NotSupported(f'device helper {name!r} returns {kind}, annotated {result_kind}', body[-1])
            return value, kind"""
 new="""            # Preserve the established lowering for one terminal return.
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
            return returned, result_kind"""
 text=once(text,old,new);(HERE/'candidate-helpers.py').write_text(text)
 text=(HERE/'baseline-pyfront.py').read_text()
 anchor='    def visit_Return(self, node: ast.Return) -> None:\n'
 addition=anchor+'''        context = getattr(self, 'helper_return', None)
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
'''
 text=once(text,anchor,addition);(HERE/'candidate-pyfront.py').write_text(text)
 text=(HERE/'baseline-modules.py').read_text()
 text=once(text,'if not self.cross_module and not self.static_values.records and not self.static_values.iterations:',
           "if not self.cross_module and not self.static_values.records and not self.static_values.iterations and not getattr(self, 'early_returns', False):")
 (HERE/'candidate-modules.py').write_text(text)

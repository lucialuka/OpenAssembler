"""Stage finite tuple-loop source semantics and native lowering, preserving source graphs."""
from pathlib import Path
HERE=Path(__file__).resolve().parent

def once(text,old,new):
 assert text.count(old)==1,old
 return text.replace(old,new,1)
if __name__=='__main__':
 text=(HERE/'baseline-static_values.py').read_text()
 text=once(text,'        self.steps=0','        self.steps=0\n        self.iterations={}')
 text=once(text,"        if not declarations and not info['imports']:return fn",'')
 anchor='''            def visit_Call(self,node):
                node.args=[self.visit(arg)for arg in node.args]'''
 replacement='''            def visit_For(self,node):
                # Existing range loops retain their independent runtime contract.
                if isinstance(node.iter,ast.Call) and names.resolve(node.iter.func)=='range':
                    return self.generic_visit(node)
                value=owner.evaluate(node.iter,info,locals_=local)
                if value is MISSING or type(value)is not tuple:
                    owner.fail('iteration requires a compile-time immutable tuple or range',node.iter,info['path'])
                if len(value)>256:owner.fail('tuple iteration exceeds 256 elements',node.iter,info['path'])
                if not isinstance(node.target,ast.Name):owner.fail('tuple iteration target must be a scalar name',node.target,info['path'])
                relative=info['path'].relative_to(owner.graph.root).as_posix()
                key=relative+':'+fn.name+':'+str(node.lineno)+':'+str(node.col_offset)
                record=owner.iterations.setdefault(key,dict(path=relative,function=fn.name,line=node.lineno,
                    column=node.col_offset,values=encoded(value),kernels={}))
                record['kernels'][owner.graph.kernel]=1
                node._oa_tuple_values=value
                node.iter=ast.copy_location(ast.Tuple(elts=[ast.copy_location(ast.Constant(item),node.iter)for item in value],ctx=ast.Load()),node.iter)
                node.body=[self.visit(st)for st in node.body]
                node.orelse=[self.visit(st)for st in node.orelse]
                return node
            def visit_Call(self,node):
                node.args=[self.visit(arg)for arg in node.args]'''
 text=once(text,anchor,replacement)
 (HERE/'candidate-static_values.py').write_text(text+(HERE/'validation_helper.txt').read_text())
 text=(HERE/'baseline-pyfront.py').read_text()
 anchor='''    def visit_For(self, node: ast.For) -> None:
        # for v in range(start, stop[, step]) -- the only `for` form supported.'''
 replacement=(HERE/'tuple_emitter.py').read_text()+'''    def visit_For(self, node: ast.For) -> None:
        if hasattr(node, '_oa_tuple_values'):
            return self._static_tuple_loop(node)
        # Existing runtime range-loop contract.'''
 text=once(text,anchor,replacement)
 anchor='''            # all threads that entered the loop meet again after it (break/continue don't'''
 addition='''            if isinstance(st, ast.For) and hasattr(st, '_oa_tuple_values'):
                a, b = self._block(st.orelse, d or le)
                ret_exit = re_ or b
                return a, ret_exit
'''+anchor
 text=once(text,anchor,addition)
 text=once(text,'  Loops: for v in range(start, stop[, step]) with a constant non-zero step, while cond:', '  Loops: bounded static numeric tuples (including break/continue/nested/else),\\n          for v in range(start, stop[, step]) with a constant non-zero step, while cond:')
 text=text.replace('nested/else),\\n','nested/else),\n')
 (HERE/'candidate-pyfront.py').write_text(text)
 text=(HERE/'baseline-helpers.py').read_text()
 anchor='''            elif isinstance(st, (ast.While, ast.For)):
                if st.orelse:'''
 replacement='''            elif isinstance(st, ast.For) and hasattr(st, '_oa_tuple_values'):
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
                if st.orelse:'''
 text=once(text,anchor,replacement)
 (HERE/'candidate-helpers.py').write_text(text)
 text=(HERE/'baseline-modules.py').read_text()
 text=once(text,'if not self.cross_module and not self.static_values.records:', 'if not self.cross_module and not self.static_values.records and not self.static_values.iterations:')
 anchor="        if self.static_values.records:\n            result['static_values'] = {key:self.static_values.records[key] for key in sorted(self.static_values.records)}"
 text=once(text,anchor,anchor+"\n        if self.static_values.iterations:\n            result['static_iterations'] = {key:self.static_values.iterations[key] for key in sorted(self.static_values.iterations)}")
 text=once(text,"or not (obj['helpers'] or obj.get('static_values')):","or not (obj['helpers'] or obj.get('static_values') or obj.get('static_iterations')):")
 anchor="    return obj\n\n\ndef validate_sources"
 addition="""    if 'static_iterations' in obj:
        from .static_values import validate_iterations
        validate_iterations(obj['static_iterations'], paths)
    return obj


def validate_sources"""
 text=once(text,anchor,addition)
 (HERE/'candidate-modules.py').write_text(text)

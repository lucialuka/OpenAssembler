"""Stage original-source static-value folding and graph custody before promotion."""
from pathlib import Path
HERE=Path(__file__).resolve().parent

def once(text,old,new):
 assert text.count(old)==1,old
 return text.replace(old,new,1)
if __name__=='__main__':
 text=(HERE/'baseline-modules.py').read_text()
 text=once(text,'        self.calls = {}','        self.calls = {}\n        from .static_values import StaticValues\n        self.static_values = StaticValues(self)')
 text=once(text,'if isinstance(value, ast.Attribute) and isinstance(value.ctx, (ast.Store, ast.Del)):','if isinstance(value, (ast.Attribute, ast.Subscript)) and isinstance(value.ctx, (ast.Store, ast.Del)):')
 text=once(text,'while isinstance(head, ast.Attribute):\n                            head = head.value','while isinstance(head, (ast.Attribute, ast.Subscript)):\n                            head = head.value')
 text=once(text,'        if not self.cross_module:\n            return None','        if not self.cross_module and not self.static_values.records:\n            return None')
 text=once(text,'        return dict(schema=1, strategy=\'static-local-python-helper-source-graph\',','        result = dict(schema=1, strategy=\'static-local-python-helper-source-graph\',')
 text=once(text,'                    helpers={key: self.calls[key] for key in sorted(self.calls)})','''                    helpers={key: self.calls[key] for key in sorted(self.calls)})
        if self.static_values.records:
            result['static_values'] = {key:self.static_values.records[key] for key in sorted(self.static_values.records)}
        return result''')
 text=once(text,"or not isinstance(obj.get('helpers'), dict) or not obj['helpers']:","or not isinstance(obj.get('helpers'), dict) or not (obj['helpers'] or obj.get('static_values')):")
 text=once(text,'    return obj\n\n\ndef validate_sources','''    if 'static_values' in obj:
        from .static_values import validate_records
        validate_records(obj['static_values'], paths)
    return obj


def validate_sources''')
 (HERE/'candidate-modules.py').write_text(text)
 text=(HERE/'baseline-pyfront.py').read_text()
 anchor='''    _check_shared_producers(fdef, names)
    if isinstance(fdef, ast.AsyncFunctionDef):'''
 replacement='''    graph = getattr(names, 'graph', None)
    if graph is not None:
        fdef = graph.static_values.fold(names, fdef)
    _check_shared_producers(fdef, names)
    if isinstance(fdef, ast.AsyncFunctionDef):'''
 text=once(text,anchor,replacement)
 text=once(text,'          captures, recursion, early returns or runtime calls are admitted.', '          mutable captures, recursion, early returns or runtime calls are admitted. Explicit\\n          typing.Final numeric/tuple source values have bounded compile-time semantics.')
 # Replace intentional literal escape with a source newline in the contract docstring.
 text=text.replace('Explicit\\n          typing.Final','Explicit\n          typing.Final')
 (HERE/'candidate-pyfront.py').write_text(text)
 text=(HERE/'baseline-helpers.py').read_text()
 text=once(text,'        if name in self.signatures:\n            return self.signatures[name]', "        graph = getattr(self.names, 'graph', None)\n        signature_key = (name, graph.kernel) if graph is not None else name\n        if signature_key in self.signatures:\n            return self.signatures[signature_key]")
 text=once(text,'        self.signatures[name] = (fn, kinds, result.ptx, body)\n        return self.signatures[name]', '        self.signatures[signature_key] = (fn, kinds, result.ptx, body)\n        return self.signatures[signature_key]')
 text=once(text,'graph.note_call(self.names, name, fn, kinds, result_kind)', 'graph.note_call(self.names, name, self.definitions[name], kinds, result_kind)')

 text=once(text,'        fn = self.definitions[name]\n        a = fn.args', '''        fn = self.definitions[name]
        graph = getattr(self.names, 'graph', None)
        if graph is not None:
            fn = graph.static_values.fold(self.names, fn)
        a = fn.args''')
 (HERE/'candidate-helpers.py').write_text(text)

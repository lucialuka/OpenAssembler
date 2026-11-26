"""Stage contextual scalar Boolean values with lazy operand emission."""
from pathlib import Path
HERE=Path(__file__).resolve().parent
def once(text,old,new):
 assert text.count(old)==1,old;return text.replace(old,new,1)
if __name__=='__main__':
 text=(HERE/'first-boolean-pyfront.py').read_text()
 start=text.index('    def _boolean_operands(self, node):\n');end=text.index('    def _mov(',start)
 replacement='''    def _boolean_operands(self, node):
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

'''
 text=text[:start]+replacement+text[end:]
 text=once(text,'''        if isinstance(node, ast.Compare) and len(node.ops) > 1:
            return self._boolean_expression(node), "pred"''','''        if isinstance(node, ast.Compare) and len(node.ops) > 1:
            result = self._boolean_expression(node)
            self._note_value(node, "pred")
            return result, "pred"''')
 text=once(text,'''        if isinstance(node, ast.Constant) and isinstance(node.value, bool):
            value, kind = self._lit_as(int(node.value), "s32", node)
            return self._boolean_value(value, kind), "pred"''','''        if isinstance(node, ast.Constant) and isinstance(node.value, bool):
            value, kind = self._lit_as(int(node.value), "s32", node)
            self._note_value(node, "pred")
            return self._boolean_value(value, kind), "pred"''')
 text=once(text,'''        if isinstance(node.op, ast.Not):
            return self._boolean_expression(node), "pred"''','''        if isinstance(node.op, ast.Not):
            result = self._boolean_expression(node)
            self._note_value(node, "pred")
            return result, "pred"''')
 text=once(text,'''            return self._boolean_value(*self._eval(args[0])), "pred"''','''            result = self._boolean_value(*self._eval(args[0]))
            self._note_value(node, "pred")
            return result, "pred"''')
 (HERE/'candidate-pyfront.py').write_text(text)
 text=(HERE/'baseline-modules.py').read_text();text=once(text,'        self.comparisons = {}\n','        self.comparisons = {}\n        self.boolean_values = {}\n')
 method='''    def note_boolean(self, names, function, expression, kind):
        relative = names.module_path.relative_to(self.root).as_posix()
        category = type(expression).__name__
        key = f'{relative}:{function.name}:{expression.lineno}:{expression.col_offset}:{category}'
        digest = hashlib.sha256(ast.dump(expression, include_attributes=False).encode()).hexdigest()
        row = self.boolean_values.setdefault(key, dict(path=relative, function=function.name,
            line=expression.lineno, column=expression.col_offset, category=category,
            result_kind=kind, specialized_expression_sha256=digest, kernels={}))
        if row['specialized_expression_sha256'] != digest or row['result_kind'] != kind:
            self.fail('Boolean value specialization differs at original location', expression)
        row['kernels'][self.kernel] = row['kernels'].get(self.kernel, 0) + 1

'''
 text=once(text,'    def contract(self):\n',method+'    def contract(self):\n')
 text=once(text,'and not self.comparisons:', 'and not self.comparisons and not self.boolean_values:')
 text=once(text,"        if self.comparisons:\n            result['comparison_chains']", "        if self.boolean_values:\n            result['boolean_values'] = {key:self.boolean_values[key] for key in sorted(self.boolean_values)}\n        if self.comparisons:\n            result['comparison_chains']")
 text=once(text,"obj.get('comparison_chains')):","obj.get('comparison_chains') or obj.get('boolean_values')):")
 text=once(text,"    return obj\n","    if 'boolean_values' in obj:\n        validate_boolean_values(obj['boolean_values'], paths)\n    return obj\n")
 text+='''\n\ndef validate_boolean_values(records, paths):
    """Validate the original value expression, fixed result kind and use counts."""
    if not isinstance(records, dict) or not records:
        raise ValueError('invalid Boolean-value inventory')
    for key, row in records.items():
        if not isinstance(row, dict) or row.get('path') not in paths:
            raise ValueError('invalid Boolean-value source')
        function, line, column = row.get('function'), row.get('line'), row.get('column')
        category = row.get('category')
        if (not isinstance(function, str) or not function.isidentifier()
                or type(line) is not int or line <= 0 or type(column) is not int or column < 0
                or category not in ('BoolOp', 'UnaryOp', 'Constant', 'Call', 'Compare')
                or key != f"{row['path']}:{function}:{line}:{column}:{category}"):
            raise ValueError('invalid Boolean-value location/category')
        if row.get('result_kind') not in ('s32', 'u32', 's64', 'u64', 'f32', 'pred'):
            raise ValueError('invalid Boolean-value result kind')
        digest = row.get('specialized_expression_sha256')
        if (not isinstance(digest, str) or len(digest) != 64
                or any(c not in '0123456789abcdef' for c in digest)):
            raise ValueError('invalid Boolean-value expression digest')
        kernels = row.get('kernels')
        if (not isinstance(kernels, dict) or not kernels
                or any(not isinstance(name, str) or not name.isidentifier()
                       or type(count) is not int or count <= 0 for name, count in kernels.items())):
            raise ValueError('invalid Boolean-value use inventory')
'''
 (HERE/'candidate-modules.py').write_text(text)

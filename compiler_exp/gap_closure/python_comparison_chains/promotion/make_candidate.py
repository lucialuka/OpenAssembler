"""Stage original short-circuit chains with existing pairwise scalar type semantics."""
from pathlib import Path
HERE=Path(__file__).resolve().parent
def once(text,old,new):
 assert text.count(old)==1,old;return text.replace(old,new,1)
if __name__=='__main__':
 text=(HERE/'baseline-pyfront.py').read_text()
 text=once(text,'''        if isinstance(test, ast.Compare):
            pred, inverted = self._compare(test, branch=True)''','''        if isinstance(test, ast.Compare) and len(test.ops) > 1:
            return self._chain_jump(test, label, when)
        if isinstance(test, ast.Compare):
            pred, inverted = self._compare(test, branch=True)''')
 anchor='    def _mov(self, dst: _Reg, src: _Reg, kind: str) -> None:\n'
 text=once(text,anchor,'''    def _chain_jump(self, test: ast.Compare, label: str, when: bool) -> None:
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

'''+anchor)
 text=once(text,'''        a, ka, b, kb = self._pair(node.left, node.comparators[0], node)
        if ka != kb:''','''        a, ka, b, kb = self._pair(node.left, node.comparators[0], node)
        return self._compare_values(a, ka, b, kb, node.ops[0], node, branch)

    def _compare_values(self, a, ka, b, kb, operator, node, branch):
        """One typed relation, shared by ordinary and chained comparisons."""
        if ka != kb:''')
 text=once(text,'        op = type(node.ops[0])\n','        op = type(operator)\n')
 (HERE/'candidate-pyfront.py').write_text(text)
 text=(HERE/'baseline-modules.py').read_text()
 text=once(text,'        self.calls = {}\n','        self.calls = {}\n        self.comparisons = {}\n')
 anchor='    def contract(self):\n'
 method='''    def note_comparison(self, names, function, expression):
        relative = names.module_path.relative_to(self.root).as_posix()
        key = f'{relative}:{function.name}:{expression.lineno}:{expression.col_offset}'
        digest = hashlib.sha256(ast.dump(expression, include_attributes=False).encode()).hexdigest()
        row = self.comparisons.setdefault(key, dict(path=relative, function=function.name,
            line=expression.lineno, column=expression.col_offset,
            operators=[type(operator).__name__ for operator in expression.ops],
            specialized_expression_sha256=digest, kernels={}))
        if row['specialized_expression_sha256'] != digest:
            self.fail('comparison chain specialization differs at original location', expression)
        row['kernels'][self.kernel] = row['kernels'].get(self.kernel, 0) + 1

'''
 text=once(text,anchor,method+anchor)
 text=once(text,"and not getattr(self, 'early_returns', False):","and not getattr(self, 'early_returns', False) and not self.comparisons:")
 text=once(text,"        if self.static_values.records:\n            result['static_values']", "        if self.comparisons:\n            result['comparison_chains'] = {key:self.comparisons[key] for key in sorted(self.comparisons)}\n        if self.static_values.records:\n            result['static_values']")
 text=once(text,"obj.get('static_iterations')):","obj.get('static_iterations') or obj.get('comparison_chains')):")
 text=once(text,"    return obj\n", "    if 'comparison_chains' in obj:\n        validate_comparisons(obj['comparison_chains'], paths)\n    return obj\n")
 text+='''\n\ndef validate_comparisons(records, paths):
    """Check original chain locations, typed expression identities and use counts."""
    if not isinstance(records, dict) or not records:
        raise ValueError('invalid comparison-chain inventory')
    allowed = {'Lt', 'LtE', 'Gt', 'GtE', 'Eq', 'NotEq'}
    for key, row in records.items():
        if not isinstance(row, dict) or row.get('path') not in paths:
            raise ValueError('invalid comparison-chain source')
        function, line, column = row.get('function'), row.get('line'), row.get('column')
        if (not isinstance(function, str) or not function.isidentifier()
                or type(line) is not int or line <= 0
                or type(column) is not int or column < 0
                or key != f"{row['path']}:{function}:{line}:{column}"):
            raise ValueError('invalid comparison-chain location')
        operators = row.get('operators')
        if (not isinstance(operators, list) or len(operators) < 2
                or any(not isinstance(operator, str) or operator not in allowed for operator in operators)):
            raise ValueError('invalid comparison-chain operators')
        digest = row.get('specialized_expression_sha256')
        if (not isinstance(digest, str) or len(digest) != 64
                or any(c not in '0123456789abcdef' for c in digest)):
            raise ValueError('invalid comparison-chain expression digest')
        kernels = row.get('kernels')
        if (not isinstance(kernels, dict) or not kernels
                or any(not isinstance(name, str) or not name.isidentifier()
                       or type(count) is not int or count <= 0 for name, count in kernels.items())):
            raise ValueError('invalid comparison-chain use inventory')
'''
 (HERE/'candidate-modules.py').write_text(text)

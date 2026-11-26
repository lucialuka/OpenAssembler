"""Stage logical helper predicates without changing any physical launch/storage ABI."""
from pathlib import Path

HERE = Path(__file__).resolve().parent


def replace(text, old, new):
    assert text.count(old) == 1, old
    return text.replace(old, new)


if __name__ == '__main__':
    text = (HERE/'baseline-pyfront.py').read_text()
    text = replace(text,
        'def _resolve_type(ann: ast.expr, names: _Names) -> Scalar | ArrayType:',
        'def _resolve_type(ann: ast.expr, names: _Names, *, allow_boolean: bool = False) -> Scalar | ArrayType:')
    text = replace(text,
        '    if not isinstance(ann, ast.Call) and r in ("pf.f32", "pf.i32", "pf.u32", "pf.i64", "pf.u64"):',
        '    if allow_boolean and not isinstance(ann, ast.Call) and r == "bool":\n'
        '        return T.Scalar("pred")\n'
        '    if not isinstance(ann, ast.Call) and r in ("pf.f32", "pf.i32", "pf.u32", "pf.i64", "pf.u64"):')
    (HERE/'candidate-pyfront.py').write_text(text)
    text = (HERE/'baseline-helpers.py').read_text()
    text = replace(text, 'c._resolve_type(arg.annotation, self.names)',
                   'c._resolve_type(arg.annotation, self.names, allow_boolean=True)')
    text = replace(text, 'c._resolve_type(fn.returns, self.names)',
                   'c._resolve_type(fn.returns, self.names, allow_boolean=True)')
    (HERE/'candidate-helpers.py').write_text(text)
    text = (HERE/'baseline-modules.py').read_text()
    text = replace(text,
        "and not self.comparisons and not self.boolean_values:\n            return None",
        "and not self.comparisons and not self.boolean_values and not any(\n"
        "                'pred' in row['inputs'] or row['output'] == 'pred'\n"
        "                for row in self.calls.values()):\n            return None")
    text = replace(text,
        "        if not isinstance(helper['kernels'], dict) or not helper['kernels'] or any(type(n) is not int or n <= 0 for n in helper['kernels'].values()):",
        "        kinds = {'s32', 'u32', 's64', 'u64', 'f32', 'pred'}\n"
        "        if (not isinstance(helper.get('inputs'), list)\n"
        "                or any(not isinstance(kind, str) or kind not in kinds for kind in helper['inputs'])\n"
        "                or not isinstance(helper.get('output'), str) or helper['output'] not in kinds):\n"
        "            raise ValueError('invalid Python helper scalar signature')\n"
        "        if not isinstance(helper['kernels'], dict) or not helper['kernels'] or any(type(n) is not int or n <= 0 for n in helper['kernels'].values()):")
    (HERE/'candidate-modules.py').write_text(text)
    print('Staged exact predicate-only helper candidate in three modules')

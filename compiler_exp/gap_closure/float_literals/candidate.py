"""Generate a standalone parser; retain existing operand classes when staging."""
import ast
import builtins
from pathlib import Path

HERE=Path(__file__).resolve().parent


def generate():
    source=(HERE/'baseline-parse.py').read_text()
    source=source.replace('import re\n','import re\nimport math\nimport struct\n',1)
    anchor='def parse_operand(t: str) -> Operand:'
    assert source.count(anchor)==1
    helpers=(HERE/'literal_helpers.py').read_text().split('\n',1)[1]
    source=source.replace(anchor,helpers+'\n\n'+anchor,1)
    old='    return Var(t)\n\n\ndef parse_instr'
    assert source.count(old)==1
    source=source.replace(old,'    if _DECIMAL_FLOAT.fullmatch(t):\n        return _decimal_float(t)\n'+old,1)
    old='    return Instr(op, tuple(mods), args, guard, line)'
    assert source.count(old)==1
    source=source.replace(old,'    args = _literal_operands(op,mods,args,line)\n'+old,1)
    return source


def install():
    import openassembler.ptx.parse as parser
    # Replacing the existing class objects would invalidate consumers that
    # already imported them. Only install new helpers and changed functions.
    tree=ast.parse((HERE/'candidate-parse.py').read_text())
    names={'_DECIMAL_FLOAT','_decimal_float','_literal_f32','_literal_operands','parse_operand','parse_instr'}
    nodes=[node for node in tree.body if isinstance(node,(ast.Import,ast.ImportFrom))
           or isinstance(node,ast.FunctionDef)and node.name in names
           or isinstance(node,ast.Assign)and any(isinstance(target,ast.Name)and target.id in names for target in node.targets)]
    exec(builtins.compile(ast.Module(body=nodes,type_ignores=[]),str(HERE/'candidate-parse.py'),'exec'),parser.__dict__)


if __name__=='__main__':(HERE/'candidate-parse.py').write_text(generate())

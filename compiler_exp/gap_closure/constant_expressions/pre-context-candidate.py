"""Generate self-contained parser source and stage functions without class churn."""
import ast
import builtins
from pathlib import Path

HERE=Path(__file__).resolve().parent


def generate():
    source=(HERE/'baseline-parse.py').read_text()
    anchor='def parse_operand(t: str) -> Operand:'
    helper=(HERE/'expression_helpers.py').read_text().split('\n',1)[1]
    source=source.replace(anchor,helper+'\n\n'+anchor,1)
    old='    if "|" in t:\n'
    assert source.count(old)==1
    source=source.replace(old,"    if re.match(r'(?:[0-9(]|\\.[0-9]|[~!+-]\\s*(?:[0-9(]|\\.[0-9]|[~!+-]))',t):\n        return _constant_expression(t)\n"+old,1)
    return source


def install():
    import openassembler.ptx.parse as parser
    tree=ast.parse((HERE/'candidate-parse.py').read_text())
    functions={'_constant_expression','parse_operand'};assignments={'_CONSTANT_TOKEN','_CONSTANT_PRECEDENCE'}
    nodes=[node for node in tree.body if isinstance(node,ast.FunctionDef)and node.name in functions
           or isinstance(node,ast.Assign)and any(isinstance(target,ast.Name)and target.id in assignments for target in node.targets)]
    exec(builtins.compile(ast.Module(body=nodes,type_ignores=[]),str(HERE/'candidate-parse.py'),'exec'),parser.__dict__)


if __name__=='__main__':(HERE/'candidate-parse.py').write_text(generate())

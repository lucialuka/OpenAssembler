"""Independent tree/rational semantics plus preserved raw toolkit controls."""
import hashlib
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
sys.path[:0]=[str(ROOT),str(HERE)]
from candidate import install
install()
from reference import generated
from openassembler.ptx.parse import parse_operand,parse_module,ImmF,ImmI


if __name__=='__main__':
    records=[];digest=hashlib.sha256();count=0
    for case in generated():
        value=parse_operand(case['expression'])
        assert isinstance(value,ImmF if case['kind']=='f'else ImmI)
        actual=value.bits if isinstance(value,ImmF)else value.value&((1<<64)-1)
        assert actual==case['bits'],(case['expression'],hex(actual),hex(case['bits']))
        digest.update(json.dumps(case,sort_keys=True).encode());count+=1
    records.append(dict(kind='independent-typed-tree-rational-semantics',passed=True,cases=count,corpus_sha256=digest.hexdigest()))
    for path in(ROOT/'tests/data').rglob('*.ptx'):
        # Existing parse coverage is separate from semantic expression checks.
        parse_module(path.read_text())
    for token in('%p','!%p','.L_x_0','-RZ','%f1','smem','$label'):
        value=parse_operand(token);assert getattr(value,'name',None)==token
    records.append(dict(kind='preserved-register-symbol-and-predicate-spellings',passed=True,cases=7))
    deep='('*140+'1.0'+')'*140
    for text in(deep,'1?1:'*140+'1','1?1:'*1000+'1','1.0+'*2100+'1.0','1**2','(.s64)1.0','0f3f800000+1.0','1.0+1','1 ? 1.5 : 2.5'):
        try:parse_operand(text)
        except SyntaxError:pass
        else:raise AssertionError('invalid or excessive expression accepted: '+text[:40])
    records.append(dict(kind='invalid-types-grammar-and-explicit-recursion-budget',passed=True,cases=9))
    result=dict(schema=1,passed=True,results=records,parser_sha256=hashlib.sha256((HERE/'candidate-parse.py').read_bytes()).hexdigest(),
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                scope='Independent generated typed AST semantics, rational per-operation binary64 rounding, preserved symbols and explicit parser/type bounds; no hardware claim')
    (HERE/'candidate-offline.json').write_text(json.dumps(result,indent=2)+'\n');print(result)

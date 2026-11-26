"""Explicit constant grammar, source diagnostics, mutation and resource boundaries."""
import hashlib,json,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4];HERE=Path(__file__).resolve().parent;sys.path[:0]=[str(ROOT),str(HERE)]
from candidate import install
install()
from fixtures import configurations,write_tree
from openassembler.pyfront import compile_file,NotSupported
MUTATIONS=[
 ('call','GAIN: Frozen[int] = 3','GAIN: Frozen[int] = side_effect()','unsupported expression'),
 ('unknown','GAIN: Frozen[int] = 3','GAIN: Frozen[int] = UNKNOWN','explicit Final'),
 ('rebind','GAIN: Frozen[int] = 3','GAIN: Frozen[int] = 3\nGAIN = 7','rebound'),
 ('annotated_rebind','GAIN: Frozen[int] = 3','GAIN: Frozen[int] = 3\nGAIN: int = 7','rebound'),
 ('conditional_rebind','GAIN: Frozen[int] = 3','GAIN: Frozen[int] = 3\nif True:\n    GAIN=7','rebound'),
 ('delete','GAIN: Frozen[int] = 3','GAIN: Frozen[int] = 3\ndel GAIN','rebound'),
 ('tuple_mutate','COEFFICIENTS: Frozen[tuple[int,int,int]] = (3,5,7)','COEFFICIENTS: Frozen[tuple[int,int,int]] = (3,5,7)\nCOEFFICIENTS[0]=19','mutated'),
 ('bad_type','GAIN: Frozen[int] = 3','GAIN: Frozen[int] = 3.0','annotation'),
 ('bad_bool','GAIN: Frozen[int] = 3','GAIN: Frozen[int] = True','annotation'),
 ('bad_tuple_type','COEFFICIENTS: Frozen[tuple[int,int,int]] = (3,5,7)','COEFFICIENTS: Frozen[tuple[int,int,int]] = (3,5.0,7)','annotation'),
 ('bad_tuple_arity','COEFFICIENTS: Frozen[tuple[int,int,int]] = (3,5,7)','COEFFICIENTS: Frozen[tuple[int,int]] = (3,5,7)','arity'),
 ('list','COEFFICIENTS: Frozen[tuple[int,int,int]] = (3,5,7)','COEFFICIENTS: Frozen = [3,5,7]','unsupported expression'),
 ('dict','GAIN: Frozen[int] = 3','GAIN: Frozen = {3:5}','unsupported expression'),
 ('missing_value','GAIN: Frozen[int] = 3','GAIN: Frozen[int]','needs a value'),
 ('division_zero','GAIN: Frozen[int] = 3','GAIN: Frozen[int] = 3//0','invalid compile-time'),
 ('float_overflow','SCALE: Frozen[float] = (1.0 / 8.0) * 4.0','SCALE: Frozen[float] = 1e308*1e308','nonfinite'),
 ('shift_negative','GAIN: Frozen[int] = 3','GAIN: Frozen[int] = 1<<-1','shift'),
 ('shift_huge','GAIN: Frozen[int] = 3','GAIN: Frozen[int] = 1<<999999999','shift'),
 ('integer_huge','GAIN: Frozen[int] = 3','GAIN: Frozen[int] = (1<<2047)*(1<<2047)','2048 bits'),
 ('type_shadow','GAIN: Frozen[int] = 3','int = 7\nGAIN: Frozen[int] = 3','type is rebound'),
 ('power_huge','GAIN: Frozen[int] = 3','GAIN: Frozen[int] = 2**999999','power exponent'),
 ('cyclic','GAIN: Frozen[int] = 3','GAIN: Frozen[int] = BIAS','cyclic'),
]
BODY_MUTATIONS=[
 ('dynamic_tuple','coefficients[1]','coefficients[x]','index must be static'),
 ('tuple_bounds','coefficients[1]','coefficients[99]','invalid compile-time'),
 ('tuple_float_index','coefficients[1]','coefficients[1.0]','integer'),
 ('tuple_slice_zero','configuration.COEFFICIENTS[1:][0]','configuration.COEFFICIENTS[::0][0]','invalid compile-time'),
 ('bare_tuple','x*u64(coefficients[1])','x*u64(coefficients)','tuple value must'),
 ('module_mutation','raise RuntimeError(\'helper module must not execute\')','configuration.GAIN=19','mutated'),
 ('import_rebind','raise RuntimeError(\'helper module must not execute\')','gain=19','rebound'),
]

if __name__=='__main__':
 rows=[]
 with tempfile.TemporaryDirectory(prefix='oa-Final-negative-')as temp:
  for name,old,new,reason in MUTATIONS+BODY_MUTATIONS:
   root=Path(temp).resolve()/name;case=next(configurations());source=write_tree(root,case)
   file=root/'library'/('helpers.py'if(name,old,new,reason)in BODY_MUTATIONS else'constants.py')
   text=file.read_text();assert old in text;file.write_text(text.replace(old,new,1))
   if name=='cyclic':file.write_text(file.read_text().replace('BIAS: types.Final[int] = -(1 << 32) + 7','BIAS: types.Final[int] = GAIN'))
   try:compile_file(source)
   except NotSupported as error:
    assert reason in str(error),(name,reason,str(error))
    assert error.path and error.lineno,(name,str(error))
    rows.append(dict(name=name,rejected=True,message=error.msg,path=Path(error.path).relative_to(root).as_posix(),line=error.lineno))
   else:raise AssertionError('unvalidated constant admitted: '+name)
 result=dict(schema=1,passed=True,results=rows,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),candidate_sha256=hashlib.sha256((HERE/'static_values.py').read_bytes()).hexdigest(),scope='Original Final declarations and use-site diagnostics for call/capture/rebind/mutation/type/bounds/divide/float/shift/bit limits, cyclic imported values and tuple use/index/slice restrictions; no hardware claim')
 (HERE/'negative-controls.json').write_text(json.dumps(result,indent=2)+'\n');print('Passed',len(rows),'negative controls')

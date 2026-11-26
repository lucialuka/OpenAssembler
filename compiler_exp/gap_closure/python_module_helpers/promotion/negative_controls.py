"""Source graph edge cases, exact imported locations and snapshot integrity."""
import ast,hashlib,json,shutil,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4];HERE=Path(__file__).resolve().parent;sys.path[:0]=[str(ROOT),str(HERE)]
from candidate import install
install()
from fixtures import configurations,write_tree
from openassembler.pyfront import compile_file,compile_kernel,find_kernel,NotSupported
from openassembler.pyfront.modules import SourceGraph,contract_from_ptx,validate_sources

MUTATIONS=[
 ('recursive','arithmetic/base.py','return x * y','return signed_product(x,y)','recursive device helper'),
 ('cross_recursive','arithmetic/base.py','return x * y','return back(x,y)','recursive device helper'),
 ('capture','arithmetic/base.py','return x * y','return x * GLOBAL','captured'),
 ('host_call','arithmetic/base.py','return x * y','return print(x)','not supported'),
 ('memory','arithmetic/base.py','return x * y','return x[0]','access memory'),
 ('early_return','arithmetic/transforms.py','result = x','if x < y:\n        return x\n    result = x','terminal return'),
 ('mixed_return','arithmetic/base.py','return x * y','return 1.0','annotated'),
 ('missing_module','arithmetic/transforms.py','from .base import signed_product as product','from .missing import signed_product as product','not found'),
 ('missing_function','arithmetic/transforms.py','from .base import signed_product as product','from .base import missing as product','missing'),
 ('untyped','arithmetic/base.py','def signed_product(x: i64, y: i64) -> i64:','def signed_product(x, y: i64) -> i64:','annotation'),
 ('not_device','arithmetic/base.py','@device\ndef signed_product','def signed_product','not a typed'),
 ('default','arithmetic/base.py','x: i64, y: i64','x: i64, y: i64 = 1','plain positional'),
 ('relative_escape','arithmetic/transforms.py','from .base import','from ...base import','escapes'),
 ('local_shadow','arithmetic/transforms.py','result = product(result, y)','product = result\n        result = product(result, y)','local value'),
 ('keyword','entry.py','repeated_product(a[i], b[i])','repeated_product(x=a[i], y=b[i])','keyword'),
 ('bad_arity','entry.py','repeated_product(a[i], b[i])','repeated_product(a[i])','expects'),
 ('bad_type','entry.py','repeated_product(a[i], b[i])','repeated_product(float(1), b[i])','expected'),
 ('rebound_import','entry.py','raise RuntimeError(\'static compilation must not execute the entry module\')','repeated_product = 7','rebound'),
 ('deleted_import','entry.py','raise RuntimeError(\'static compilation must not execute the entry module\')','del repeated_product','rebound'),
 ('conditional_rebind','entry.py','raise RuntimeError(\'static compilation must not execute the entry module\')','if True:\n    repeated_product = 7','rebound'),
 ('module_attribute','entry.py','raise RuntimeError(\'static compilation must not execute the entry module\')','transforms.unsigned_step = 7','mutated'),
 ('conditional_import','entry.py','from arithmetic.transforms import signed_step as repeated_product','if True:\n    from arithmetic.transforms import signed_step as repeated_product','not supported'),
 ('star_import','entry.py','from arithmetic.transforms import signed_step as repeated_product','from arithmetic.transforms import *','not supported'),
 ('bad_annotation_alias','arithmetic/base.py','from openassembler.pyfront import device, i64, u64, f32','from openassembler.pyfront import device, i64, u64, f32\ni64 = 7','annotation'),
 ('string_annotation','arithmetic/base.py','x: i64, y: i64','x: "Unknown", y: i64','annotation'),
 ('nested_import_alias','arithmetic/base.py','return x * y','import math as i64\n    return x * y','scalar assignments'),
]

def run(root):
 records=[];base=next(configurations())
 for name,file,old,new,reason in MUTATIONS:
  folder=root.resolve()/name;source=write_tree(folder,base);path=folder/file;text=path.read_text();assert old in text;path.write_text(text.replace(old,new,1))
  if name=='cross_recursive':
   text=path.read_text();path.write_text('from .transforms import signed_step as back\n'+text)
  try:compile_file(source)
  except NotSupported as error:
   assert reason.lower()in str(error).lower(),(name,reason,str(error))
   assert error.path and error.lineno,(name,str(error))
   if name in('capture','memory','untyped','bad_annotation_alias','string_annotation'):
    assert Path(error.path).resolve()==path.resolve() and error.lineno>0,(name,error.path,path)
   records.append(dict(name=name,rejected=True,message=error.msg,path=Path(error.path).relative_to(folder).as_posix(),line=error.lineno))
  else:raise AssertionError('unvalidated graph admitted: '+name)
 # Stable references are reparsed, and changed helper bytes change the graph identity.
 folder=root/'snapshot';source=write_tree(folder,base);ref=find_kernel(source,'signed');before=compile_kernel(ref)
 helper=folder/'arithmetic/base.py';helper.write_text(helper.read_text().replace('return x * y','return x + y',1));after=compile_kernel(ref);assert before!=after
 try:validate_sources(source,contract_from_ptx(before))
 except ValueError:records.append(dict(name='changed_dependency',rejected=True))
 else:raise AssertionError('changed dependency accepted')
 # Symlinked modules and file/package ambiguity cannot silently select another source.
 for name in('symlink','ambiguous'):
  folder=root.resolve()/name;source=write_tree(folder,base);helper=folder/'arithmetic/base.py'
  if name=='symlink':
   original=folder/'outside.py';helper.rename(original);helper.symlink_to(original)
  else:(folder/'arithmetic/base').mkdir()
  try:compile_file(source)
  except NotSupported as error:records.append(dict(name=name,rejected=True,message=error.msg))
  else:raise AssertionError('ambiguous/symlink source admitted')
 return records

if __name__=='__main__':
 with tempfile.TemporaryDirectory(prefix='oa-module-graph-negative-')as temp:rows=run(Path(temp))
 result=dict(schema=1,passed=True,results=rows,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),module_sha256=hashlib.sha256((HERE/'modules.py').read_bytes()).hexdigest(),scope='Original dependency/source lines for helper diagnostics, unsupported imports/types/effects/recursion/rebinding, fresh KernelRef snapshots, changed source identity, symlink and ambiguity rejection; no hardware claim')
 (HERE/'negative-controls.json').write_text(json.dumps(result,indent=2)+'\n');print('Passed',len(rows),'negative/snapshot controls')

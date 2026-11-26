"""Next G05/G06 hypothesis: original static tuple loop to bounded source expansion."""
import ast,copy,hashlib,json,struct,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;sys.path[:0]=[str(ROOT)]
from openassembler.pyfront import compile_file,NotSupported
from openassembler.pyfront.static_values import StaticValues
from openassembler.pyfront import compile as c
from openassembler.pyfront.modules import SourceGraph
from openassembler import build
from openassembler.ptx.parse import parse_module
from openassembler.ptx.interp import Interp,Launch as PLaunch
from openassembler.emu.machine import Memory,Machine,Launch
from openassembler.emu.program import load_listing
from tests.backend_harness import listing
if __name__=='__main__':
 source=HERE/'source.py';rows=[];baseline=None
 try:compile_file(source)
 except NotSupported as error:baseline=dict(rejected=True,error=str(error))
 assert baseline
 tree=ast.parse(source.read_text());names=c._Names(tree);graph=SourceGraph(source,tree,names);info=graph.modules[graph.entry]
 constants=graph.static_values
 class Unroll(ast.NodeTransformer):
  def visit_For(self,node):
   value=constants.evaluate(node.iter,info,required=True)
   if type(value)is not tuple or len(value)>64 or not isinstance(node.target,ast.Name)or node.orelse:raise ValueError('candidate loop requires finite tuple/name and no else')
   if any(isinstance(n,(ast.Break,ast.Continue,ast.Return,ast.For,ast.While))for st in node.body for n in ast.walk(st)):raise ValueError('candidate loop control flow is unvalidated')
   result=[]
   for item in value:
    if type(item)is not int or not -(1<<31)<=item<(1<<31):raise ValueError('candidate loop target requires i32 literal')
    assignment=ast.copy_location(ast.Assign(targets=[copy.deepcopy(node.target)],value=ast.Constant(item)),node)
    result.append(assignment);result.extend(copy.deepcopy(node.body))
   return result
 revised=ast.fix_missing_locations(Unroll().visit(copy.deepcopy(tree)));normalized=ast.unparse(revised)+'\n';(HERE/'normalized.py').write_text(normalized)
 with tempfile.TemporaryDirectory(prefix='oa-static-loop-')as temp:
  path=Path(temp)/'normalized.py';path.write_text(normalized);text,entries=compile_file(path);assert entries==['evaluate'];(HERE/'candidate.ptx').write_text(text);module=parse_module(text)
  mask=(1<<64)-1;data=[0,1,mask,1<<63,3,7]+[(i*0x123456789abcdef)&mask for i in range(122)]
  guard=b'\xa5'*64;ap,out=0x100010000,0x100000000;before=guard+struct.pack('<128Q',*data)+guard
  wanted=guard+struct.pack('<128Q',*([((3*x-5)*x+7)&mask if i<91 else 0x13579bdf2468ace0 for i,x in enumerate(data)]))+guard
  for level in('safe','mined','sched','opt'):
   code,labels,words,_,regs=build.generate(module.kernels['evaluate'],'sm_80',level);assert words
   for engine in('ptx','normal','poisoned'):
    memory=Memory();memory.write_bytes(ap-64,before);memory.write_bytes(out-64,guard+struct.pack('<128Q',*([0x13579bdf2468ace0]*128))+guard)
    if engine=='ptx':Interp(module,'evaluate',PLaunch(block=(128,1,1),params={'a':ap,'out':out,'n':91}),memory).run()
    else:Machine(load_listing(listing(code,labels)),Launch(block=(128,1,1),params=struct.pack('<QQI',ap,out,91),poison=0xdeadbeef if engine=='poisoned'else None),memory).run()
    assert bytes(memory.read(out-64+i,1)for i in range(len(wanted)))==wanted and bytes(memory.read(ap-64+i,1)for i in range(len(before)))==before
    rows.append(dict(level=level,engine=engine,passed=True))
 result=dict(schema=1,passed=True,baseline=baseline,executions=len(rows),results=rows,source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest()for p in(HERE/'source.py',HERE/'normalized.py',Path(__file__))},scope='Isolated finite tuple-loop expansion to a separately normalized source tree; independent modulo64 polynomial, full guarded output/tail/unchanged input across four levels and three engines. Original source graph/control/diagnostic/remote identity, heterogeneous values and break/continue/nesting/else remain unvalidated; no production or hardware claim')
 (HERE/'candidate-offline.json').write_text(json.dumps(result,indent=2)+'\n');print('Recorded original static-loop rejection; passed',len(rows),'normalized executions')

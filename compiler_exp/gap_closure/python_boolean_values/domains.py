"""Independent exact operand bits and truth flags across numeric scalar domains."""
import hashlib,json,struct,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;sys.path[:0]=[str(ROOT),str(HERE)]
from candidate import install
install()
from openassembler.pyfront import compile_file
from openassembler import build,oracle
from openassembler.ptx.parse import parse_module
from openassembler.ptx.interp import Interp,Launch as PLaunch
from openassembler.emu.machine import Memory,Machine,Launch
from openassembler.emu.program import load_listing
from tests.backend_harness import listing
if __name__=='__main__':
 rows=[];cases=[];artifacts={};guard=b'\xa5'*64;HERE.joinpath('oracle').mkdir(exist_ok=True)
 for dtype in('i32','u32','i64','u64','f32'):
  width=8 if dtype.endswith('64')else 4;mask=(1<<(8*width))-1
  if dtype=='f32':
   first=[0,0x80000000,0x7fc12345,0xffc54321,0x7f800000,0xff800000,1,0x80000001];second=[0x80000000,0,0x3f800000,0xbf800000,0,0x80000000,0x7fc54321,0x40400000]
   truth=lambda bits:bool(struct.unpack('<f',bits.to_bytes(4,'little'))[0])
  else:first=[0,1,mask,1<<(width*8-1),(1<<(width*8-1))-1,3,7,19];second=[mask,0,1,3,0,7,19,0];truth=lambda bits:bool(bits)
  raw_a=b''.join(x.to_bytes(width,'little')for x in first);raw_b=b''.join(x.to_bytes(width,'little')for x in second)
  and_bits=[b if truth(a)else a for a,b in zip(first,second)];or_bits=[a if truth(a)else b for a,b in zip(first,second)];flags=[int(not truth(a))+2*int(truth(c))+4*int(truth(d))for a,c,d in zip(first,and_bits,or_bits)]
  wanted_and=b''.join(x.to_bytes(width,'little')for x in and_bits);wanted_or=b''.join(x.to_bytes(width,'little')for x in or_bits);wanted_flags=struct.pack('<8I',*flags)
  source=f'''from openassembler.pyfront import *
@kernel
def k(a:Array({dtype}),b:Array({dtype}),left:Array({dtype}),right:Array({dtype}),flags:Array(i32)):
    i=tid_x()
    conjunction=a[i] and b[i]
    disjunction=a[i] or b[i]
    inverse=not bool(a[i])
    left[i]=conjunction
    right[i]=disjunction
    flags[i]=i32(inverse)+i32(bool(conjunction))*2+i32(bool(disjunction))*4
'''
  path=HERE/'oracle'/(dtype+'.py');path.write_text(source);text,_=compile_file(path);ptx=path.with_suffix('.ptx');ptx.write_text(text);binary=path.with_suffix('.cubin');tool=oracle.run_tool('ptxas','-arch=sm_80','-o',oracle.tool_path(binary),oracle.tool_path(ptx),check=False);path.with_suffix('.log').write_text(tool.stdout+tool.stderr);assert tool.returncode==0,tool.stderr;path.with_suffix('.sass').write_text(oracle.disasm_text(binary));module=parse_module(text)
  expected=[raw_a,raw_b,wanted_and,wanted_or,wanted_flags];addresses=[0x100010000,0x100020000,0x100030000,0x100040000,0x100050000]
  for level in('safe','mined','sched','opt'):
   code,labels,words,_,regs=build.generate(module.kernels['k'],'sm_80',level)
   for engine in('ptx','normal','poisoned'):
    memory=Memory()
    for j,(address,data)in enumerate(zip(addresses,expected)):memory.write_bytes(address-64,guard+(data if j<2 else bytes(len(data)))+guard)
    if engine=='ptx':Interp(module,'k',PLaunch(block=(8,1,1),params=dict(zip(('a','b','left','right','flags'),addresses))),memory).run()
    else:Machine(load_listing(listing(code,labels)),Launch(block=(8,1,1),params=struct.pack('<5Q',*addresses),poison=0xdeadbeef if engine=='poisoned'else None),memory).run()
    for address,data in zip(addresses,expected):assert bytes(memory.read(address-64+j,1)for j in range(len(data)+128))==guard+data+guard,(dtype,level,engine,address)
    rows.append(dict(dtype=dtype,level=level,engine=engine,passed=True))
  cases.append(dict(dtype=dtype,source=source,input_a_hex=raw_a.hex(),input_b_hex=raw_b.hex(),expected_and_hex=wanted_and.hex(),expected_or_hex=wanted_or.hex(),expected_flags=flags))
  for p in HERE.joinpath('oracle').glob(dtype+'.*'):artifacts[str(p.relative_to(HERE))]=hashlib.sha256(p.read_bytes()).hexdigest()
 report=dict(schema=1,passed=True,executions=60,results=rows,cases=cases,artifact_hashes=artifacts,source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest()for p in(HERE/'candidate.py',HERE/'candidate-pyfront.py',Path(__file__))},scope='Five homogeneous operand-returning scalar domains with exact independent raw selected bits, FP32 NaN payload/infinity/signed-zero/subnormal truth, explicit Boolean integer conversion and full unchanged input/output/flag/guard bytes at four PTX/normal/poisoned levels plus actual ptxas controls; no GPU, Boolean ABI/arrays, general mixed-type or production claim')
 (HERE/'domains.json').write_text(json.dumps(report,indent=2)+'\n');print('Passed 60 operand-bit/special-value truth executions and five toolkit controls')

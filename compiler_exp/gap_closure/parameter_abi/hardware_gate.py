"""Mixed32/64 parameter ABI with pointer metadata and independent A100 bytes."""
from dataclasses import asdict
import ctypes as C
import hashlib
import json
from pathlib import Path
import struct
import sys
from openassembler import build
from openassembler.gpu.cuda_driver import Driver
from openassembler.native_ownership import validate
HEADER='.version 8.7\n.target sm_80\n.address_size 64\n'
BODY='''.reg .u32 %i,%n,%v; .reg .u64 %out,%in,%off,%p; .reg .pred %end;
ld.param.u32 %n,[n];ld.param.u64 %out,[output];ld.param.u64 %in,[input];
mov.u32 %i,%tid.x;setp.ge.u32 %end,%i,%n;@%end bra done;
mul.wide.u32 %off,%i,4;add.u64 %p,%in,%off;ld.global.u32 %v,[%p];
add.u64 %p,%out,%off;st.global.u32 [%p],%v;done:ret;}
'''


def run(destination):
 root=Path(destination);root.mkdir(parents=True,exist_ok=True);records=[]
 driver=Driver();driver.open();guard=bytes([0xa5])*256;size=96*4
 expected=struct.pack('<96I',*[((i*0x9e3779b9)^0x8147ec48)&0xffffffff for i in range(96)])
 try:
  for i,(dtype,annotation) in enumerate([('u64','.ptr .global .align 16'),('u64','.ptr .global'),('b64','.ptr .align 8'),('b64','.ptr')]):
   name='pointer_'+str(i)
   text=HEADER+f'.entry {name}(.param .u32 n,.param .{dtype} {annotation} output,.param .u64 .ptr .global .align 8 input){{'+BODY
   path=root/(name+'.ptx');path.write_text(text)
   for level in ('safe','mined','sched','opt'):
    output=root/(name+'-'+level);result=build.compile_paths(path,output,build.Options(opt_level=level,verify=True))
    if result.rc:raise ValueError(result)
    entry=json.loads((output/'manifest.json').read_text())['entries'][0];assert entry['kernels'][0]['verify']=='ok'
    validate(entry,(output/entry['cubin']).read_bytes())
    for compiler,key in [('oa','cubin'),('ptxas','reference')]:
     raw=(output/entry[key]).read_bytes();module=driver.load(raw);pointers=[]
     try:
      fn=driver.function(module,name);pointers=[driver.alloc(size+512) for _ in range(2)]
      driver.htod(pointers[0],guard+expected+guard);driver.htod(pointers[1],guard+b'\x3c'*size+guard)
      driver.launch(fn,(1,1,1),(96,1,1),[C.c_uint32(67),C.c_uint64(pointers[1]+256),C.c_uint64(pointers[0]+256)])
      actual=driver.dtoh(pointers[1],size+512);truth=guard+expected[:67*4]+b'\x3c'*(size-67*4)+guard
      assert actual==truth and driver.dtoh(pointers[0],size+512)==guard+expected+guard
      records.append({'name':name,'dtype':dtype,'annotation':annotation,'level':level,'compiler':compiler,'bytes':size,'passed':True,
       'binary_sha256':hashlib.sha256(raw).hexdigest(),'output_sha256':hashlib.sha256(actual).hexdigest()})
      (output/(compiler+'-output.bin')).write_bytes(actual)
     finally:
      for pointer in pointers:driver.free(pointer)
      driver.unload(module)
  report={'schema':1,'passed':True,'device':asdict(driver.info()),'python':sys.version,'records':records,
   'scope':'32 mixed-slot annotatedpointer native/ptxaslaunches; exactoutput/tails/guards/immutability, no generalPTXABI orperformanceclaim'}
  (root/'report.json').write_text(json.dumps(report,indent=2)+'\n')
 finally:driver.close()

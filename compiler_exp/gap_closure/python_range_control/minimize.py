"""Derive isolated instruction, constant-loop and runtime-bound toolkit controls."""
import hashlib,json,re,sys,zipfile
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
from probe import CASES
def signed(value):
 bits=value&0xffffffff;return bits-(1<<32)if bits&(1<<31)else bits
def wrapped(start,stop,step):
 target=17;count=total=0;counter=start
 for _ in range(8):
  if(counter>=stop if step>0 else counter<=stop):break
  target=counter;total+=target;count+=1
  if count==8:break
  counter=signed(counter+step)
 return[total,count,target]
if __name__=='__main__':
 report=json.loads((HERE/'candidate-a100-20261124T060602Z.json').read_text());archive=HERE/report['archive']['path'];assert hashlib.sha256(archive.read_bytes()).hexdigest()==report['archive']['sha256']
 with zipfile.ZipFile(archive)as retained:text=retained.read('baseline/safe/source.ptx').decode()
 header=text[:text.index('// openassembler-pyfront-types-v1')];bodies=[];rows=[]
 for name,start,stop,step in CASES:
  part=text[text.index('.visible .entry '+name):];part=part[:part.index('\n}\n')+3]
  for mode in('static_signed','static_unsigned','runtime_signed'):
   entry=mode+'_'+name;body=part.replace('.entry '+name+'(','.entry '+entry+'(',1)
   if mode=='static_unsigned':body=body.replace(' add.s32 ',' add.u32 ').replace(' sub.s32 ',' sub.u32 ')
   if mode=='runtime_signed':
    body=body.replace('.param .u64 out)','.param .u64 out, .param .s32 start, .param .s32 stop)',1)
    body,n=re.subn(r' mov.s32 %r5, -?\d+;',r' ld.param.s32 %r5, [start];',body);assert n==1
    body,n=re.subn(r' mov.s32 %r6, -?\d+;',r' ld.param.s32 %r6, [stop];',body);assert n==1
   values=list(range(start,stop,step));bodies.append(body);rows.append(dict(entry=entry,mode=mode,start=start,stop=stop,step=step,python_reference=[sum(values),len(values),values[-1]if values else 17],wrapped_reference=wrapped(start,stop,step)))
 for kind in('s32','u32'):
  entry='isolated_add_'+kind;bodies.append(f'''.visible .entry {entry}(.param .u64 out)
{{
 .reg .b64 %rd<3>;
 .reg .b32 %r<2>;
 ld.param.u64 %rd1,[out];
 mov.s32 %r1,2147483646;
 add.{kind} %r1,%r1,2;
 cvt.s64.s32 %rd2,%r1;
 st.global.s64 [%rd1],%rd2;
 st.global.s64 [%rd1+8],%rd2;
 st.global.s64 [%rd1+16],%rd2;
 ret;
}}
''');rows.append(dict(entry=entry,mode='isolated_add',python_reference=[-(1<<31)]*3,wrapped_reference=[-(1<<31)]*3))
 source=header+'\n'.join(bodies);(HERE/'minimized.ptx').write_text(source)
 result=dict(schema=1,source_sha256=hashlib.sha256(source.encode()).hexdigest(),results=rows,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),scope='Derived original baseline loop PTX with signed/unsigned increments, fixed/runtime bounds and isolated wrapping add controls; exact mathematical Python and explicit i32-wrap watchdog references remain distinct; no execution or ISA interpretation claim')
 (HERE/'minimized.json').write_text(json.dumps(result,indent=2)+'\n');print('Prepared',len(rows),'minimized controls')

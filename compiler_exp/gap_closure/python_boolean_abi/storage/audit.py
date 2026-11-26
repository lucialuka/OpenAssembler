"""Audit complete original Boolean storage outputs and exact native/source custody."""
import argparse,hashlib,json,sys,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4];HERE=Path(__file__).resolve().parent;sys.path[:0]=[str(ROOT),str(HERE)]
from fixtures import configurations,NAMES,raw_input,initial,initial_counter,references

def sha(data):return hashlib.sha256(data).hexdigest()
def exact(actual,expected,identity):assert actual==expected,identity
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('report',type=Path);parser.add_argument('output',type=Path);parser.add_argument('--candidate',action='store_true');args=parser.parse_args()
 if args.candidate:
  from candidate import install
  install()
 from openassembler import build
 from openassembler.verify import parameter_contract_from_ptx
 from openassembler.pyfront.buffers import contract_from_ptx as buffer_contract
 from openassembler.pyfront.modules import contract_from_ptx
 from openassembler.ptx.parse import parse_module
 from openassembler.container.cubin import text_words,regcount
 from openassembler.backend.storage import local_frame_inventory
 from openassembler.native_ownership import validate
 path=args.report.resolve();report=json.loads(path.read_text());assert report['passed']and report['error']is None
 replacements={name:HERE/('candidate-'+suffix+'.py')for suffix,name in json.loads((HERE/'inputs.json').read_text()).items()}
 for name,digest in report['source_hashes'].items():
  current=replacements[name]if args.candidate and name in replacements else ROOT/name
  if name in ('openassembler/kernels/sm80_gemm_dispatch.json','openassembler/kernels/sm80_rowwise_dispatch.json') and sha(current.read_bytes())!=digest:
   previous=json.loads((HERE.parent/'scalar/validation.json').read_text());assert previous['source_hashes'][name]==digest
   table=json.loads(current.read_text());policy_path=ROOT/table['source_continuity']['audit'];policy=json.loads(policy_path.read_text())
   assert policy['passed']and policy['counts']==dict(gemm_contracts=44,rowwise_contracts=5,retained_session_checks=98)and sha(policy_path.read_bytes())==table['source_continuity']['sha256']
   assert all(sha((ROOT/key).read_bytes())==value for key,value in policy['compiler_inputs_sha256'].items())
   continue
  exact(sha(current.read_bytes()),digest,('source',name))
 exact(report['harness_sha256'],sha((HERE/'run_modal.py').read_bytes()),'harness');archive_path=path.parent/report['archive']['path'];exact(sha(archive_path.read_bytes()),report['archive']['sha256'],'archive')
 cases={c['name']:c for c in configurations()};levels=('safe','mined','sched','opt');compilers=('ptxas','oa-safe','oa-mined','oa-sched','oa-opt')
 domain={(c,compiler,name,seed,flag,n)for c in cases for compiler in compilers for name in NAMES for seed in(73519,73520)for flag in(False,True)for n in(0,1,259,384)}
 keys=[tuple(row[k]for k in('configuration','compiler','entry','seed','flag','n'))for row in report['results']];assert len(keys)==len(set(keys))==len(domain)==4320 and set(keys)==domain
 explicit={(c,level,name,flag)for c in cases for level in levels for name in NAMES for flag in(False,True)}
 keys=[tuple(row[k]for k in('configuration','level','entry','flag'))for row in report['explicit_launch_results']];assert len(keys)==len(set(keys))==len(explicit)==432 and set(keys)==explicit
 native=[];count=0;binaries={};guard=b'\xa5'*256;bguard=bytes([0,1])*128
 with zipfile.ZipFile(archive_path)as archive:
  for configuration,case in cases.items():
   hashes={name:sha(text.encode())for name,text in case['files'].items()}
   for name,text in case['files'].items():exact(archive.read(configuration+'/original/'+name),text.encode(),('original',configuration,name))
   for level in levels:
    prefix=configuration+'/'+level+'/';text=archive.read(prefix+'source.ptx').decode();module=parse_module(text);assert list(module.kernels)==list(NAMES);graph=contract_from_ptx(text);exact({row['path']:row['sha256']for row in graph['files']},hashes,('graph',configuration))
    manifest=json.loads(archive.read(prefix+'manifest.json'));entry,=manifest['entries'];assert entry['source_sha256']==hashes[case['entry']]and entry['language']=='python'
    binary=archive.read(prefix+'oa.cubin');validate(entry,binary);binaries[configuration,'oa-'+level]=binary
    if level=='safe':binaries[configuration,'ptxas']=archive.read(configuration+'/ptxas.cubin')
    for kernel in entry['kernels']:
     name=kernel['name'];contract=parameter_contract_from_ptx(text,module.kernels[name],allow_buffers=True);exact(kernel['logical_parameter_contract'],contract,('logical contract',configuration,name));assert [row['dtype']for row in contract['parameters']]==['bool','bool','s32','bool','s32']and[row['kind']for row in contract['parameters']]==['array','array','array','scalar','scalar']
     shaped=buffer_contract(text,module.kernels[name]);exact(kernel.get('buffer_contract'),shaped,('buffer contract',configuration,name))
     if shaped:assert all(row['dtype']=='bool'and row['element_bytes']==1 for row in shaped['buffers'])
     resources={};_,_,words,_,regs=build.generate(module.kernels[name],'sm_80',level,resource_info=resources);exact(words,text_words(binary,name),('words',configuration,name,level));assert regs==regcount(binary,name)==kernel['registers'];assert resources['local_bytes']==local_frame_inventory(binary)[name]==0
     native.append(dict(configuration=configuration,entry=name,level=level,registers=regs,local_bytes=0,words_sha256=sha(b''.join(word.to_bytes(16,'little')for word in words)),passed=True))
  for row in report['results']:
   c,compiler,name,seed,flag,n=(row[k]for k in('configuration','compiler','entry','seed','flag','n'));prefix=f'{c}/{compiler}/{name}/{seed}/{int(flag)}/{n}/'
   before,after,actual,counter=(archive.read(prefix+key+'.bin')for key in('before-a','a','output','counter'));wanted_in,wanted_out,wanted_counter=references(name,seed,flag,n,384)
   exact(before,guard+raw_input(seed,384)+guard,(prefix,'before'));exact(after,guard+wanted_in+guard,(prefix,'input'));exact(actual,guard+wanted_out+guard,(prefix,'output'));exact(counter,guard+wanted_counter+guard,(prefix,'counter'))
   binary=binaries[c,compiler];assert row['passed']and row['output_sha256']==sha(actual)and row['cubin_sha256']==sha(binary)and row['resources']['num_regs']==regcount(binary,name)
   if compiler.startswith('oa-'):assert row['resources']['local_bytes']==0
   count+=len(after)+len(actual)+len(counter);last_expected_output=guard+wanted_out+guard
  for row in report['explicit_launch_results']:
   c,level,name,flag=(row[k]for k in('configuration','level','entry','flag'));prefix=f'{c}/explicit/{level}/{name}/{int(flag)}/';spec=json.loads(archive.read(prefix+'spec.json'));assert spec['steps'][0]['args']==[dict(buf='a',offset=256),dict(buf='a'if name=='alias_invert'else'out',offset=256),dict(buf='counter',offset=256),dict(bool=flag),dict(i32=259)]and type(spec['steps'][0]['args'][3]['bool'])is bool
   for key in('a','out'):assert spec['buffers'][key]['dtype']=='bool'and all(type(value)is bool for value in spec['buffers'][key]['init'])
   assert bytes(map(int,spec['buffers']['a']['init']))==bguard+raw_input(73519,384,True)+bguard and bytes(map(int,spec['buffers']['out']['init']))==bguard+initial(384,True)+bguard
   wanted_in,wanted_out,wanted_counter=references(name,73519,flag,259,384,True);wanted=dict(a=bguard+wanted_in+bguard,out=bguard+(initial(384,True)if name=='alias_invert'else wanted_out)+bguard,counter=guard+wanted_counter+guard)
   for key,data in wanted.items():exact(archive.read(prefix+key+'.bin'),data,(prefix,key));count+=len(data)
   assert row['passed']
  corrupt=bytearray(actual);corrupt[256]^=1;rejected=False
  try:exact(bytes(corrupt),last_expected_output,'corrupted output')
  except AssertionError:rejected=True
  assert rejected
 assert count==18247680 and len(native)==216,(count,len(native))
 result=dict(schema=1,passed=True,report=str(path.relative_to(ROOT)),report_sha256=sha(path.read_bytes()),archive_sha256=report['archive']['sha256'],launches=4320,explicit_launches=432,bytes_checked=count,native_comparisons=native,audited_backend='isolated candidate'if args.candidate else'production',auditor_sha256=sha(Path(__file__).read_bytes()),corrupted_output_rejected=True,scope='All original Boolean byte storage/source/logical/shaped contracts and canonical host Boolean initialization, complete independent output/input/counter/guard/tail bytes, exact current native/register/frame/ownership; no timing')
 args.output.write_text(json.dumps(result,indent=2)+'\n');print('Passed 4320 original launches plus 432 public specs;',count,'complete bytes; 216 native outputs')

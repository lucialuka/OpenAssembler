"""Independently audit original logical Boolean ABI outputs, host specs and native custody."""
import argparse,hashlib,json,sys,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4];HERE=Path(__file__).resolve().parent;sys.path[:0]=[str(ROOT),str(HERE)]
from fixtures import configurations,NAMES,parameters,values,raw_input,initial,expected,expected_counter
def sha(data):return hashlib.sha256(data).hexdigest()

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('report',type=Path);parser.add_argument('output',type=Path);parser.add_argument('--candidate',action='store_true');args=parser.parse_args()
 if args.candidate:
  from candidate import install
  install()
 from openassembler import build
 from openassembler.pyfront import compile_file
 from openassembler.verify import parameter_contract_from_ptx
 from openassembler.pyfront.modules import contract_from_ptx
 from openassembler.ptx.parse import parse_module
 from openassembler.container.cubin import text_words,regcount
 from openassembler.backend.storage import local_frame_inventory
 from openassembler.native_ownership import validate
 path=args.report.resolve();report=json.loads(path.read_text());assert report['passed']and report['error']is None
 replacements={name:HERE/('candidate-'+suffix+'.py')for suffix,name in json.loads((HERE/'inputs.json').read_text()).items()}
 for name,digest in report['source_hashes'].items():
  current=replacements[name]if args.candidate and name in replacements else ROOT/name
  if name in('openassembler/kernels/sm80_gemm_dispatch.json','openassembler/kernels/sm80_rowwise_dispatch.json')and sha(current.read_bytes())!=digest:
   previous=json.loads((HERE.parents[1]/'narrow_parameters/validation.json').read_text());assert previous['source_hashes'][name]==digest
   table=json.loads(current.read_text());policy_path=ROOT/table['source_continuity']['audit'];policy=json.loads(policy_path.read_text())
   assert policy['passed']and policy['counts']==dict(gemm_contracts=44,rowwise_contracts=5,retained_session_checks=98)and sha(policy_path.read_bytes())==table['source_continuity']['sha256']
   assert all(sha((ROOT/key).read_bytes())==value for key,value in policy['compiler_inputs_sha256'].items())
   continue
  assert sha(current.read_bytes())==digest,name
 assert report['harness_sha256']==sha((HERE/'run_modal.py').read_bytes());archive_path=path.parent/report['archive']['path'];assert sha(archive_path.read_bytes())==report['archive']['sha256']
 cases={c['name']:c for c in configurations()};pairs=[(False,False),(False,True),(True,False),(True,True)]
 domain={(c,compiler,name,seed,first,second,n)for c in cases for compiler in('ptxas','oa-safe','oa-mined','oa-sched','oa-opt')for name in NAMES for seed in(73117,73118)for first,second in pairs for n in(0,1,259,384)}
 keys=[tuple(row[k]for k in('configuration','compiler','entry','seed','first','second','n'))for row in report['results']];assert len(keys)==len(set(keys))==len(domain)==9600 and set(keys)==domain
 explicit={(c,level,name,first,second)for c in cases for level in('safe','mined','sched','opt')for name in NAMES for first,second in pairs}
 keys=[tuple(row[k]for k in('configuration','level','entry','first','second'))for row in report['explicit_launch_results']];assert len(keys)==len(set(keys))==len(explicit)==960 and set(keys)==explicit
 native=[];count=0;binaries={};guard=b'\xa5'*256
 with zipfile.ZipFile(archive_path)as archive:
  for configuration,case in cases.items():
   hashes={name:sha(text.encode())for name,text in case['files'].items()}
   for name,text in case['files'].items():assert archive.read(configuration+'/original/'+name)==text.encode()
   for level in('safe','mined','sched','opt'):
    prefix=configuration+'/'+level+'/';text=archive.read(prefix+'source.ptx').decode();module=parse_module(text);assert list(module.kernels)==list(NAMES);graph=contract_from_ptx(text);assert{row['path']:row['sha256']for row in graph['files']}==hashes
    manifest=json.loads(archive.read(prefix+'manifest.json'));entry,=manifest['entries'];assert entry['source_sha256']==hashes[case['entry']]and entry['language']=='python'
    binary=archive.read(prefix+'oa.cubin');validate(entry,binary);binaries[configuration,'oa-'+level]=binary
    if level=='safe':binaries[configuration,'ptxas']=archive.read(configuration+'/ptxas.cubin')
    for kernel in entry['kernels']:
     name=kernel['name'];contract=parameter_contract_from_ptx(text,module.kernels[name]);assert kernel['logical_parameter_contract']==contract and any(row['dtype']=='bool'for row in contract['parameters'])
     for (source_name,dtype),row in zip(parameters(name),contract['parameters']):
      assert row['name']==source_name and row['dtype']==({'i32':'s32','i64':'s64'}.get(dtype[6:-1],dtype[6:-1])if dtype.startswith('Array(')else{'i32':'s32','i64':'s64'}.get(dtype,dtype))
     resources={};_,_,words,_,regs=build.generate(module.kernels[name],'sm_80',level,resource_info=resources);assert words==text_words(binary,name)and regs==regcount(binary,name)==kernel['registers'];assert resources['local_bytes']==local_frame_inventory(binary)[name]==0
     native.append(dict(configuration=configuration,entry=name,level=level,registers=regs,passed=True))
  for row in report['results']:
   c,compiler,name,seed,first,second,n=(row[k]for k in('configuration','compiler','entry','seed','first','second','n'));prefix=f'{c}/{compiler}/{name}/{seed}/{int(first)}{int(second)}/{n}/'
   before,after,actual,counter=(archive.read(prefix+key+'.bin')for key in('before-a','a','output','counter'))
   assert before==after==guard+raw_input(name,seed,384)+guard
   assert actual==guard+expected(name,seed,first,second,n,384)+guard and counter==guard+expected_counter(name,seed,first,second,n,384)+guard
   binary=binaries[c,compiler];assert row['passed']and row['output_sha256']==sha(actual)and row['cubin_sha256']==sha(binary)and row['resources']['num_regs']==regcount(binary,name)
   if compiler.startswith('oa-'):assert row['resources']['local_bytes']==0
   count+=len(after)+len(actual)+len(counter)
  for row in report['explicit_launch_results']:
   c,level,name,first,second=(row[k]for k in('configuration','level','entry','first','second'));prefix=f'{c}/explicit/{level}/{name}/{int(first)}{int(second)}/';spec=json.loads(archive.read(prefix+'spec.json'));scalar_values=values(name,first,second,259)
   for(key,dtype),argument in zip(parameters(name),spec['steps'][0]['args']):
    if dtype.startswith('Array'):assert argument==dict(buf=key,offset=256)
    elif dtype=='bool':assert set(argument)=={'bool'}and type(argument['bool'])is bool and argument['bool']==scalar_values[key]
    else:assert argument=={dtype:scalar_values[key]}
   wanted=dict(a=guard+raw_input(name,73117,384)+guard,out=guard+expected(name,73117,first,second,259,384)+guard,counter=guard+expected_counter(name,73117,first,second,259,384)+guard)
   for key,data in wanted.items():assert archive.read(prefix+key+'.bin')==data;count+=len(data)
   assert row['passed']
  corrupt=bytearray(actual);corrupt[256]^=1;assert corrupt!=guard+expected(name,seed,first,second,n,384)+guard
 assert count==69746688 and len(native)==240
 result=dict(schema=1,passed=True,report=str(path.relative_to(ROOT)),report_sha256=sha(path.read_bytes()),archive_sha256=report['archive']['sha256'],launches=9600,explicit_launches=960,bytes_checked=count,native_comparisons=native,audited_backend='isolated candidate'if args.candidate else'production',auditor_sha256=sha(Path(__file__).read_bytes()),corrupted_output_rejected=True,scope='Original logical Boolean source/physical signatures and canonical host Boolean spec fields, complete independent numeric/effect/raw FP32/output/input/counter/guard/tail bytes, exact current native/register/frame/ownership; no timing')
 args.output.write_text(json.dumps(result,indent=2)+'\n');print('Passed 9600 original launches plus 960 public specs;',count,'complete bytes; 240 native outputs')

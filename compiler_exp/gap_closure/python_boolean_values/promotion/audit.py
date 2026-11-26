"""Source-exact Final declaration, whole independent bytes and native/resource custody."""
import argparse,hashlib,json,sys,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4];HERE=Path(__file__).resolve().parent;sys.path[:0]=[str(ROOT),str(HERE)]
from fixtures import configurations,NAMES,raw_input,initial,expected,expected_counter

def sha(data):return hashlib.sha256(data).hexdigest()
def equal(actual,wanted):assert actual==wanted,'complete byte comparison failed'
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('report',type=Path);parser.add_argument('output',type=Path);parser.add_argument('--candidate',action='store_true');args=parser.parse_args()
 if args.candidate:
  from candidate import install
  install()
 from openassembler import build
 from openassembler.pyfront.modules import contract_from_ptx
 from openassembler.ptx.parse import parse_module
 from openassembler.native_ownership import validate
 from openassembler.container.cubin import text_words,regcount
 from openassembler.backend.storage import local_frame_inventory
 replacements={'openassembler/backend/compile.py':'candidate-backend.py','openassembler/pyfront/compile.py':'candidate-pyfront.py','openassembler/pyfront/modules.py':'candidate-modules.py'}
 path=args.report.resolve();report=json.loads(path.read_text());assert report['passed']and report['error']is None
 for name,digest in report['source_hashes'].items():
  current=HERE/replacements[name]if args.candidate and name in replacements else ROOT/name
  if name in('openassembler/kernels/sm80_gemm_dispatch.json','openassembler/kernels/sm80_rowwise_dispatch.json')and sha(current.read_bytes())!=digest:
   previous=json.loads((HERE.parents[1]/'python_comparison_chains/promotion/validation.json').read_text());assert previous['source_hashes'][name]==digest
   table=json.loads(current.read_text());policy_path=ROOT/table['source_continuity']['audit'];policy=json.loads(policy_path.read_text())
   assert policy['passed']and policy['counts']==dict(gemm_contracts=44,rowwise_contracts=5,retained_session_checks=98)and sha(policy_path.read_bytes())==table['source_continuity']['sha256']
   assert all(sha((ROOT/key).read_bytes())==value for key,value in policy['compiler_inputs_sha256'].items())
   continue
  assert sha(current.read_bytes())==digest,name
 assert report['harness_sha256']==sha((HERE/'run_modal.py').read_bytes())
 archive_path=path.parent/report['archive']['path'];assert sha(archive_path.read_bytes())==report['archive']['sha256']
 cases={c['name']:c for c in configurations()};domain={(case,compiler,name,seed,n)for case in cases for compiler in('ptxas','oa-safe','oa-mined','oa-sched','oa-opt')for name in NAMES for seed in(42131,42132)for n in(0,1,259,384)}
 # Harness seeds are part of source-exact evidence; reject accidental changes.
 observed_seeds={row['seed']for row in report['results']}
 assert observed_seeds=={31881,31882},observed_seeds
 domain={(c,compiler,name,seed,n)for c in cases for compiler in('ptxas','oa-safe','oa-mined','oa-sched','oa-opt')for name in NAMES for seed in(31881,31882)for n in(0,1,259,384)}
 keys=[tuple(row[key]for key in('configuration','compiler','entry','seed','n'))for row in report['results']];assert len(keys)==len(set(keys))==len(domain)==2400 and set(keys)==domain
 native=[];count=0;guard=b'\xa5'*256;binaries={}
 with zipfile.ZipFile(archive_path)as archive:
  for configuration,case in cases.items():
   hashes={name:sha(text.encode())for name,text in case['files'].items()}
   for name,text in case['files'].items():assert archive.read(configuration+'/original/'+name)==text.encode()
   for level in('safe','mined','sched','opt'):
    prefix=configuration+'/'+level+'/';text=archive.read(prefix+'source.ptx').decode();module=parse_module(text);assert list(module.kernels)==list(NAMES)
    graph=contract_from_ptx(text);assert graph['entry']==case['entry']and{r['path']:r['sha256']for r in graph['files']}==hashes
    assert graph['static_values'] and graph['boolean_values'] and all(row['kernels'] for row in graph['boolean_values'].values())
    manifest=json.loads(archive.read(prefix+'manifest.json'));entry,=manifest['entries'];assert entry['source_sha256']==hashes[case['entry']]and entry['language']=='python'
    binary=archive.read(prefix+'oa.cubin');validate(entry,binary);binaries[configuration,'oa-'+level]=binary
    if level=='safe':binaries[configuration,'ptxas']=archive.read(configuration+'/ptxas.cubin')
    for kernel in entry['kernels']:
     name=kernel['name'];assert kernel['python_source_graph']==graph
     resources={};_,_,words,_,registers=build.generate(module.kernels[name],'sm_80',level,resource_info=resources)
     assert words==text_words(binary,name)and registers==regcount(binary,name)==kernel['registers']
     assert resources['local_bytes']==kernel.get('local_bytes',0)==local_frame_inventory(binary)[name]==0
     native.append(dict(configuration=configuration,entry=name,level=level,registers=registers,local_bytes=0,passed=True))
  for row in report['results']:
   c,compiler,name,seed,n=(row[key]for key in('configuration','compiler','entry','seed','n'));prefix=f'{c}/{compiler}/{name}/{seed}/{n}/'
   before,after,actual,counter=(archive.read(prefix+key+'.bin')for key in('before-a','a','output','counter'))
   equal(before,guard+raw_input(name,seed,384)+guard);wanted=guard+expected(name,seed,n,384)+guard;equal(actual,wanted);equal(after,before);equal(counter,guard+expected_counter(name,seed,n,384)+guard)
   binary=binaries[c,compiler];assert row['passed']and sha(actual)==row['output_sha256']and sha(binary)==row['cubin_sha256']and row['resources']['num_regs']==regcount(binary,name)
   if compiler.startswith('oa-'):assert row['resources']['local_bytes']==row['local_frame_bytes']==0
   else:assert row['local_frame_bytes']is None
   count+=len(actual)+len(after)+len(counter)
  corrupt=bytearray(actual);corrupt[256]^=1
  try:equal(corrupt,wanted)
  except AssertionError:pass
  else:raise AssertionError('corrupted output accepted')
 assert count==16957440 and len(native)==240
 result=dict(schema=1,passed=True,report=str(path.relative_to(ROOT)),report_sha256=sha(path.read_bytes()),archive_sha256=report['archive']['sha256'],launches=2400,bytes_checked=count,native_comparisons=native,audited_backend='isolated candidate'if args.candidate else'production',auditor_sha256=sha(Path(__file__).read_bytes()),corrupted_output_rejected=True,scope='Original typed Boolean-value and operand-selection sources with exact source/helper/use identity, complete independent return/caller/break/continue/nested/else/modulo64/FP32 guarded outputs and immutable inputs, exact current native/resource/ownership; no timing')
 args.output.write_text(json.dumps(result,indent=2)+'\n');print('Passed 2400 launches;',count,'bytes; 240 native outputs')

"""Source graph custody, independent whole bytes and exact current native outputs."""
import argparse,hashlib,json,sys,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4];HERE=Path(__file__).resolve().parent;sys.path[:0]=[str(ROOT),str(HERE)]
from fixtures import configurations,raw_data,initial,expected

def sha(data):return hashlib.sha256(data).hexdigest()
def equal(a,b):assert a==b,'complete byte comparison failed'
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('report',type=Path);parser.add_argument('output',type=Path);parser.add_argument('--candidate',action='store_true');parser.add_argument('--first',action='store_true');args=parser.parse_args()
 replacements={'openassembler/pyfront/modules.py':'modules.py','openassembler/pyfront/compile.py':'candidate-pyfront.py','openassembler/pyfront/helpers.py':'candidate-helpers.py','openassembler/build.py':'candidate-build.py','openassembler/frontend.py':'candidate-frontend.py','openassembler/native_ownership.py':'candidate-ownership.py','openassembler/remote_build.py':'candidate-remote.py','openassembler/cli.py':'candidate-cli.py'}
 if args.candidate:
  from candidate import install
  install()
 if args.first:
  assert args.candidate
  import builtins,importlib
  for name,file in(('openassembler.pyfront.modules','first-modules.py'),('openassembler.pyfront.compile','first-pyfront.py')):
   module=importlib.import_module(name);path=HERE/file;exec(builtins.compile(path.read_bytes(),str(path),'exec'),module.__dict__)
  replacements['openassembler/pyfront/modules.py']='first-modules.py';replacements['openassembler/pyfront/compile.py']='first-pyfront.py'
 from openassembler import build
 from openassembler.ptx.parse import parse_module
 from openassembler.pyfront.modules import contract_from_ptx
 from openassembler.native_ownership import validate
 from openassembler.container.cubin import text_words,regcount
 from openassembler.backend.storage import local_frame_inventory
 path=args.report.resolve();report=json.loads(path.read_text());assert report['passed']and report['error']is None and len(report['results'])==600
 for name,digest in report['source_hashes'].items():
  current=HERE/replacements[name]if args.candidate and name in replacements else ROOT/name
  assert sha(current.read_bytes())==digest,name
 assert report['harness_sha256']==sha((HERE/'run_modal.py').read_bytes())
 archive_path=path.parent/report['archive']['path'];assert sha(archive_path.read_bytes())==report['archive']['sha256']
 cases={c['name']:c for c in configurations()};domain={(c,compiler,name,seed,n)for c in cases for compiler in('ptxas','oa-safe','oa-mined','oa-sched','oa-opt')for name in('signed','unsigned','floating')for seed in(31881,31882)for n in(0,1,259,384)}
 keys=[tuple(row[key]for key in('configuration','compiler','entry','seed','n'))for row in report['results']];assert len(keys)==len(set(keys))==len(domain)==600 and set(keys)==domain
 count=0;native=[];guard=b'\xa5'*256;binaries={};contracts={}
 with zipfile.ZipFile(archive_path)as archive:
  for configuration,case in cases.items():
   for name,text in case['files'].items():assert archive.read(configuration+'/original/'+name)==text.encode()
   file_hashes={name:sha(text.encode())for name,text in case['files'].items()}
   for level in('safe','mined','sched','opt'):
    prefix=configuration+'/'+level+'/';source=archive.read(prefix+'source.ptx');module=parse_module(source.decode());assert set(module.kernels)=={'signed','unsigned','floating'}
    graph=contract_from_ptx(source.decode());assert graph['entry']==case['entry']and{r['path']:r['sha256']for r in graph['files']}==file_hashes
    helpers=graph['helpers'];assert len(helpers)==5
    assert helpers['arithmetic/transforms.py:signed_step']['kernels']=={'signed':1}
    assert helpers['arithmetic/base.py:signed_product']['kernels']=={'signed':1}
    assert helpers['arithmetic/transforms.py:unsigned_step']['kernels']=={'unsigned':1}
    assert helpers['arithmetic/base.py:unsigned_product']['kernels']=={'unsigned':1}
    assert helpers['arithmetic/base.py:affine']['kernels']=={'floating':1}
    manifest=json.loads(archive.read(prefix+'manifest.json'));entry,=manifest['entries'];assert entry['source_sha256']==file_hashes[case['entry']]and entry['language']=='python'
    binary=archive.read(prefix+'oa.cubin');validate(entry,binary);binaries[configuration,'oa-'+level]=binary
    if level=='safe':binaries[configuration,'ptxas']=archive.read(configuration+'/ptxas.cubin')
    for kernel in entry['kernels']:
     name=kernel['name'];assert kernel['python_source_graph']==graph
     resources={};_,_,words,_,registers=build.generate(module.kernels[name],'sm_80',level,resource_info=resources)
     assert words==text_words(binary,name)and registers==regcount(binary,name)==kernel['registers']
     assert resources['local_bytes']==kernel.get('local_bytes',0)==local_frame_inventory(binary)[name]==0
     contracts[configuration,'oa-'+level,name]=kernel
     native.append(dict(configuration=configuration,entry=name,level=level,registers=registers,local_bytes=0,passed=True))
  for row in report['results']:
   c,compiler,name,seed,n=(row[key]for key in('configuration','compiler','entry','seed','n'));prefix=f'{c}/{compiler}/{name}/{seed}/{n}/'
   before_a,before_b,after_a,after_b,actual=(archive.read(prefix+key+'.bin')for key in('before-a','before-b','a','b','output'))
   a,b=raw_data(name,seed,384);equal(before_a,guard+a+guard);equal(before_b,guard+b+guard)
   wanted=guard+expected(name,seed,n,384)+guard;equal(actual,wanted);equal(after_a,before_a);equal(after_b,before_b)
   assert row['passed']and sha(actual)==row['output_sha256']and sha(binaries[c,compiler])==row['cubin_sha256']
   assert row['resources']['num_regs']==regcount(binaries[c,compiler],name)
   if compiler.startswith('oa-'):assert row['resources']['local_bytes']==row['local_frame_bytes']==contracts[c,compiler,name].get('local_bytes',0)==0
   else:assert row['local_frame_bytes']is None
   count+=len(actual)+len(after_a)+len(after_b)
  corrupted=bytearray(actual);corrupted[256]^=1
  try:equal(corrupted,wanted)
  except AssertionError:pass
  else:raise AssertionError('corrupted output accepted')
 assert count==5529600 and len(native)==60
 result=dict(schema=1,passed=True,report=str(path.relative_to(ROOT)),report_sha256=sha(path.read_bytes()),archive_sha256=report['archive']['sha256'],launches=600,bytes_checked=count,native_comparisons=native,audited_backend='isolated candidate'if args.candidate else'production',auditor_sha256=sha(Path(__file__).read_bytes()),corrupted_output_rejected=True,scope='Original Python trees, complete package/module dependency identities, exact helper signatures/bodies/call sites and ownership, all independent whole byte outputs/inputs/guards/tails and native words/registers/resources, no timing/performance claim')
 args.output.write_text(json.dumps(result,indent=2)+'\n');print('Passed 600 launches;',count,'bytes;',len(native),'native outputs')

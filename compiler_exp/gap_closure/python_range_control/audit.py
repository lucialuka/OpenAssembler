"""Independent Python range bytes and exact native custody for the bounded A100 probe."""
import argparse,hashlib,json,struct,sys,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;sys.path[:0]=[str(ROOT),str(HERE)]
from probe import CASES
from openassembler import build
from openassembler.ptx.parse import parse_module
from openassembler.container.cubin import text_words,regcount
from openassembler.native_ownership import validate
def sha(data):return hashlib.sha256(data).hexdigest()
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('report',type=Path);args=parser.parse_args();path=args.report.resolve();report=json.loads(path.read_text());assert report['passed']and report['error']is None
 assert report['harness_sha256']==sha((HERE/'run_modal.py').read_bytes())and report['reference_sha256']==sha((HERE/'baseline.json').read_bytes())
 assert all(sha((ROOT/n).read_bytes())==h for n,h in report['payload_hashes'].items())
 assert all(sha((HERE/(n+'-pyfront.py')).read_bytes())==h for n,h in report['compiler_hashes'].items())
 archive_path=path.parent/report['archive']['path'];assert sha(archive_path.read_bytes())==report['archive']['sha256']
 expected={};guard=b'\xa5'*64;mask=(1<<64)-1
 for name,start,stop,step in CASES:
  values=list(range(start,stop,step));assert len(values)<8
  expected[name]=guard+struct.pack('<3Q',sum(values)&mask,len(values),(values[-1]if values else 17)&mask)+guard
 domain={(strategy,backend,name)for strategy in('baseline','candidate')for backend in('oa-safe','oa-mined','oa-sched','oa-opt','ptxas')for name in expected}
 keys=[tuple(row[k]for k in('strategy','backend','entry'))for row in report['results']];assert len(keys)==len(set(keys))==60 and set(keys)==domain
 native=[];bytes_checked=0;known_mismatches=0;toolkit_observations=[]
 with zipfile.ZipFile(archive_path)as archive:
  assert sha(archive.read('original.py'))==report['source_sha256'];binaries={}
  for strategy in('baseline','candidate'):
   assert sha(archive.read(strategy+'/compiler.py'))==report['compiler_hashes'][strategy]
   for level in('safe','mined','sched','opt'):
    prefix=strategy+'/'+level+'/';text=archive.read(prefix+'source.ptx');binary=archive.read(prefix+'oa.cubin');manifest=json.loads(archive.read(prefix+'manifest.json'));validate(manifest['entries'][0],binary);binaries[strategy,'oa-'+level]=binary
    if level=='safe':binaries[strategy,'ptxas']=archive.read(strategy+'/ptxas.cubin')
    for name,kernel in parse_module(text.decode()).kernels.items():
     _,_,words,_,registers=build.generate(kernel,'sm_80',level)
     assert words==text_words(binary,name)and registers==regcount(binary,name)
     native.append(dict(strategy=strategy,level=level,entry=name,registers=registers,passed=True))
  for row in report['results']:
   strategy,backend,name=(row[k]for k in('strategy','backend','entry'));actual=archive.read(f'{strategy}/{backend}/{name}/output.bin');binary=binaries[strategy,backend]
   assert len(actual)==152 and actual[:64]==actual[-64:]==guard
   matches=actual==expected[name];wanted=None if strategy=='baseline'and backend=='ptxas'else strategy=='candidate'or name in('ordinary','empty')
   assert matches==row['matches_reference']and row['expected_matches_reference']==wanted
   if wanted is None:toolkit_observations.append(dict(entry=name,matches_python_range=matches,observed_signed=list(struct.unpack('<3q',actual[64:-64]))))
   else:assert matches==wanted
   assert sha(actual)==row['output_sha256']and sha(binary)==row['cubin_sha256']and row['registers']==row['resources']['num_regs']==regcount(binary,name)
   known_mismatches+=wanted is False;bytes_checked+=len(actual)
  candidate=archive.read('candidate/oa-opt/positive_overflow/output.bin');corrupt=bytearray(candidate);corrupt[64]^=1;assert bytes(corrupt)!=expected['positive_overflow']
 assert len(native)==48 and known_mismatches==16 and len(toolkit_observations)==6 and bytes_checked==9120
 result=dict(schema=1,passed=True,production_fix=False,report=str(path.relative_to(ROOT)),report_sha256=sha(path.read_bytes()),archive_sha256=report['archive']['sha256'],launches=60,candidate_correct_launches=30,baseline_oa_control_correct_launches=8,baseline_oa_known_mismatch_launches=16,baseline_ptxas_observations=toolkit_observations,bytes_checked=bytes_checked,native_comparisons=native,corrupted_output_rejected=True,auditor_sha256=sha(Path(__file__).read_bytes()),scope='Bounded range overflow hardware counterexamples and isolated guard correction, 9120 complete output/guard bytes and 48 exact native word/register comparisons. 30 candidate numerical passes, 16 required baseline OA mismatches, eight baseline OA controls and six explicitly observational baseline ptxas results are distinct. No production integration, universal PTX semantics or timing claim')
 (HERE/'candidate-audit.json').write_text(json.dumps(result,indent=2)+'\n');print('Audited 60 observations; 30 candidate passes / 16 baseline OA counterexamples / 8 controls / 6 observational ptxas results; 9120 bytes; 48 native comparisons')

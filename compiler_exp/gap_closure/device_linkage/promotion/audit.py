"""Read-only original-linked controls, complete library/byte/resource/native custody."""
import argparse,hashlib,json,struct,sys,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4];HERE=Path(__file__).resolve().parent;sys.path[:0]=[str(ROOT),str(HERE)]
from fixtures import cases,values,expected

def sha(data):return hashlib.sha256(data).hexdigest()
AUDITOR=sha(Path(__file__).read_bytes())
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('report',type=Path);parser.add_argument('output',type=Path);parser.add_argument('--candidate',action='store_true');args=parser.parse_args()
 if args.candidate:
  from candidate import install
  install()
 from openassembler import build
 from openassembler.ptx.link import resolve
 from openassembler.ptx.parse import parse_module
 from openassembler.container.cubin import text_words,regcount
 from openassembler.native_ownership import validate
 from openassembler.backend.storage import local_frame_inventory
 path=args.report.resolve();report=json.loads(path.read_text());assert report['passed']and len(report['results'])==576
 replacements={'openassembler/ptx/link.py':'link.py','openassembler/ptx/inlining.py':'candidate-inlining.py','openassembler/build.py':'candidate-build.py','openassembler/cppfront.py':'candidate-cppfront.py','openassembler/cli.py':'candidate-cli.py','openassembler/native_ownership.py':'candidate-native_ownership.py','openassembler/remote_build.py':'candidate-remote_build.py'}
 for name,digest in report['source_hashes'].items():
  p=HERE/replacements[name]if args.candidate and name in replacements else ROOT/name
  assert sha(p.read_bytes())==digest,name
 assert report['harness_sha256']==sha((HERE/'run_modal.py').read_bytes())
 archive_path=HERE/report['archive']['path'];assert sha(archive_path.read_bytes())==report['archive']['sha256']
 fixtures={c['name']:c for c in cases()};fixtures.update({name:dict(name=name)for name in('cuda_database','cuda_remote')})
 domain={(name,compiler,n,seed)for name in fixtures for compiler in('original-linked','merged-ptxas','oa-safe','oa-mined','oa-sched','oa-opt')for n in(0,1,259,384)for seed in(21941,21942)}
 assert{tuple(r[k]for k in('name','compiler','n','seed'))for r in report['results']}==domain
 native=[];count=0;guard=b'\xa5'*256;binaries={};contracts={}
 with zipfile.ZipFile(archive_path)as archive:
  for name,case in fixtures.items():
   cuda=name.startswith('cuda_')
   if not cuda:
    for i,text in enumerate([case['root'],*case['libraries']]):assert archive.read(name+f'/unit{i}.ptx')==text.encode()
    independent,link=resolve([('entry',case['root'])]+[(str(i),text)for i,text in enumerate(case['libraries'])])
   else:
    assert b'helper'in archive.read(name+'/unit0.cu')and b'MUL+OFFSET'in archive.read(name+'/unit1.cu')
    assert archive.read(name+'/settings.cuh')==b'#define OFFSET 9u\n'
   binaries[name,'original-linked']=archive.read(name+'/original-linked.cubin');binaries[name,'merged-ptxas']=archive.read(name+'/merged-ptxas.cubin')
   for level in('safe','mined','sched','opt'):
    prefix=name+'/'+level+'/';source=archive.read(prefix+'merged.ptx').decode();module=parse_module(source)
    if not cuda:assert source==independent
    manifest=json.loads(archive.read(prefix+'manifest.json'));entry=manifest['entries'][0];contract=entry['kernels'][0];binary=archive.read(prefix+'oa.cubin');validate(entry,binary)
    assert contract['device_link']['linked_ptx_sha256']==sha(source.encode())
    assert contract['inlined_device_calls']['strategy']=='direct-scalar-static-inlining'
    assert len(contract['device_link']['libraries'])==(1 if cuda else len(case['libraries']))
    if cuda:
     item=contract['device_link']['libraries'][0];assert item['source_sha256']==sha(archive.read(name+'/unit1.cu'))
     assert '-DMUL=17'in item['compile_contract']['device_flags']
     assert any(k.endswith('settings.cuh')and digest==sha(archive.read(name+'/settings.cuh'))for k,digest in item['dependencies'].items())
     assert contract['device_link']['entry_compilation']['dependencies']
     if name=='cuda_remote':assert item['compile_contract']['submitted_origin']
    else:
     for i,item in enumerate(contract['device_link']['libraries']):assert item['source_sha256']==sha(case['libraries'][i].encode())
    resources={};_,_,words,_,registers=build.generate(module.kernels['probe'],'sm_80',level,resource_info=resources)
    assert words==text_words(binary,'probe')and registers==regcount(binary,'probe')
    assert resources['local_bytes']==contract.get('local_bytes',0)==(32 if name=='local_frame'else 0)
    binaries[name,'oa-'+level]=binary;contracts[name,level]=contract
    native.append(dict(name=name,level=level,registers=registers,local_bytes=resources['local_bytes'],passed=True))
  for row in report['results']:
   name,compiler,n,seed=(row[k]for k in('name','compiler','n','seed'));prefix=f'{name}/{compiler}/{n}/{seed}/'
   before=archive.read(prefix+'before-input.bin');unchanged=archive.read(prefix+'input.bin');actual=archive.read(prefix+'output.bin');data=values(seed,384)
   assert before==guard+struct.pack('<384Q',*data)+guard and unchanged==before
   wanted=guard+expected(data,fixtures[name],n)+guard;assert actual==wanted and row['passed']and len(actual)==3584
   assert sha(actual)==row['output_sha256']and sha(unchanged)==row['input_sha256']
   binary=binaries[name,compiler];assert sha(binary)==row['cubin_sha256']
   if compiler.startswith('oa-'):
    level=compiler.removeprefix('oa-');assert local_frame_inventory(binary)['probe']==row['local_frame_bytes']==row['resources']['local_bytes']==contracts[name,level].get('local_bytes',0)
    assert row['resources']['num_regs']==regcount(binary,'probe')
   else:assert row['local_frame_bytes']is None
   count+=len(actual)+len(unchanged)
  mutant=bytearray(actual);mutant[256]^=1;assert mutant!=wanted
 result=dict(schema=1,passed=True,report=str(path.relative_to(ROOT)),report_sha256=sha(path.read_bytes()),archive_sha256=report['archive']['sha256'],launches=len(domain),bytes_checked=count,native_comparisons=native,audited_backend='isolated candidate'if args.candidate else'production',auditor_sha256=AUDITOR,corrupted_output_rejected=True,scope='Complete original separately linked/merged controls versus normal CLI and remote library builds, exact source/dependency/link/ownership contracts, independent bytes and all current native words/resources; no new execution or timing')
 args.output.write_text(json.dumps(result,indent=2)+'\n');print(len(domain),count,len(native))

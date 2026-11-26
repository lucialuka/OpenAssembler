"""Independently classify all retained optimized/unoptimized counter observations."""
import hashlib,json,struct,zipfile
from pathlib import Path
HERE=Path(__file__).resolve().parent
def sha(data):return hashlib.sha256(data).hexdigest()
if __name__=='__main__':
 path=HERE/'minimized-a100-20261124T061630Z.json';report=json.loads(path.read_text());controls=json.loads((HERE/'minimized.json').read_text());assert report['passed']and report['error']is None
 assert report['harness_sha256']==sha((HERE/'run_minimized.py').read_bytes())and report['controls_sha256']==sha((HERE/'minimized.json').read_bytes())and report['source_sha256']==controls['source_sha256']==sha((HERE/'minimized.ptx').read_bytes())
 archive=HERE/report['archive']['path'];assert sha(archive.read_bytes())==report['archive']['sha256'];cases={r['entry']:r for r in controls['results']};rows=[]
 keys={(r['entry'],r['optimization'])for r in report['results']};assert len(keys)==len(report['results'])==40 and keys=={(n,o)for n in cases for o in(0,3)}
 with zipfile.ZipFile(archive)as retained:
  assert sha(retained.read('probe.ptx'))==report['source_sha256']
  for row in report['results']:
   name,level=row['entry'],row['optimization'];control=cases[name];actual=retained.read(f'o{level}/{name}/output.bin');assert sha(actual)==row['output_sha256']and actual[:64]==actual[-64:]==b'\xa5'*64
   values=list(struct.unpack('<3q',actual[64:-64]));assert values==row['observed_signed']
   assert row['matches_python']==(values==control['python_reference'])and row['matches_wrapped']==(values==control['wrapped_reference'])
   # This is an assertion of this retained finite observation, not an ISA rule.
   expected=control['python_reference']if level==3 and control['mode']in('static_signed','static_unsigned')else control['wrapped_reference']
   assert values==expected,(name,level,values,expected);rows.append(dict(entry=name,optimization=level,observed_signed=values,passed=True))
 result=dict(schema=1,passed=True,observations=40,bytes_checked=6080,report=path.name,report_sha256=sha(path.read_bytes()),archive_sha256=report['archive']['sha256'],auditor_sha256=sha(Path(__file__).read_bytes()),results=rows,finding='CUDA 12.8.93 O0 and runtime-bound loops retain i32-wrap watchdog outcomes; O3 folds the four static-bound overflow loops to mathematical range results with either signed or unsigned increments. Isolated signed and unsigned adds wrap at O0/O3. This isolates external loop optimization; its general legality/implementation remains unclaimed.',scope='Independent complete raw output/guard/source/archive/tool-harness identity and explicit finite reference classification; no production fix, universal PTX rule or performance claim')
 (HERE/'minimized-audit.json').write_text(json.dumps(result,indent=2)+'\n');print('Audited 40 minimized observations and 6080 complete bytes')

"""Invoke the actual public compile and saved-run commands in the A100 worker."""
import json,subprocess,sys

def run(root):
    records=[]
    commands=[['compile','repo','--backend','local','--verify','--no-sass','--out-dir','cli-output'],
              ['run','cli-output','--backend','local','--verify']]
    for args in commands:
        result=subprocess.run([sys.executable,'-m','openassembler',*args],cwd=root,capture_output=True,text=True,timeout=600)
        records.append({'args':args,'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr})
        if result.returncode:break
    reports={p.name:json.loads(p.read_text()) for p in (root/'cli-output').glob('*report.json')}
    if (root/'cli-output/model-run.json').exists():
        reports['model-run.json']=json.loads((root/'cli-output/model-run.json').read_text())
    return {'commands':records,'reports':reports,'passed':len(records)==2 and all(r['returncode']==0 for r in records)}

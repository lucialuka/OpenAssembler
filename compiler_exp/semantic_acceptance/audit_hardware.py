"""Re-encode final sources against the actual retained A100 executable words."""
from pathlib import Path
import hashlib
import json
import sys
import zipfile
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from openassembler.build import generate
from openassembler.container.cubin import text_words,regcount
from openassembler.ptx.parse import parse_module


if __name__=='__main__':
    report_path=Path(sys.argv[1]);record=json.loads(report_path.read_text())
    archive=report_path.parent/record['artifact_archive']['path']
    assert hashlib.sha256(archive.read_bytes()).hexdigest()==record['artifact_archive']['sha256']
    assert record['passed'] and all(r['mismatches']==0 and r['guards'] and r['inputs_unchanged'] for r in record['results'])
    checks=[]
    with zipfile.ZipFile(archive) as z:
        for name in z.namelist():
            if not name.endswith('/oa.cubin'):continue
            family,level,_=name.split('/')
            source=z.read(family+'/source.ptx').decode()
            code,labels,words,exits,registers=generate(parse_module(source).kernels['probe'],'sm_80',level)
            binary=z.read(name)
            assert text_words(binary,'probe')==words
            assert regcount(binary,'probe')==registers
            checks.append({'artifact':name,'registers':registers,'instructions':len(code),
                           'native_text_sha256':hashlib.sha256(b''.join(w.to_bytes(16,'little') for w in words)).hexdigest()})
    root=Path(__file__).resolve().parents[2]
    paths=[root/'openassembler/backend/ra.py',root/'openassembler/backend/contracts.py',root/'openassembler/ptx/parse.py']
    output={'schema':1,'scope':'Final sources re-encode exactly the 24 native modules executed on A100; no new timing',
            'record':report_path.name,'record_sha256':hashlib.sha256(report_path.read_bytes()).hexdigest(),
            'final_source_hashes':{p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
            'checks':checks,'passed':len(checks)==24}
    target=report_path.parent/'final-hardware-continuity.json';target.write_text(json.dumps(output,indent=2)+'\n')
    print(target,'checks',len(checks),'passed',output['passed'])

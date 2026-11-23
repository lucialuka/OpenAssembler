"""Adversarial read-only audit controls, using disposable copies of evidence.

Rebind archive/cubin/text hashes deliberately after changing a control bit or
register count. The continuity audit must still reject exact executable drift.
Original hardware records and archives remain untouched.
"""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import zipfile

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE));sys.path.insert(0,str(ROOT))
from hardware_continuity import run
from openassembler.container.elf import Elf

if __name__=='__main__':
    source=ROOT/'compiler_exp/gap_closure/operators/hardware-20261121T124125Z.json'
    original=json.loads(source.read_text())
    original_archive=source.parent/original['archive']['path']
    original_hash=hashlib.sha256(original_archive.read_bytes()).hexdigest()
    results=[]
    for kind in ('control_bit','register_count'):
        with tempfile.TemporaryDirectory(prefix='.hardware-audit-negative-',dir=HERE) as temporary:
            folder=Path(temporary)
            report=json.loads(source.read_text())
            with zipfile.ZipFile(original_archive) as archive:
                files={name:archive.read(name) for name in archive.namelist()}
            ledger_path=next(name for name in sorted(files) if name.endswith('/native-ledger.json'))
            prefix=ledger_path.removesuffix('native-ledger.json');ledger=json.loads(files[ledger_path])
            cubin_path=prefix+ledger['cubin'];elf=Elf.parse(files[cubin_path])
            name=next(iter(ledger['kernels']));section=elf.section('.text.'+name)
            if kind=='control_bit':
                text=bytearray(section.data)
                text[105//8]^=1<<(105%8)
                section.data=bytes(text)
                ledger['kernels'][name]['text_sha256']=hashlib.sha256(section.data).hexdigest()
            else:
                section.info += 1<<24
                ledger['kernels'][name]['registers']+=1
            files[cubin_path]=elf.serialize()
            ledger['cubin_sha256']=hashlib.sha256(files[cubin_path]).hexdigest()
            files[ledger_path]=json.dumps(ledger,sort_keys=True,separators=(',',':')).encode()
            forged=folder/'negative-control.zip'
            with zipfile.ZipFile(forged,'w',zipfile.ZIP_DEFLATED) as archive:
                for path,data in files.items():archive.writestr(path,data)
            report['archive']={'path':forged.name,'sha256':hashlib.sha256(forged.read_bytes()).hexdigest()}
            fake_report=folder/'negative-control.json';fake_report.write_text(json.dumps(report))
            outcome=run([fake_report])
            expected=[row for row in outcome['failures'] if row.get('bundle')==prefix]
            assert outcome['passed'] is False and expected,(kind,outcome['failures'])
            assert 'native words, controls, register count or exits differ' in expected[0]['reason']
            results.append({'control':kind,'passed':True,'expected_audit_rejection':expected[0]['reason'],
                            'artifact_copy_only':True,'outer_and_inner_hashes_rebound':True})
    assert hashlib.sha256(original_archive.read_bytes()).hexdigest()==original_hash==original['archive']['sha256']
    result={'schema':1,'passed':True,'hardware':'not run; disposable adversarial artifact copies only',
            'original_report':source.relative_to(ROOT).as_posix(),'original_archive_sha256':original_hash,
            'original_archive_unchanged':True,'controls':results}
    (HERE/'hardware-continuity-negative.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

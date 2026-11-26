"""Check an isolated export including current nonignored research evidence."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from tools.check_docs import Checker,destinations


if __name__=='__main__':
    names=subprocess.check_output(['git','ls-files','-co','--exclude-standard','-z'],cwd=ROOT).decode().split('\0')
    files={ROOT/name for name in names if name}
    checker=Checker(ROOT)
    # A linked raw observation can be intentionally Git-ignored. Export those
    # exact targets too; their data are not interpreted as instructions.
    for document in checker.documents():
        files.add(document)
        for destination,_ in destinations(document.read_text()):
            resolved=checker.local_target(document,destination)
            if resolved is not None:
                target,_=resolved
                if target.is_relative_to(ROOT)and target.exists():files.add(target)
    with tempfile.TemporaryDirectory(prefix='oa-spill-groups-docs-')as temporary:
        exported=Path(temporary)
        count=0
        for source in sorted(files):
            if not source.exists()or not source.resolve().is_relative_to(ROOT):continue
            target=exported/source.relative_to(ROOT)
            if source.is_dir():target.mkdir(parents=True,exist_ok=True);continue
            target.parent.mkdir(parents=True,exist_ok=True)
            # The checker is read-only; hardlinks preserve the exact raw bytes
            # without duplicating large binary evidence archives.
            os.link(source,target,follow_symlinks=True);count+=1
        print('Isolated export:',count,'files',flush=True)
        result=subprocess.run([sys.executable,str(exported/'tools/check_docs.py'),'--root',str(exported)],cwd=exported)
        raise SystemExit(result.returncode)

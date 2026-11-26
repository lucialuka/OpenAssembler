"""Check exceptional constant values and conditional chosen-branch types."""
import hashlib
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
sys.path[:0]=[str(ROOT),str(HERE)]
from oracle_probe import source
from openassembler import oracle


if __name__=='__main__':
    folder=HERE/'edges';folder.mkdir(exist_ok=True);rows=[]
    expressions=['0 % 0','-1 % 0','18446744073709551615U % 0','(1 % 0)<0',
        '(1?1:-1U)>0','(1?1:-1U)<0','(0?1U:-1)/2','(1?-1:1U)/2','(0?-1:1U)/2',
        '(0d7ff8000000000000/0.0)','(0d7ff8000000000000/1.0)',
        '0d7ff0000000000000/0d7ff0000000000000','0.0/0d7ff0000000000000',
        '(0d7ff0000000000001*0dfff8000000000002)',
        '(0d7ff8000000000001/0dfff8000000000002)',
        '((1<2)?-1:1U)>0','(-1&&1)','(-1||0)','1 % (0)',
        '(1<<64)','(1>>64)','(0d8000000000000000*0d7ff0000000000000)']
    for i,expression in enumerate(expressions):
        name='edge_'+str(i);text=source(name,expression);p=folder/(name+'.ptx');p.write_text(text);binary=p.with_suffix('.cubin')
        r=oracle.run_tool('ptxas','-arch=sm_80','-o',oracle.tool_path(binary),oracle.tool_path(p),check=False)
        raw=r.stdout+r.stderr;p.with_suffix('.tool.log').write_text(raw)
        row=dict(name=name,expression=expression,accepted=r.returncode==0,source_sha256=hashlib.sha256(text.encode()).hexdigest(),tool_log_sha256=hashlib.sha256(raw.encode()).hexdigest())
        if r.returncode==0:
            listing=oracle.disasm_text(binary);p.with_suffix('.sass').write_text(listing)
            row.update(cubin_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),sass_sha256=hashlib.sha256(listing.encode()).hexdigest())
        rows.append(row);print(row,flush=True)
    (HERE/'edge-oracle.json').write_text(json.dumps(dict(schema=1,results=rows,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='Exceptional and conditional expression toolkit controls; no native GPU or independent semantic claim'),indent=2)+'\n')

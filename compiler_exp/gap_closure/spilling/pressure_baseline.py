"""Retain baseline pressure/resource observations before changing allocation."""
from pathlib import Path
import json
import sys

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
sys.path[:0]=[str(ROOT),str(HERE)]
from probes import source
from openassembler.build import generate
from openassembler.ptx.parse import parse_module


if __name__=='__main__':
    records=[]
    for count,limit in((40,32),(96,32),(260,64)):
        text=source(count,limit)
        for level in('safe','mined','sched','opt'):
            try:
                code,labels,words,exits,regs=generate(parse_module(text).kernels['pressure'],'sm_80',level)
                records.append(dict(count=count,limit=limit,level=level,generated=True,registers=regs,within_declared_limit=regs<=limit))
            except Exception as exc:records.append(dict(count=count,limit=limit,level=level,generated=False,error=type(exc).__name__+': '+str(exc)))
    (HERE/'baseline-pressure.json').write_text(json.dumps(dict(schema=1,gap='G17',scope='Code generation only; ordinary builder additionally rejects maxnreg overflow; no native/GPU gate',results=records),indent=2)+'\n')
    print(records)

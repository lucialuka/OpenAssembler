"""A100 full compiler/control-flow gate; independent Python reference, no timings."""
from __future__ import annotations
import ast
import ctypes as C
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import struct
import subprocess

from openassembler import build
from openassembler.gpu.cuda_driver import Driver
from openassembler.pyfront import compile_file

HERE=Path(__file__).resolve().parent


def run(root):
    root=Path(root);root.mkdir(exist_ok=True)
    fixture=HERE/'fixture.py'
    tree=ast.parse(fixture.read_text())
    functions=[n for n in tree.body if isinstance(n,ast.FunctionDef)]
    for fn in functions:fn.decorator_list=[]
    env={'tid_x':lambda:thread}
    exec(compile(ast.Module(body=[tree.body[0],*functions],type_ignores=[]),str(fixture),'exec'),env)
    ptx=compile_file(fixture)[0]
    (root/'source.py').write_bytes(fixture.read_bytes());(root/'source.ptx').write_text(ptx)
    variants={};manifests={}
    for level in ('safe','mined','sched','opt'):
        output=root/level
        result=build.compile_paths(fixture,output,build.Options(opt_level=level,no_sass=False))
        assert result.rc==0,result
        manifest=json.loads((output/'manifest.json').read_text());entry=manifest['entries'][0]
        assert entry['native_ownership'],entry
        variants['oa_'+level]=(output/entry['cubin']).read_bytes();manifests[level]=manifest
        if level=='safe':variants['same_ptx_ptxas']=(output/entry['reference']).read_bytes()
    source_a=[-20,-7,-1,0,1,5,12,30,100,-90,7,3,-4,6,-8,40]
    source_b=[-4.0,-1.5,-0.0,0.0,0.25,1.0,4.0,16.0,-8.0,2.0,-2.0,0.5,3.0,6.0,8.0,10.0]
    driver=Driver();driver.open();records=[]
    try:
        device=asdict(driver.info());assert device['cc']=='8.0' and 'A100' in device['name'],device
        for variant,binary in variants.items():
            module=driver.load(binary)
            try:
                fn=driver.function(module,'evaluate')
                for iteration,n in enumerate((0,14,16)):
                    arrays={'a':[x+iteration for x in source_a],'b':[x+iteration*.25 for x in source_b],
                            'out':[-999]*16,'fo':[-999.0]*16}
                    expected={k:v[:] for k,v in arrays.items()}
                    for thread in range(32):env['evaluate'](*(expected[k] for k in arrays),n)
                    guard=b'\xa5'*256;addresses={};payloads={};allocations=[]
                    try:
                        for name,values in arrays.items():
                            payload=struct.pack('<16'+('f' if name in ('b','fo') else 'i'),*values)
                            allocation=driver.alloc(512+len(payload));allocations.append(allocation)
                            addresses[name]=allocation+256;payloads[name]=guard+payload+guard
                            driver.htod(allocation,payloads[name])
                        driver.launch(fn,(1,1,1),(32,1,1),[*(C.c_uint64(addresses[name]) for name in arrays),C.c_int32(n)])
                        actual={};guards={};unchanged={}
                        for name in arrays:
                            raw=driver.dtoh(addresses[name]-256,len(payloads[name]))
                            guards[name]=raw[:256]==raw[-256:]==guard
                            actual[name]=list(struct.unpack('<16'+('f' if name in ('b','fo') else 'i'),raw[256:-256]))
                            unchanged[name]=raw==payloads[name] if name in ('a','b') else actual[name][n:]==arrays[name][n:]
                        passed=actual==expected and all(guards.values()) and all(unchanged.values())
                        record={'variant':variant,'iteration':iteration,'n':n,'actual':actual,'expected':expected,
                                'inputs':arrays,'guards':guards,'immutable_inputs_and_tails':unchanged,'passed':passed,
                                'cubin_sha256':hashlib.sha256(binary).hexdigest(),'resources':driver.func_attrs(fn)}
                        records.append(record)
                        (root/'partial-results.json').write_text(json.dumps(records,indent=2)+'\n')
                        assert passed,record
                    finally:
                        for address in allocations:driver.free(address)
            finally:driver.unload(module)
    finally:driver.close()
    report={'schema':1,'gap':'G04/G06','passed':all(r['passed'] for r in records),'results':records,
            'device':device,'python':platform.python_version(),
            'ptxas':subprocess.check_output(['ptxas','--version'],text=True),
            'source_sha256':hashlib.sha256(fixture.read_bytes()).hexdigest(),
            'ptx_sha256':hashlib.sha256(ptx.encode()).hexdigest(),'comparisons':len(records)*64,
            'launches':len(records),'native_ownership_manifests':manifests,'performance':'not_measured'}
    (root/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    return report

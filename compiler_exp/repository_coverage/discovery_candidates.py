"""Compare forwarding, static-header and nvcc-based discovery on real fixtures."""
import json
from pathlib import Path
import sys
import tempfile
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from openassembler import oracle, project
from openassembler.cppfront import CppFrontendError, compile_to_ptx_module


def main():
    rows=[]
    oracle.SCRATCH.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=oracle.SCRATCH) as temporary:
        root=Path(temporary)
        (root/'qualifier.cuh').write_text('#define OA_GLOBAL __global__\n')
        fixtures={
            'direct':('extern "C" __global__ void k() {}',True),
            'macro':('#include "qualifier.cuh"\nextern "C" OA_GLOBAL void k() {}',True),
            'host':('int host() { return 1; }',False),
            'unused_macro':('#include "qualifier.cuh"\nint host() { return 1; }',False),
        }
        for name,(text,expected) in fixtures.items():
            path=root/(name+'.cu');path.write_text(text)
            for method in ('forward_all','static_header_closure','nvcc_ptx_entries'):
                started=time.perf_counter();error=None
                if method=='forward_all':candidate=True
                elif method=='static_header_closure':candidate=project.classify(path,path.name).language=='cuda'
                else:
                    try:_,names=compile_to_ptx_module(path);candidate=bool(names)
                    except CppFrontendError as exc:
                        if 'produced no __global__ kernels' not in str(exc):raise
                        candidate=False;error=str(exc)
                rows.append({'fixture':name,'method':method,'expected_kernel':expected,'candidate':candidate,
                             'seconds':time.perf_counter()-started,'error':error})
    result={'scope':'Four synthetic translation units; compile/discovery cost only, no GPU measurement',
            'results':rows,'decision':'Static header closure preserves tool-free inspection; nvcc validates candidates during compilation. Unused macros can overselect; build-graph integration remains open.'}
    target=Path(__file__).with_name('discovery-candidates.json')
    target.write_text(json.dumps(result,indent=2)+'\n');print(target)


if __name__=='__main__':main()

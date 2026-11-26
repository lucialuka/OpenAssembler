"""Paired host code-generation cost; distinct from CUDA kernel intervals."""
import cProfile
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import random
import statistics
import sys
import time
import types

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
sys.path[:0]=[str(ROOT),str(HERE)]
from candidate import install as install_candidate
from hardware_probes import cases
from openassembler import build
from openassembler.backend import ra
from openassembler.ptx.parse import parse_module


def install_baseline():
    name='openassembler.backend.spill';module=types.ModuleType(name);module.__file__=str(HERE/'baseline-spill.py')
    sys.modules[name]=module;exec(compile((HERE/'baseline-spill.py').read_bytes(),module.__file__,'exec'),module.__dict__)


if __name__=='__main__':
    fixtures=[case for case in cases()if case['name']in('scalar96','pred24','group4_auto')]
    build.encoder('sm_80');rows=[];samples={};rng=random.Random(1907)
    for case in fixtures:
        kernel=parse_module(case['source']).kernels[case['name']]
        names=['baseline','candidate'];samples[case['name']]={name:[]for name in names}
        for repetition in range(3):
            rng.shuffle(names)
            for name in names:
                if name=='baseline':install_baseline()
                else:install_candidate()
                resources={};start=time.perf_counter()
                code,labels,words,_,registers=build.generate(kernel,'sm_80','opt',resource_info=resources)
                elapsed=time.perf_counter()-start;samples[case['name']][name].append(elapsed)
                rows.append(dict(name=case['name'],compiler=name,repetition=repetition,seconds=elapsed,resources=resources,registers=registers,
                                 native_sha256=hashlib.sha256(b''.join(word.to_bytes(16,'little')for word in words)).hexdigest()))
                print(rows[-1],flush=True)
    install_candidate();from openassembler.backend.pressure import schedule
    case=next(case for case in fixtures if case['name']=='group4_auto')
    selector=ra.Selector(parse_module(case['source']).kernels[case['name']]);ops=ra.coalesce(ra.dce(selector.run()))
    profiler=cProfile.Profile();start=time.perf_counter();profiler.runcall(schedule,ops);schedule_seconds=time.perf_counter()-start
    profiler.dump_stats(str(HERE/'pressure-profile.pstats'))
    result=dict(schema=1,passed=True,python=sys.version,platform=platform.platform(),records=rows,samples_seconds=samples,
                medians_seconds={name:{compiler:statistics.median(values)for compiler,values in variants.items()}for name,variants in samples.items()},
                source_hashes={name:hashlib.sha256((HERE/name).read_bytes()).hexdigest()for name in('pressure.py','spill.py','baseline-spill.py','compile_cost.py')},
                profiled_schedule_seconds=schedule_seconds,
                scope='Three paired opt code-generation/encoding samples per exact source in randomized order, warm encoder, parsed PTX input; excludes parsing, toolkit/packaging, GPU/whole-application and emulator work. One separately instrumented schedule profile; concurrent host work is not controlled.')
    (HERE/'compile-cost.json').write_text(json.dumps(result,indent=2)+'\n')

"""Record source-build attribution availability without requiring GPU execution."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from openassembler import build, oracle


def run(output):
    source = ROOT / 'examples/div_by_const.ptx'
    options = build.Options(no_sass=True)
    started = time.perf_counter()
    result = build.build_source(source, 'ptx', 'division', options, single=True)
    report = {
        'schema': 1, 'python': sys.version, 'platform': platform.platform(),
        'scope': 'worker-local source compilation, no execution or performance claim',
        'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'compiler_sha256': hashlib.sha256(Path(build.__file__).read_bytes()).hexdigest(),
        'ptxas_available': oracle.available('ptxas'), 'rc': result.rc,
        'error': result.error, 'external_wall_ms': (time.perf_counter() - started) * 1000,
        'compile_stage_ms': getattr(result, 'compile_stage_ms', None),
        'compile_total_ms': getattr(result, 'compile_total_ms', None),
        'cubin_sha256': {name: hashlib.sha256(value).hexdigest() for name, value in result.files.items()
                         if name.startswith('cubin/') and isinstance(value, bytes)},
    }
    path = Path(output)
    with path.open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output')
    run(parser.parse_args().output)

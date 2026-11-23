"""Replay G01 on A100 with exact reference, original and forced-path controls.

Run: modal run compiler_exp/gap_closure/verification/run_modal.py
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import modal

HERE = Path(__file__).resolve().parent
ROOT = Path.cwd()  # Modal imports this module at /root/run_modal.py remotely.
app = modal.App('openassembler-exact-division-gap')
image = modal.Image.from_registry('nvidia/cuda:12.8.1-devel-ubuntu22.04', add_python='3.12')


@app.function(image=image, gpu='A100-40GB', timeout=600, max_containers=1, retries=0)
def execute(files):
    import base64
    import ctypes as C
    from dataclasses import asdict, replace
    import io
    import os
    import struct
    import subprocess
    import sys
    import tempfile
    import zipfile

    with tempfile.TemporaryDirectory(prefix='oa-g01-') as temp:
        root = Path(temp)
        for name, data in files.items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        (root / 'env/cuda.sh').chmod(0o700)
        os.chdir(root)
        sys.path[:0] = [str(root), str(root / 'compiler_exp/gap_closure/verification')]
        from probe import exact_division, inputs
        from openassembler import build
        from openassembler.container.elf import Elf
        from openassembler.emu.program import load_listing
        from openassembler.gpu.cuda_driver import Driver
        from openassembler.isa.encode import Encoder
        from openassembler.isa.word import control
        source = root / 'corpus/sm_80/ptx/p_div_rn_f32.ptx'
        frozen = (root / 'corpus/sm_80/cubin/p_div_rn_f32.cubin').read_bytes()
        listing = (root / 'corpus/sm_80/sass/p_div_rn_f32.sass').read_text()
        program = load_listing(listing)
        encoder = Encoder()
        def patch(address, text, target=None, wait=None):
            insn = program.insns[address]
            ctl = control(insn.word)
            if wait is not None:
                ctl = replace(ctl, wait=ctl.wait | wait)
            word = (encoder.encode_branch(text, address, target, ctl) if target is not None
                    else encoder.encode(text, ctl))
            elf = Elf.parse(frozen)
            data = bytearray(elf.section('.text.p_div_rn_f32').data)
            data[address:address + 16] = word.to_bytes(16, 'little')
            elf.replace_data('.text.p_div_rn_f32', bytes(data))
            return elf.serialize()
        variants = {'frozen': frozen,
                    'frozen_force_fast': patch(0xf0, 'BRA `(.L_x_1)', 0x120),
                    'frozen_force_slow': patch(0xf0, 'NOP'),
                    # Diagnostic only: replace the final correction with P0 capture.
                    # Its low result word is discarded; high word is 0 iff FCHK true.
                    'frozen_fchk': patch(0xe0, 'SEL R7, RZ, 0x1, P0', wait=2)}
        fresh = root / 'fresh.cubin'
        completed = subprocess.run(['ptxas', '-arch=sm_80', '-O3', str(source), '-o', str(fresh)],
                                   capture_output=True, text=True)
        completed.check_returncode()
        variants['fresh_ptxas'] = fresh.read_bytes()
        for level in ('safe', 'mined', 'sched', 'opt'):
            directory = root / ('native_' + level)
            compilation = build.compile_paths(source, directory, build.Options(opt_level=level))
            if compilation.rc:
                raise RuntimeError(f'OA compile failed: {compilation}')
            entry = json.loads((directory / 'manifest.json').read_text())['entries'][0]
            variants['oa_' + level] = (directory / entry['cubin']).read_bytes()
        rows = inputs()
        expected = [exact_division(a, b) for a, b, _, _ in rows]
        padded = rows + [rows[0]] * (-len(rows) % 32)
        raw_input = b''.join(struct.pack('<4I', *row) for row in padded)
        guard = b'\xa5' * 256
        driver = Driver()
        driver.open()
        records = []
        archive_bytes = io.BytesIO()
        try:
            device = asdict(driver.info())
            if device['cc'] != '8.0' or 'A100' not in device['name']:
                raise RuntimeError(f'A100 required, got {device}')
            with zipfile.ZipFile(archive_bytes, 'w', zipfile.ZIP_DEFLATED) as archive:
                archive.writestr('source.ptx', source.read_bytes())
                archive.writestr('inputs.json', json.dumps(rows))
                archive.writestr('expected.json', json.dumps(expected))
                for name, binary in variants.items():
                    path = root / (name + '.cubin')
                    path.write_bytes(binary)
                    dis = subprocess.run(['nvdisasm', '-hex', '-c', str(path)], capture_output=True, text=True)
                    dis.check_returncode()
                    archive.writestr(name + '.cubin', binary)
                    archive.writestr(name + '.sass.txt', dis.stdout)
                    module = driver.load(binary)
                    try:
                        function = driver.function(module, 'p_div_rn_f32')
                        src, dst = driver.alloc(len(raw_input) + 512), driver.alloc(len(raw_input) + 512)
                        try:
                            driver.htod(src, guard + raw_input + guard)
                            driver.htod(dst, guard + b'\x3c' * len(raw_input) + guard)
                            # Corpus kernels index tid.x only; one warp per launch,
                            # advance pointers explicitly to preserve the frozen ABI.
                            for offset in range(0, len(padded), 32):
                                driver.launch(function, (1, 1, 1), (32, 1, 1),
                                              [C.c_uint64(dst + 256 + offset * 16),
                                               C.c_uint64(src + 256 + offset * 16),
                                               C.c_uint32(0), C.c_uint32(0)])
                            output = driver.dtoh(dst, len(raw_input) + 512)
                            words = [struct.unpack_from('<2I', output, 256 + 16 * i) for i in range(len(rows))]
                            mismatches = [i for i, (word, truth) in enumerate(zip(words, expected)) if word[0] != truth]
                            archive.writestr(name + '.outputs.json', json.dumps(words))
                            records.append({'variant': name, 'diagnostic_only': name == 'frozen_fchk',
                                            'cubin_sha256': hashlib.sha256(binary).hexdigest(),
                                            'resources': driver.func_attrs(function), 'pairs': len(rows),
                                            'mismatches': None if name == 'frozen_fchk' else len(mismatches),
                                            'first_mismatches': [] if name == 'frozen_fchk' else mismatches[:20],
                                            'counterexample_output': list(words[0]),
                                            'guards': output[:256] == output[-256:] == guard,
                                            'inputs_unchanged': driver.dtoh(src, len(raw_input) + 512) == guard + raw_input + guard,
                                            'fchk_true_rows': [i for i, word in enumerate(words) if word[1] == 0]
                                                if name == 'frozen_fchk' else None})
                        finally:
                            driver.free(src)
                            driver.free(dst)
                    finally:
                        driver.unload(module)
        finally:
            driver.close()
        report = {'schema': 1, 'completed_utc': datetime.now(timezone.utc).isoformat(),
                  'device': device, 'python': sys.version,
                  'ptxas': subprocess.check_output(['ptxas', '--version'], text=True),
                  'scope': 'Exact division reference; frozen, forced-path and fresh OA/ptxas controls; no timing',
                  'pairs': len(rows), 'expected_counterexample': expected[0], 'results': records,
                  'source_hashes': {n: hashlib.sha256(v).hexdigest() for n, v in files.items()}}
        gates = [r for r in records if r['variant'] not in ('frozen_force_fast', 'frozen_fchk')]
        report['passed'] = all(r['mismatches'] == 0 and r['guards'] and r['inputs_unchanged'] for r in gates)
        return report, base64.b64encode(archive_bytes.getvalue()).decode()


@app.local_entrypoint()
def main():
    import base64
    import sys
    sys.path.insert(0, str(ROOT))
    from openassembler.transport import runtime_files
    names = ['corpus/sm_80/ptx/p_div_rn_f32.ptx',
             'corpus/sm_80/sass/p_div_rn_f32.sass',
             'corpus/sm_80/cubin/p_div_rn_f32.cubin',
             'compiler_exp/gap_closure/verification/probe.py']
    files = runtime_files(native=True)
    files.update({name: (ROOT / name).read_bytes() for name in names})
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    report, payload = execute.remote(files)
    archive = HERE / ('a100-' + stamp + '.artifacts.zip')
    archive.write_bytes(base64.b64decode(payload))
    report['artifact_archive'] = {'path': archive.name, 'sha256': hashlib.sha256(archive.read_bytes()).hexdigest()}
    report['harness_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    output = HERE / ('a100-' + stamp + '.json')
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(output)
    for record in report['results']:
        print(record['variant'], record['mismatches'], record['counterexample_output'])

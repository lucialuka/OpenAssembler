"""Replay preceding Boolean originals and inherit exact retained backend proofs."""
import argparse
import hashlib
import json
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('report', type=Path)
    args = parser.parse_args()
    report_path = args.report.resolve()
    report = json.loads(report_path.read_text())
    assert report['passed']
    from openassembler import build
    from openassembler.pyfront import compile_file
    from openassembler.ptx.parse import parse_module
    from openassembler.container.cubin import text_words, regcount
    from openassembler.backend.storage import local_frame_inventory
    from openassembler.native_ownership import validate
    prior_dir = HERE.parents[1]/'python_boolean_values/promotion'
    validation_path = prior_dir/'validation.json'
    prior = json.loads(validation_path.read_text())
    assert prior['passed']
    allowed = {'openassembler/pyfront/compile.py': 'pyfront',
               'openassembler/pyfront/helpers.py': 'helpers',
               'openassembler/pyfront/modules.py': 'modules'}
    changed = {name for name, digest in prior['source_hashes'].items() if sha(ROOT/name) != digest}
    assert changed == set(allowed), changed
    for name, suffix in allowed.items():
        assert sha(HERE/('baseline-'+suffix+'.py')) == prior['source_hashes'][name]
    inherited = {name: digest for name, digest in prior['source_hashes'].items()
                 if name.startswith(('openassembler/', 'specs/')) and name not in allowed}
    assert all(sha(ROOT/name) == digest == report['source_hashes'][name] for name, digest in inherited.items())
    chain_path = prior_dir/'application-continuity.json'
    chain = json.loads(chain_path.read_text())
    assert chain['passed'] and chain['native_comparison_count'] == chain['native_exact_count'] == 3254
    assert sha(chain_path) == prior['evidence_sha256']['application-continuity.json']
    assert all(sha(ROOT/name) == digest for name, digest in chain['backend_hashes'].items() if name not in allowed)
    assert all(sha(ROOT/name) == digest for name, digest in chain['retained_hashes'].items())
    assert all(sha(prior_dir/name) == digest for name, digest in prior['evidence_sha256'].items())
    proof_path = prior_dir/'promotion-audit.json'
    proof = json.loads(proof_path.read_text())
    assert proof['passed'] and proof['bytes_checked'] == 16957440
    old_report_path = ROOT/proof['report']
    old = json.loads(old_report_path.read_text())
    assert sha(old_report_path) == proof['report_sha256']
    archive_path = old_report_path.parent/old['archive']['path']
    assert sha(archive_path) == old['archive']['sha256']
    rows, sources = [], []
    with tempfile.TemporaryDirectory(prefix='oa-logical-helper-prior-') as temporary, zipfile.ZipFile(archive_path) as archive:
        configurations = sorted({name.split('/')[0] for name in archive.namelist() if '/original/' in name})
        for configuration in configurations:
            folder = Path(temporary)/configuration
            prefix = configuration+'/original/'
            for name in archive.namelist():
                if name.startswith(prefix):
                    path = folder/name.removeprefix(prefix)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(archive.read(name))
            entry = 'kernels/entry.py' if configuration == 'remote_package' else 'entry.py'
            text, names = compile_file(folder/entry)
            for level in ('safe', 'mined', 'sched', 'opt'):
                prefix = configuration+'/'+level+'/'
                assert text.encode() == archive.read(prefix+'source.ptx'), (configuration, level)
                binary = archive.read(prefix+'oa.cubin')
                manifest = json.loads(archive.read(prefix+'manifest.json'))
                validate(manifest['entries'][0], binary)
                for name, kernel in parse_module(text).kernels.items():
                    resources = {}
                    _, _, words, _, registers = build.generate(kernel, 'sm_80', level, resource_info=resources)
                    assert words == text_words(binary, name) and registers == regcount(binary, name)
                    assert resources['local_bytes'] == local_frame_inventory(binary)[name] == 0
                    rows.append(dict(configuration=configuration, entry=name, level=level, registers=registers, passed=True))
            sources.append(dict(configuration=configuration, kernels=names, ptx_sha256=hashlib.sha256(text.encode()).hexdigest()))
    assert len(rows) == 240
    result = dict(schema=1, passed=True, native_comparison_count=3494,
                  inherited_native_comparison_count=3254, replayed_native_comparison_count=240,
                  native_comparisons=rows, source_comparisons=sources, unchanged_backend_hashes=inherited,
                  changed_production_inputs=sorted(changed), added_production_inputs=[],
                  previous_validation=str(validation_path.relative_to(ROOT)), previous_validation_sha256=sha(validation_path),
                  inherited_application_chain=str(chain_path.relative_to(ROOT)), inherited_application_chain_sha256=sha(chain_path),
                  preceding_hardware_report=str(old_report_path.relative_to(ROOT)), preceding_hardware_report_sha256=sha(old_report_path),
                  preceding_archive_sha256=sha(archive_path), validated_backend_report=str(report_path.relative_to(ROOT)),
                  validated_backend_report_sha256=sha(report_path), auditor_sha256=sha(Path(__file__)),
                  scope='3254 exact preceding native comparisons inherited only after exact proof/archive and unchanged executable/backend input identities; 240 preceding Boolean-value originals freshly regenerate to identical complete PTX/native words/registers/frames. New Boolean-helper semantics have their separate original-source A100 gate. No repeat GPU execution or timing claim')
    (HERE/'application-continuity.json').write_text(json.dumps(result, indent=2)+'\n')
    print('Passed 240 regenerated native outputs; 3254 exact comparisons inherited under unchanged backend identities')

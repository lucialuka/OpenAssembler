"""Compilation attribution must describe actual work and preserve artifacts."""
import base64
import json
from pathlib import Path
import pytest

from openassembler import build, oracle
from openassembler.frontend import FrontendError

ROOT = Path(__file__).resolve().parents[1]


def test_failed_frontend_retains_completed_phase_time(tmp_path, monkeypatch):
    import openassembler.frontend as frontend
    source = tmp_path / 'broken.py'
    source.write_text('uncompilable')
    def fail(*args, **kwargs):
        raise FrontendError('deliberate frontend failure')
    monkeypatch.setattr(frontend, 'to_ptx_module', fail)
    result = build.build_source(source, 'python', 'broken', build.Options(), single=True)
    assert result.rc == 1 and 'deliberate' in result.error
    assert result.compile_stage_ms['source_hash'] > 0
    assert result.compile_stage_ms['frontend'] > 0
    assert result.compile_stage_ms['backend'] == result.compile_stage_ms['ptxas_container'] == 0
    assert sum(result.compile_stage_ms.values()) == pytest.approx(result.compile_total_ms)
    assert not result.files


def _toolkit():
    if not oracle.available('ptxas'):
        pytest.skip('requires CUDA ptxas for real bundle attribution')


def test_source_manifest_times_exact_work_with_binary_only_build():
    _toolkit()
    result = build.build_source(ROOT / 'examples/div_by_const.ptx', 'ptx', 'division',
                                build.Options(no_sass=True, verify=True), single=True)
    assert result.rc == 0, result.error
    entry = result.entry(True)
    stages = entry['compile_stage_ms']
    assert all(value >= 0 for value in stages.values())
    assert all(stages[name] > 0 for name in ('frontend','ptx_parse','backend','verification',
                                           'ptxas_container','packaging_ownership'))
    assert stages['disassembly'] == 0
    assert sum(stages.values()) == pytest.approx(entry['compile_total_ms'])
    assert entry['native_ownership'] and entry['kernels'][0]['verify'] == 'ok'


def test_cache_hit_reports_no_replayed_compilation_timings(tmp_path):
    _toolkit()
    from openassembler.modelfront.kernels import build_bundle
    from openassembler.modelfront.runtime import validate_bundle
    from tests.test_model_runtime import plan
    cache = tmp_path / 'cache'
    cold, warm = tmp_path / 'cold', tmp_path / 'warm'
    build_bundle(plan(), {}, cold, cache_dir=cache, sass_listings=False)
    build_bundle(plan(), {}, warm, cache_dir=cache, sass_listings=False)
    miss = json.loads((cold / 'model-cache-result.json').read_text())
    hit = json.loads((warm / 'model-cache-result.json').read_text())
    assert not miss['hit'] and hit['hit']
    assert miss['source_builds'] and hit['source_builds'] == []
    assert miss['compile_stage_ms']['native_build_and_weights'] > 0
    assert hit['compile_stage_ms']['native_build_and_weights'] == 0
    assert hit['compile_stage_ms']['restore_weights_and_validate'] > 0
    assert all(value >= 0 for row in (miss,hit) for value in row['compile_stage_ms'].values())
    assert validate_bundle(cold)[2] == validate_bundle(warm)[2]


def test_model_worker_carries_nested_cache_and_source_attribution(tmp_path, monkeypatch):
    pytest.importorskip('torch')
    _toolkit()
    from openassembler.modelfront.frontend import load_config
    from openassembler.modelfront.worker import process
    from openassembler.torchfront.archive import pack
    (tmp_path/'model.py').write_text('import torch\ndef create_model():\n    return torch.nn.Linear(3,2)\n')
    (tmp_path/'openassembler.toml').write_text('[model]\nentrypoint="model.py"\ninput_shapes=[[2,3]]\nsass_listings=false\n')
    monkeypatch.setenv('OPENASSEMBLER_MODEL_CACHE', str(tmp_path/'cache'))
    config=load_config(tmp_path)
    source=pack({p.name:p.read_bytes() for p in tmp_path.iterdir() if p.is_file()})
    result=process({'action':'compile','config':config,'source':base64.b64encode(source).decode(),
                    'source_digest':'measurement-test'})
    assert 'error' not in result, result.get('error')
    report=result['report']
    stages=report['compile_stage_ms']
    assert {'repository_native_build','application_validation','cpu_examples','source_unpack'} <= stages.keys()
    assert sum(stages.values()) <= report['compile_worker_total_ms']
    assert 'nested' in report['compile_stage_scope']
    assert report['model_bundle_cache']['source_builds']

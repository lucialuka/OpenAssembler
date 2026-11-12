"""Native model repository routing and transport ownership."""
import json
from pathlib import Path

import pytest

from openassembler.cli import main
from openassembler.modelfront.frontend import ModelError, load_config, resolve_model_output
from openassembler.repository import select_frontend


def repository(tmp_path):
    root = tmp_path / 'repo'
    root.mkdir()
    (root / 'model.py').write_text('def create_model(): pass\n')
    (root / 'openassembler.toml').write_text('[model]\nentrypoint="model.py"\ninput_shapes=[[17,8]]\n')
    return root


def test_configured_model_auto_routes_to_oa_without_torch_import(tmp_path):
    root = repository(tmp_path)
    assert select_frontend(root, root / 'output', 'auto')[0] == 'model'
    assert select_frontend(root, root / 'output', 'torch')[0] == 'model'
    assert select_frontend(root, root / 'output', 'native')[0] == 'native'
    assert load_config(root)['math'] == 'fp32'


@pytest.mark.parametrize('backend', ['modal', 'training-a100', 'local'])
def test_model_dry_run_is_read_only_and_native_on_each_transport(tmp_path, capsys, backend):
    root = repository(tmp_path)
    assert main(['cnr', str(root), '--backend', backend, '--verify', '--dry-run']) == 0
    output = capsys.readouterr().out
    assert 'OA GPU code only' in output and 'nothing was run' in output
    assert not (root / 'output').exists()


@pytest.mark.parametrize('frontend', ['triton', 'inductor'])
def test_vendor_frontends_reject_before_any_work(tmp_path, frontend, capsys):
    root = repository(tmp_path)
    assert main(['compile', str(root), '--frontend', frontend]) == 1
    assert 'disabled' in capsys.readouterr().err
    assert not (root / 'output').exists()


def test_legacy_torch_application_cannot_reach_inductor(tmp_path, monkeypatch, capsys):
    root = repository(tmp_path)
    (root / 'openassembler.toml').write_text('[application]\nfrontend="torch"\nentrypoint="model.py"\n')
    import openassembler.torchfront.frontend as old
    monkeypatch.setattr(old, 'compile_application', lambda *a, **k: pytest.fail('vendor compiler invoked'))
    assert main(['compile', str(root), '--frontend', 'torch', '--backend', 'modal']) == 1
    assert 'legacy Inductor' in capsys.readouterr().err


def test_vendor_kernel_fallback_rejects(tmp_path, capsys):
    source = tmp_path / 'kernel.ptx'
    source.write_text('.version 8.7\n.target sm_80\n.address_size 64\n.visible .entry k() { ret; }\n')
    assert main(['compile', str(source), '--fallback', 'ptxas', '--dry-run']) == 1
    assert 'fallback is disabled' in capsys.readouterr().err


def test_model_config_rejects_precision_and_path_escape(tmp_path):
    root = repository(tmp_path)
    (root / 'openassembler.toml').write_text('[model]\nentrypoint="../private.py"\ninput_shapes=[[17,8]]\n')
    with pytest.raises(ModelError, match='inside the repository'):
        load_config(root)
    (root / 'openassembler.toml').write_text('[model]\nentrypoint="model.py"\ninput_shapes=[[17,8]]\nmath="fp16"\n')
    with pytest.raises(ModelError, match='fp32'):
        load_config(root)


def test_model_output_does_not_select_another_repository(tmp_path):
    source = repository(tmp_path)
    out = source / 'output'
    out.mkdir()
    (out / 'model.json').write_text(json.dumps({'source_root': '/another/repository'}))
    with pytest.raises(ModelError, match='another source'):
        resolve_model_output(source)


def test_remote_compiler_bundle_contains_native_dispatch_tables():
    from openassembler.transport import runtime_files
    files = runtime_files(native=True)
    assert 'openassembler/kernels/sm80_gemm_dispatch.json' in files
    assert 'openassembler/kernels/sm80_rowwise_dispatch.json' in files
    assert 'openassembler/backend/ptxas_fragments.json' in files


@pytest.mark.parametrize('value', ['true', '1'])
def test_storage_reuse_is_an_explicit_boolean_model_contract(tmp_path, value):
    root = repository(tmp_path)
    assert load_config(root)['reuse_storage'] is False
    path = root / 'openassembler.toml'
    path.write_text(path.read_text() + 'reuse_storage=' + value + chr(10))
    if value == 'true':
        assert load_config(root)['reuse_storage'] is True
    else:
        with pytest.raises(ModelError, match='reuse_storage'):
            load_config(root)


@pytest.mark.parametrize('action', ['compile', 'run'])
def test_ssh_transports_frozen_recipe_contracts_for_both_actions(tmp_path, monkeypatch, action):
    import io
    import zipfile
    from openassembler.modelfront import frontend
    sent = []

    def remote(host, req, res, raw, module, budget):
        assert host == 'a100-test'
        assert module == 'openassembler.modelfront.worker'
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            sent.extend(archive.namelist())
        res.write_text(json.dumps({'passed': True}))

    monkeypatch.setattr(frontend, 'ssh_request', remote)
    result = frontend._request({'action': action},
                               {'build_backend': 'ssh', 'build_host': 'a100-test', 'budget': 900})
    assert result['passed'] is True
    assert 'openassembler/kernels/sm80_gemm_dispatch.json' in sent
    assert 'openassembler/backend/ptxas_fragments.json' in sent


def test_deployment_archive_excludes_reference_and_untracked_modules(tmp_path, monkeypatch):
    from openassembler.modelfront.frontend import deployment_files, export_model
    from openassembler.modelfront import runtime
    from openassembler.torchfront.archive import unpack
    ledger = {'cubin': 'native/cubin/owned.cubin', 'tensor_kernels': {
        'tensor': {'module': 'native_tensor/tensor/native.cubin'}}}
    plan = {'parameters': [{'name': 'w'}]}
    monkeypatch.setattr(runtime, 'validate_bundle', lambda root: (plan, ledger, b'oa'))
    files = {'plan.json': '{}', 'native-ledger.json': '{}', 'model.json': '{}',
             'weights.json': json.dumps({'w': {'path': 'weights/w.bin'}}),
             'weights/w.bin': 'data', 'native/cubin/owned.cubin': 'oa',
             'native_tensor/tensor/native.cubin': 'oa-tensor',
             'model-source.zip': 'cpu-source', 'native/cubin/untracked.cubin': 'vendor',
             'native_tensor/tensor/reference.cubin': 'vendor-reference',
             'native/.build/reference/owned.cubin': 'vendor-reference', 'source.ptx': 'ptx'}
    for name, value in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value)
    deployed = deployment_files(tmp_path)
    assert not any('reference' in name or 'untracked' in name or name.endswith('.ptx') for name in deployed)
    assert 'model-source.zip' not in deployed
    assert deployment_files(tmp_path, verification=True)['model-source.zip'] == b'cpu-source'
    destination = export_model(tmp_path, tmp_path / 'deployment.zip')
    unpack(destination.read_bytes(), tmp_path / 'unpacked')
    assert json.loads((tmp_path / 'unpacked/model-artifacts.json').read_text())['schema'] == 1

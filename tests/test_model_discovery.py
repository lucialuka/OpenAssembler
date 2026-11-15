"""Configuration-free discovery must preserve exact contracts and source files."""
import base64
import json
from pathlib import Path
import subprocess
import sys
import tomllib

import pytest

from openassembler.cli import main
from openassembler.modelfront.discovery import config_text, discover_model
from openassembler.modelfront.frontend import ModelError, load_config
from openassembler.repository import select_frontend


NET = '''import torch
class Net(torch.nn.Module):
    def __init__(self, width=8):
        super().__init__()
        self.fc = torch.nn.Linear(width, 4)
    def forward(self, x):
        return self.fc(x)
'''


def write(root, name, text):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def test_model_class_and_named_examples_need_no_config_or_import(tmp_path, capsys):
    write(tmp_path, 'net.py', NET + 'example_inputs = (torch.randn(17, 8),)\nraise RuntimeError("not imported")\n')
    assert select_frontend(tmp_path, tmp_path / 'output')[0] == 'model'
    assert load_config(tmp_path)['input_shapes'] == [[17, 8]]
    assert main(['inspect', str(tmp_path), '--json']) == 0
    report = json.loads(capsys.readouterr().out)
    assert report['model_discovery']['ready']
    assert main(['compile', str(tmp_path), '--backend', 'modal', '--dry-run']) == 0
    assert 'forward model' in capsys.readouterr().out
    assert sorted(p.name for p in tmp_path.iterdir()) == ['net.py']


def test_observed_call_preserves_constructor_arguments_and_shapes(tmp_path):
    write(tmp_path, 'network.py', NET)
    write(tmp_path, 'evaluate.py', '''from network import Net as Network
import torch as t
def main():
    model = Network(width=12).eval()
    x = t.zeros((5, 12), dtype=t.float32)
    output = model(x)
''')
    config = load_config(tmp_path)
    assert (config['entrypoint'], config['factory']) == ('network.py', 'Net')
    assert config['factory_kwargs'] == {'width': 12}
    assert config['input_shapes'] == [[5, 12]]


def test_nested_packages_reexports_and_relative_imports(tmp_path):
    write(tmp_path, 'src/models/__init__.py', 'from .network import Net\n')
    write(tmp_path, 'src/models/network.py', NET)
    write(tmp_path, 'evaluate.py', 'from models import Net\nimport torch\nm = Net()\nx = torch.randn(3,8)\ny = m(x)\n')
    assert load_config(tmp_path)['entrypoint'] == 'src/models/network.py'
    assert load_config(tmp_path)['input_shapes'] == [[3, 8]]


def test_required_constructor_and_multiple_inputs(tmp_path):
    write(tmp_path, 'net.py', NET.replace('width=8', 'width'))
    write(tmp_path, 'eval.py', 'from net import Net\nimport torch\nm = Net(8)\nx=torch.randn(2,8)\ny=m(x,x)\n')
    config = load_config(tmp_path)
    assert config['factory_args'] == [8]
    assert config['input_shapes'] == [[2, 8], [2, 8]]


def test_tuple_constructor_arguments_are_not_changed_to_lists(tmp_path):
    write(tmp_path, 'net.py', NET.replace('width=8', 'width'))
    write(tmp_path, 'evaluate.py', 'from net import Net\nimport torch\nm=Net((8,4))\ny=m(torch.randn(2,8))\n')
    with pytest.raises(ModelError):
        load_config(tmp_path)


def test_no_synthetic_batch_or_training_objective_is_guessed(tmp_path):
    write(tmp_path, 'net.py', NET)
    with pytest.raises(ModelError, match='exact FP32 input shapes'):
        load_config(tmp_path)
    config = load_config(tmp_path, input_shapes=[[128, 8]])
    assert config['mode'] == 'inference' and config['math'] == 'fp32'
    assert config['input_shapes'] == [[128, 8]]


def test_ambiguous_models_or_shapes_require_overrides(tmp_path):
    write(tmp_path, 'a.py', NET + 'example_inputs = [torch.randn(2,8)]\n')
    write(tmp_path, 'b.py', NET + 'example_inputs = [torch.randn(3,8)]\n')
    with pytest.raises(ModelError):
        load_config(tmp_path)
    config = load_config(tmp_path, entrypoint='b.py:Net', input_shapes=[[9, 8]])
    assert config['entrypoint'] == 'b.py' and config['input_shapes'] == [[9, 8]]
    write(tmp_path, 'evaluate.py', 'from a import Net\nimport torch\nm=Net()\ny=m(torch.randn(2,8))\nz=m(torch.randn(3,8))\n')
    with pytest.raises(ModelError, match='multiple'):
        load_config(tmp_path)
    assert load_config(tmp_path, entrypoint='a.py:Net', input_shapes=[[4, 8]])['input_shapes'] == [[4, 8]]


def test_a_factory_does_not_hide_an_independent_model(tmp_path):
    write(tmp_path, 'a.py', NET + 'def create_model(): return Net()\nexample_inputs = [torch.randn(2,8)]\n')
    write(tmp_path, 'b.py', NET)
    with pytest.raises(ModelError):
        load_config(tmp_path)


def test_tabular_defaults_do_not_override_conflicting_batch_sizes(tmp_path):
    write(tmp_path, 'net.py', NET)
    write(tmp_path, 'train.py', 'p.add_argument("--batch-size", default=128)\nq.add_argument("--batch-size", default=64)\n')
    with pytest.raises(ModelError):
        load_config(tmp_path)


def test_tensor_dtype_is_not_silently_changed(tmp_path):
    write(tmp_path, 'net.py', NET + 'example_input=torch.randn(2,8,dtype=torch.float64)\n')
    with pytest.raises(ModelError):
        load_config(tmp_path)


def test_submodule_unit_test_is_not_mistaken_for_whole_application(tmp_path):
    write(tmp_path, 'child.py', NET.replace('Net', 'Child'))
    write(tmp_path, 'app.py', 'from child import Child\nimport torch\nclass App(torch.nn.Module):\n def __init__(self):\n  super().__init__()\n  self.child=Child()\n def forward(self,x): return self.child(x)\n')
    write(tmp_path, 'test_child.py', 'from child import Child\nimport torch\nm=Child()\ny=m(torch.randn(2,8))\n')
    report = discover_model(tmp_path)
    assert not report['ready']
    assert load_config(tmp_path, input_shapes=[[3, 8]])['factory'] == 'App'


def test_literal_batch_default_plus_direct_input_linear(tmp_path):
    write(tmp_path, 'net.py', NET)
    write(tmp_path, 'train.py', 'import argparse\np=argparse.ArgumentParser()\np.add_argument("--batch-size",type=int,default=128)\n')
    config = load_config(tmp_path)
    assert config['input_shapes'] == [[128, 8]] and config['mode'] == 'inference'


@pytest.mark.parametrize('shapes', [[[0, 8]], [[True, 8]], [[2**29]], 'auto', [], [8]])
def test_invalid_shape_overrides_reject_without_output(tmp_path, shapes):
    write(tmp_path, 'net.py', NET)
    assert main(['compile', str(tmp_path), '--input-shapes', json.dumps(shapes), '--dry-run']) == 1
    assert not (tmp_path / 'output').exists()


def test_explicit_config_wins_and_cli_overrides_do_not_edit_it(tmp_path):
    write(tmp_path, 'net.py', NET)
    path = write(tmp_path, 'openassembler.toml', '[model]\nentrypoint="net.py"\nfactory="Net"\ninput_shapes=[[2,8]]\nmode="training"\n')
    original = path.read_bytes()
    assert load_config(tmp_path)['mode'] == 'training'
    assert load_config(tmp_path, input_shapes=[[3, 8]])['input_shapes'] == [[3, 8]]
    assert path.read_bytes() == original
    assert main(['compile', str(tmp_path), '--frontend', 'native', '--input-shapes', '[[3,8]]', '--dry-run']) == 1


def test_gitignored_examples_and_output_do_not_influence_discovery(tmp_path):
    subprocess.run(['git', 'init', '-q', str(tmp_path)], check=True)
    write(tmp_path, '.gitignore', 'ignored.py\n')
    write(tmp_path, 'net.py', NET + 'example_input=torch.randn(2,8)\n')
    for name in ('ignored.py', 'output/net.py', '.venv/net.py'):
        write(tmp_path, name, NET + 'example_input=torch.randn(3,8)\n')
    assert load_config(tmp_path)['input_shapes'] == [[2, 8]]


def test_generated_config_roundtrips_optional_constructor_arguments(tmp_path):
    write(tmp_path, 'net.py', NET)
    config = load_config(tmp_path, input_shapes=[[2, 8]])
    config.update(factory_args=[None, {'flag': True}], factory_kwargs={'optional': None})
    text = config_text(config)
    assert tomllib.loads(text)['model']['mode'] == 'inference'
    write(tmp_path, 'openassembler.toml', text)
    assert load_config(tmp_path) == config


@pytest.mark.parametrize('backend', ['modal', 'user@a100', 'local'])
def test_all_transports_receive_the_same_inferred_contract(tmp_path, monkeypatch, backend):
    from openassembler.modelfront import frontend, runtime
    from openassembler.torchfront.archive import pack
    write(tmp_path, 'net.py', NET + 'example_input=torch.randn(2,8)\n')
    seen = []
    def request(body, config):
        seen.append((body, config))
        metadata = {k: body[k] for k in ('source_root', 'source_digest', 'config')}
        metadata.update(schema=1, backend='openassembler')
        files = {'plan.json': b'{}', 'native-ledger.json': b'{}',
                 'model.json': json.dumps(metadata).encode()}
        return {'report': {'passed': True}, 'artifacts': base64.b64encode(pack(files)).decode()}
    monkeypatch.setattr(frontend, '_request', request)
    monkeypatch.setattr(runtime, 'validate_bundle', lambda p: None)
    assert main(['compile', str(tmp_path), '--backend', backend, '--verify']) == 0
    assert seen[0][0]['verify'] and seen[0][1]['input_shapes'] == [[2, 8]]
    assert (tmp_path / 'output/inferred-openassembler.toml').is_file()
    assert json.loads((tmp_path / 'output/model-discovery.json').read_text())['ready']
    assert not (tmp_path / 'openassembler.toml').exists()


def test_worker_loads_nested_package_model_with_constructor_kwargs(tmp_path, monkeypatch):
    torch = pytest.importorskip('torch')
    from openassembler.modelfront.worker import _model
    monkeypatch.syspath_prepend(str(tmp_path / 'src'))
    write(tmp_path, 'src/unique_oa_package/__init__.py', '')
    write(tmp_path, 'src/unique_oa_package/constants.py', 'WIDTH = 8\n')
    write(tmp_path, 'src/unique_oa_package/net.py', 'from .constants import WIDTH\n' + NET)
    write(tmp_path, 'evaluate.py', 'from unique_oa_package.net import Net\nimport torch\nm=Net(width=12)\ny=m(torch.randn(2,12))\n')
    config = load_config(tmp_path)
    model = _model(tmp_path, config)
    assert model(torch.ones(2, 12)).shape == (2, 4)
    for name in list(sys.modules):
        if name.startswith('unique_oa_package'):
            monkeypatch.delitem(sys.modules, name)


def test_worker_loads_namespace_package_and_reports_capture_location(tmp_path, monkeypatch):
    torch = pytest.importorskip('torch')
    from openassembler.modelfront.worker import _model
    from openassembler.modelfront.graph import ModelGraphError, capture_model
    write(tmp_path, 'src/oa_namespace/constants.py', 'WIDTH = 8\n')
    write(tmp_path, 'src/oa_namespace/net.py', 'from .constants import WIDTH\n' + NET + 'example_input=torch.randn(2,8)\n')
    config = load_config(tmp_path)
    model = _model(tmp_path, config)
    assert model(torch.ones(2, 8)).shape == (2, 4)
    class ControlFlow(torch.nn.Module):
        def forward(self, x):
            if x.is_cuda:
                return x * 2
            return x
    with pytest.raises(ModelGraphError, match='test_model_discovery.py:.*forward'):
        capture_model(ControlFlow(), (torch.ones(2, 8),))
    for name in list(sys.modules):
        if name.startswith('oa_namespace'):
            monkeypatch.delitem(sys.modules, name)

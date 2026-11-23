"""User-source locations and narrow static bug rejection without user execution."""
import importlib.util
import io
from pathlib import Path
import subprocess
import sys
import pytest

from openassembler.pyfront import compile_file, NotSupported


@pytest.mark.parametrize('body', [
    'out[0] = s[0]',
    's[0] = 1.0\n    out[0] = s[1]',
])
def test_shared_read_without_possible_producer_names_file_and_line(tmp_path,body):
    source=tmp_path/'missing_writer.py'
    source.write_text('from openassembler.pyfront import *\n@kernel\ndef k(out: Array(f32)):\n'
                      '    s = shared_array(f32, 8)\n    '+body+'\n')
    with pytest.raises(NotSupported,match='no possible writer') as error:
        compile_file(source)
    assert str(source) in str(error.value)
    assert error.value.lineno in (5,6)


def test_shared_dynamic_producer_and_synchronized_read_remain_accepted(tmp_path):
    source=tmp_path/'valid.py'
    source.write_text('from openassembler.pyfront import *\n@kernel\ndef k(out: Array(f32)):\n'
                      '    s = shared_array(f32, 32)\n    s[tid_x()] = 1.0\n'
                      '    syncthreads()\n    out[tid_x()] = s[0]\n')
    assert compile_file(source)[1]==['k']


def test_color_detection_resolves_current_stream_and_tolerates_closed_stream(monkeypatch):
    from openassembler.ui.colors import supports_color
    monkeypatch.delenv('NO_COLOR',raising=False)
    monkeypatch.delenv('OPENASSEMBLER_FORCE_COLOR',raising=False)
    monkeypatch.setenv('TERM','xterm')
    closed=io.StringIO();closed.close()
    monkeypatch.setattr(sys,'stdout',closed)
    assert supports_color() is False
    assert supports_color(closed) is False


@pytest.mark.parametrize('expression,needle', [('torch.sin(x)','unsupported native model node'),
                                             ('torch.add(x, self.bias, alpha=2)','add alpha other than one is unsupported')])
def test_model_rejection_points_to_user_forward(tmp_path,expression,needle):
    torch=pytest.importorskip('torch')
    from openassembler.modelfront.graph import capture_model,ModelGraphError
    source=tmp_path/'user_network.py'
    source.write_text('import torch\nclass Net(torch.nn.Module):\n'
                      '    def __init__(self):\n        super().__init__()\n'
                      '        self.bias = torch.nn.Parameter(torch.zeros(3))\n'
                      '    def forward(self,x):\n        return '+expression+'\n')
    spec=importlib.util.spec_from_file_location('oa_user_diagnostic_test',source)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    with pytest.raises(ModelGraphError,match=needle) as error:
        capture_model(module.Net(),[torch.ones(2,3)])
    assert f'{source}:7 in forward' in str(error.value)


def test_supported_bias_broadcasting_is_captured():
    torch=pytest.importorskip('torch')
    from openassembler.modelfront.graph import capture_model,extract_weights
    from openassembler.modelfront.kernels import prepare
    from openassembler.modelfront.reference import CPUReference
    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bias=torch.nn.Parameter(torch.tensor([-1.,0.,2.]))
        def forward(self,x):return x+self.bias
    model=Net();x=torch.arange(6,dtype=torch.float32).reshape(2,3)
    plan=capture_model(model,[x]);prepare(plan)
    assert plan['steps'][0]['kernel']=='broadcast_add'
    actual=CPUReference(plan,extract_weights(model,plan)).forward({'x':x})
    torch.testing.assert_close(actual[plan['outputs'][0]],model(x))

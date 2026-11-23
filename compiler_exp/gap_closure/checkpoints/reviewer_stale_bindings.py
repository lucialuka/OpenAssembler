"""CPU-only public-binding counterexample; CUDA metadata is stubbed by format fixture."""
import json
from pathlib import Path
import sys
from types import MethodType

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import torch
from tests.test_model_checkpoints import runtime
from openassembler.modelfront.autograd import CompiledModel
from openassembler.modelfront.runtime import Runtime
from openassembler.modelfront import checkpoint


if __name__ == '__main__':
    import tempfile
    r = runtime()
    r.plan['inputs'] = []
    with tempfile.TemporaryDirectory(prefix='oa-checkpoint-binding-review-') as temporary:
        owner = CompiledModel(r.plan, r.weights, temporary)
        r.weights = owner.native_parameters()
        r.save_checkpoint = MethodType(Runtime.save_checkpoint, r)
        owner._native_runtime = r
        state = owner.state_dict()
        state['_oa_state_0'] = torch.full((3,), 99., dtype=torch.float32)
        owner.load_state_dict(state, assign=True)
        path = Path(temporary) / 'state.oa-checkpoint'
        owner.save_native_checkpoint(path, include_rng=False)
        saved = checkpoint.read_checkpoint(r, path, restore_rng=False)
        result = {'scope': 'CPU-only checkpoint binding review; pure format and public nn.Module state replacement, no GPU claim',
                  'public_module_weight': owner.native_parameters()['p'].tolist(),
                  'runtime_cached_weight': r.weights['p'].tolist(),
                  'saved_checkpoint_weight': saved['weights']['p'].tolist(),
                  'matches_current_module': torch.equal(saved['weights']['p'], owner.native_parameters()['p'])}
        output = Path(__file__).with_name('reviewer-stale-bindings.json')
        output.write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result, indent=2))

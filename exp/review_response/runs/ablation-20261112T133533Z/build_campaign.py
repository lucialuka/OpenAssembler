"""Paired fresh-output build ablation: native cubins with/without SASS listings."""
import hashlib
import json
from pathlib import Path
import runpy
import time
from unittest.mock import patch
import torch
from openassembler import build
from openassembler.modelfront.graph import capture_model, extract_weights
from openassembler.modelfront.optimization import optimize_plan
from openassembler.modelfront.runtime import build_bundle, validate_bundle


def main():
    torch.manual_seed(271)
    model = runpy.run_path('cases/native_tensor_model/model.py')['create_model']()
    plan = optimize_plan(capture_model(model, [torch.randn(128,128)*.25], math_mode='f16_rne_f32_accum'))
    weights = extract_weights(model, plan)
    records = []
    original_options = build.Options
    for round_index, order in enumerate([['listing', 'binary'], ['binary', 'listing'], ['listing', 'binary']]):
        paired = {}
        for mode in order:
            root = Path('results/build')/str(round_index)/mode
            before = time.perf_counter()
            if mode == 'binary':
                with patch.object(build, 'Options', lambda *a, **k: original_options(*a, **k, no_sass=True)), patch.object(build, 'disasm', lambda binary: ''):
                    build_bundle(plan, weights, root)
            else:
                build_bundle(plan, weights, root)
            elapsed = (time.perf_counter()-before)*1000
            validate_bundle(root)
            binaries = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in root.rglob('*.cubin') if 'reference' not in p.parts and p.name!='reference.cubin'}
            paired[mode] = {'ms': elapsed, 'native_cubin_sha256': binaries}
        assert paired['listing']['native_cubin_sha256'] == paired['binary']['native_cubin_sha256'], 'changed native cubin'
        records.append({'round': round_index, 'order': order, 'methods': paired,
                        'speedup': paired['listing']['ms']/paired['binary']['ms']})
    Path('results/build/result.json').write_text(json.dumps({'passed': True, 'rounds': records,
        'scope': 'fresh-output complete TensorCore model build; identical native modules, strict validate_bundle; diagnostics omitted; hardware execution proven separately on identical bytes'},indent=2))


if __name__ == '__main__':
    main()

"""Bounded graph composition preserves the original three-projection order."""
from copy import deepcopy

import pytest

from openassembler.modelfront.kernels import prepare, tensor_recipes
from openassembler.modelfront.tensor_projection import match
from openassembler.modelfront.tensor_schedule import workspace_plan


def plan():
    matrix = {'shape': [128, 128], 'dtype': 'f32'}
    parameters = []
    buffers = {'x': matrix}
    for name in ('q', 'k', 'v'):
        for suffix, shape in (('w', [128, 128]), ('b', [128])):
            parameter = {'name': name+suffix, 'shape': shape, 'dtype': 'f32'}
            parameters.append(parameter)
            buffers[name+suffix] = {'shape': shape, 'dtype': 'f32'}
    for name in ('q', 'k', 'v', 'sum', 'out'):
        buffers[name] = matrix
    def linear(name):
        return {'kernel': 'linear', 'inputs': ['x', name+'w', name+'b'],
                'output': name, 'attrs': {'m': 128, 'n': 128, 'k': 128}}
    return {'schema': 1, 'math': 'f16_rne_f32_accum', 'inputs': [{'name': 'x', **matrix}],
            'parameters': parameters, 'buffers': buffers, 'outputs': ['out'],
            'steps': [linear('q'), linear('k'), {'kernel': 'add', 'inputs': ['q', 'k'], 'output': 'sum', 'attrs': {}},
                      linear('v'), {'kernel': 'add', 'inputs': ['sum', 'v'], 'output': 'out', 'attrs': {}}]}


def test_normal_prepare_builds_one_composite_step_with_typed_workspace():
    model = plan()
    specs, steps, _ = prepare(model)
    assert len(steps) == 1 and steps[0]['kernel'] == 'projection_sum3'
    assert steps[0]['inputs'] == ['x', 'qw', 'qb', 'kw', 'kb', 'vw', 'vb']
    recipes = tensor_recipes(model, specs)
    assert list(recipes) == [steps[0]['entry']]
    recipe = recipes[steps[0]['entry']]
    assert len(recipe.stages) == 4
    assert recipe.stages[2].gemm_plan.n == 384
    schedule = workspace_plan(model, steps, recipes)
    assert sorted(spec['dtype'] for spec in schedule['allocations'].values()) == ['f16', 'f16', 'f32']
    source = recipe.stages[-1].ptx
    assert source.index('add.rn.f32 %f6,%f0,%f1') < source.index('add.rn.f32 %f6,%f6,%f2')


@pytest.mark.parametrize('change', ['math', 'dual', 'bias', 'shape', 'order', 'exposed', 'backward_use', 'attrs'])
def test_unmeasured_or_observed_intermediates_keep_original_graph(change):
    model = deepcopy(plan())
    if change == 'math':model['math'] = 'fp32'
    elif change == 'dual':model['tensor_packing'] = 'dual'
    elif change == 'bias':model['steps'][0]['inputs'].pop()
    elif change == 'shape':
        model['inputs'][0]['shape'] = [64, 128]
        model['buffers']['x'] = {'shape': [64, 128], 'dtype': 'f32'}
    elif change == 'order':model['steps'][2]['inputs'].reverse()
    elif change == 'exposed':model['outputs'].append('q')
    elif change == 'backward_use':model['backward'] = {'steps': [{'inputs': ['q']}]}
    elif change == 'attrs':model['steps'][0]['attrs']['m'] = 64
    assert match(model) is None

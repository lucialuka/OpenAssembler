"""Storage transformations preserve clone, gradient and parameter semantics."""
import copy

import pytest

from openassembler.modelfront.optimization import optimize_plan, validate_aliases, storage_layout, validate_storage


def plan():
    return {'schema': 1, 'inputs': [{'name': 'x', 'shape': [2, 3], 'dtype': 'f32'}],
            'parameters': [{'name': 'w', 'shape': [2, 3], 'dtype': 'f32'}],
            'buffers': {name: {'shape': dims, 'dtype': 'f32'} for name, dims in
                        [('x', [2, 3]), ('w', [2, 3]), ('view', [6]), ('clone', [6]),
                         ('wv', [6]), ('wv2', [2, 3]), ('seed', [6]), ('dx', [2, 3])]},
            'steps': [{'kernel': 'copy', 'inputs': ['x'], 'output': 'view', 'attrs': {}, 'view': True},
                      {'kernel': 'copy', 'inputs': ['view'], 'output': 'clone', 'attrs': {}},
                      {'kernel': 'copy', 'inputs': ['w'], 'output': 'wv', 'attrs': {}, 'view': True},
                      {'kernel': 'copy', 'inputs': ['wv'], 'output': 'wv2', 'attrs': {}, 'view': True}],
            'outputs': ['clone'],
            'backward': {'seed_grads': {'clone': 'seed'}, 'steps': [
                {'kernel': 'copy', 'inputs': ['seed'], 'output': 'dx', 'attrs': {}, 'view': True}],
                         'grad_inputs': {'x': 'dx'}, 'grad_parameters': {}}}


def test_view_selection_preserves_clone_and_updateable_parameters():
    original = plan()
    before = copy.deepcopy(original)
    optimized = optimize_plan(original)
    assert original == before
    assert set(optimized['aliases']) == {'view', 'wv2', 'dx'}
    assert optimized['aliases']['wv2']['source'] == 'wv'  # wv itself remains an independent copy
    assert optimized['aliases']['dx'] == {'source': 'seed', 'shape': [2, 3], 'phase': 'backward'}
    assert optimized['steps'] == original['steps']
    assert validate_aliases(optimized) == optimized['aliases']


def test_disabled_view_optimization_retains_materializations():
    assert optimize_plan(plan(), storage_views=False) == plan()


@pytest.mark.parametrize('mutation', ['shape', 'dtype', 'source', 'phase'])
def test_alias_mutations_are_rejected(mutation):
    p = optimize_plan(plan())
    if mutation in ('shape', 'source', 'phase'):
        p['aliases']['view'][mutation] = {'shape': [5], 'source': 'w', 'phase': 'backward'}[mutation]
    else:
        p['buffers']['view']['dtype'] = 'f16'
    with pytest.raises(ValueError, match='view|aliases'):
        validate_aliases(p)


def branched_plan():
    names = ['x', 'w', 'a', 'b', 'c', 'y', 'seed', 'ga', 'dx', 'dw']
    return {'schema': 1, 'inputs': [{'name': 'x', 'shape': [2, 3]}],
            'parameters': [{'name': 'w', 'shape': [2, 3]}],
            'buffers': {n: {'shape': [2, 3], 'dtype': 'f32'} for n in names},
            'steps': [{'kernel': 'relu', 'inputs': ['x'], 'output': 'a'},
                      {'kernel': 'relu', 'inputs': ['a'], 'output': 'b'},
                      {'kernel': 'add', 'inputs': ['a', 'b'], 'output': 'c'},
                      {'kernel': 'relu', 'inputs': ['c'], 'output': 'y'}],
            'outputs': ['y'], 'backward': {'seed_grads': {'y': 'seed'},
                'grad_inputs': {'x': 'dx'}, 'grad_parameters': {'w': 'dw'}, 'steps': [
                    {'kernel': 'relu_backward', 'inputs': ['seed', 'c'], 'output': 'ga'},
                    {'kernel': 'relu_backward', 'inputs': ['ga', 'x'], 'output': 'dx'},
                    {'kernel': 'copy', 'inputs': ['ga'], 'output': 'dw'}]}}


def test_complete_live_intervals_protect_branches_seeds_and_returned_results():
    p = branched_plan()
    layout = storage_layout(p)
    assert layout['physical_slots'] < layout['logical_buffers']
    assert layout['physical_bytes'] < layout['logical_bytes']
    assert not set(layout['assignments']) & {'x', 'w'}
    assert layout['lifetimes']['a'] == [0, 2]
    assert layout['lifetimes']['c'] == [2, 5]  # needed after forward ends
    assert layout['lifetimes']['seed'] == [4, 5]  # write before backward starts
    for name in ['y', 'dx', 'dw']:
        assert layout['lifetimes'][name][1] == 9
    for slot in layout['slots']:
        for left, right in zip(slot['members'], slot['members'][1:]):
            assert layout['lifetimes'][left][1] < layout['lifetimes'][right][0]
    assert layout['assignments']['c'] != layout['assignments']['a']
    assert layout['assignments']['ga'] != layout['assignments']['seed']
    assert layout == storage_layout(p)


def test_alias_groups_extend_the_real_root_until_every_exposed_view_expires():
    p = branched_plan()
    p['buffers']['av'] = {'shape': [6]}
    p['steps'].insert(1, {'kernel': 'copy', 'inputs': ['a'], 'output': 'av', 'view': True})
    p['outputs'].append('av')
    p = optimize_plan(p, reuse_storage=True)
    layout = p['storage']
    assert 'av' not in layout['assignments']
    assert layout['lifetimes']['a'] == [0, 10]
    assert validate_storage(p) == layout
    assert validate_aliases(p) == p['aliases']


@pytest.mark.parametrize('mutation', ['assignments', 'lifetimes', 'slots', 'physical_bytes', 'policy'])
def test_persisted_storage_schedule_is_rederived_and_rejects_mutations(mutation):
    p = optimize_plan(branched_plan(), reuse_storage=True)
    p['storage'][mutation] = {}
    with pytest.raises(ValueError, match='storage schedule'):
        validate_storage(p)


def test_storage_reuse_can_be_enabled_independently_from_views():
    p = branched_plan()
    result = optimize_plan(p, storage_views=False, reuse_storage=True)
    assert result['storage'] == storage_layout(p)
    assert 'aliases' not in result
    assert 'storage' not in p


def test_execution_preparation_rejects_an_injected_storage_schedule():
    from openassembler.modelfront.kernels import prepare
    p = optimize_plan(branched_plan(), reuse_storage=True)
    p['storage']['assignments']['c'] = p['storage']['assignments']['a']
    with pytest.raises(ValueError, match='storage schedule'):
        prepare(p)


def test_adding_native_loss_replans_the_complete_region_and_pins_both_outputs():
    from openassembler.modelfront.graph import append_training
    from openassembler.modelfront.kernels import prepare
    p = optimize_plan(branched_plan(), reuse_storage=True)
    result = append_training(p)
    assert result['storage'] != p['storage']
    assert validate_storage(result) == storage_layout(result)
    horizon = len(result['steps']) + len(result['backward']['steps']) + 2
    for name in result['outputs']:
        assert result['storage']['lifetimes'][name][1] == horizon
    prepare(result)

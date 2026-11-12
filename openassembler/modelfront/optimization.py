"""Immutable storage views and guarded lifetime planning above the machine backend.

Only capture-tagged, contiguous, immutable shape reinterpretations are admitted.
Explicit clones retain their allocation and updateable parameter storage never
backs an exposed alias. No arithmetic, precision or reduction order changes.
"""
from __future__ import annotations

import copy
import math


def optimize_plan(plan: dict, *, storage_views: bool = True, reuse_storage: bool = False) -> dict:
    result = copy.deepcopy(plan)
    if result.get('aliases') or result.get('storage'):
        raise ValueError('optimize a captured plan before introducing storage metadata')
    if not storage_views:
        if reuse_storage:
            result['storage'] = storage_layout(result)
        return result
    specs = result.get('buffers', {})
    parameters = {x['name'] for x in result.get('parameters', [])}
    aliases = {}
    for phase, steps in (('forward', result.get('steps', [])),
                         ('backward', result.get('backward', {}).get('steps', []))):
        for step in steps:
            if (step.get('kernel') != 'copy' or step.get('view') is not True
                    or len(step.get('inputs', [])) != 1 or step.get('attrs')):
                continue
            source, output = step['inputs'][0], step.get('output')
            if source not in specs or output not in specs or source == output:
                raise ValueError('storage view needs two defined distinct tensors')
            left, right = specs[source], specs[output]
            if left.get('dtype', 'f32') != right.get('dtype', 'f32'):
                raise ValueError('storage views cannot change tensor dtype')
            if math.prod(left['shape']) != math.prod(right['shape']):
                raise ValueError('storage views must preserve element count')
            root = source
            while root in aliases:
                root = aliases[root]['source']
            # Parameter updates occur after forward/backward. Retaining a copy
            # preserves the original value of views exposed in model outputs.
            if root in parameters:
                continue
            aliases[output] = {'source': source, 'shape': list(right['shape']), 'phase': phase}
    if aliases:
        result['aliases'] = aliases
    result['optimization'] = {'storage_views': storage_views, 'aliased_copies': len(aliases),
                              'numerical_transformations': False}
    if reuse_storage:
        result['storage'] = storage_layout(result)
    return result


def validate_aliases(plan: dict) -> dict:
    """Re-derive aliases from capture tags; reject injected or changed storage."""
    aliases = plan.get('aliases', {})
    if not isinstance(aliases, dict):
        raise ValueError('native plan storage aliases must be a mapping')
    captured = {key: value for key, value in plan.items() if key not in ('aliases', 'optimization', 'storage')}
    expected = optimize_plan(captured).get('aliases', {})
    if aliases and aliases != expected:
        raise ValueError('native plan storage aliases differ from the captured immutable views')
    return aliases


def storage_layout(plan: dict) -> dict:
    '''Assign exact-size guarded slots to tensors with disjoint live intervals.

    The timeline spans the entire forward/backward region. Output and returned
    gradient roots remain live through its end. All aliases extend the lifetime
    of their real storage root; caller inputs and parameters never enter a slot.
    Inputs and outputs of the same launch cannot share a slot. Execution must
    remain on the runtime's one bound CUDA stream. Tensor recipe packing and
    loss/update helpers keep their separate workspaces.
    '''
    aliases = validate_aliases(plan)
    specs = dict(plan.get('buffers', {}))
    external = {s['name'] for s in plan.get('inputs', []) + plan.get('parameters', [])}
    for spec in plan.get('inputs', []) + plan.get('parameters', []):
        specs[spec['name']] = spec

    def root(name):
        seen = set()
        while name in aliases:
            if name in seen:
                raise ValueError('native storage aliases contain a cycle')
            seen.add(name)
            name = aliases[name]['source']
        if name not in specs:
            raise ValueError('native storage refers to an undefined tensor')
        return name

    forward = plan.get('steps', [])
    backward = plan.get('backward', {})
    end = len(forward) + len(backward.get('steps', [])) + 2
    private = set(specs) - external - set(aliases)
    lifetimes = {}

    def touch(name, point):
        name = root(name)
        if name in private:
            interval = lifetimes.setdefault(name, [point, point])
            interval[0] = min(interval[0], point)
            interval[1] = max(interval[1], point)

    for point, step in enumerate(forward):
        for name in [*step.get('inputs', []), step['output']]:
            touch(name, point)
    # Seeds may be filled or bound immediately after forward, before the first
    # backward launch. Reserve them from this boundary rather than first read.
    for name in backward.get('seed_grads', {}).values():
        touch(name, len(forward))
    for point, step in enumerate(backward.get('steps', []), len(forward) + 1):
        for name in [*step.get('inputs', []), step['output']]:
            touch(name, point)
    exposed = [*plan.get('outputs', []), *backward.get('grad_inputs', {}).values(),
               *backward.get('grad_parameters', {}).values()]
    for name in exposed:
        touch(name, end)
    for name in private:
        # Keep unused declarations dedicated instead of inferring a write or
        # deleting them. prepare() separately validates actual dataflow.
        lifetimes.setdefault(name, [0, end])
    slots, assignments, occupied = [], {}, []
    for name in sorted(private, key=lambda n: (lifetimes[n][0], n)):
        spec = specs[name]
        dtype = spec.get('dtype', 'f32')
        count = math.prod(spec['shape'])
        if dtype != 'f32' or count <= 0:
            raise ValueError('native storage slots require nonempty FP32 tensors')
        begin, finish = lifetimes[name]
        eligible = [i for i, slot in enumerate(slots)
                    if slot['dtype'] == dtype and slot['elements'] == count and occupied[i] < begin]
        if eligible:
            index = eligible[0]
        else:
            index = len(slots)
            slots.append({'dtype': dtype, 'elements': count, 'members': []})
            occupied.append(-1)
        slots[index]['members'].append(name)
        assignments[name] = index
        occupied[index] = finish
    return {'schema': 1, 'policy': 'exact-size-liveness-v1', 'slots': slots,
            'assignments': assignments, 'lifetimes': lifetimes,
            'logical_buffers': len(private), 'physical_slots': len(slots),
            'logical_bytes': sum(math.prod(specs[n]['shape']) * 4 for n in private),
            'physical_bytes': sum(slot['elements'] * 4 for slot in slots)}


def validate_storage(plan: dict) -> dict | None:
    '''Re-derive the schedule rather than trusting persisted pointer aliases.'''
    if 'storage' not in plan:
        return None
    layout = plan['storage']
    if not isinstance(layout, dict) or layout != storage_layout(plan):
        raise ValueError('native storage schedule differs from its complete live intervals')
    return layout

"""Bounded three-projection composition using one stacked native product."""
from dataclasses import asdict, replace
from hashlib import sha256
from unittest.mock import patch

from openassembler.kernels.gemm import GemmPlan, generate
from openassembler.modelfront import kernels, kernels_tensor, runtime, tensor_schedule
from openassembler.modelfront.kernels import canonical
from openassembler.modelfront.kernels_tensor import TensorRecipe, TensorStage


prepare = kernels.prepare
recipes = kernels.tensor_recipes
workspace_plan = tensor_schedule.workspace_plan


def match(plan):
    steps = plan['steps']
    if plan.get('math') != kernels_tensor.MATH_MODE or plan.get('tensor_packing', 'separate') != 'separate' or len(steps) != 5:
        return None
    a, b, ab, c, abc = steps
    if ([s['kernel'] for s in steps] != ['linear', 'linear', 'add', 'linear', 'add']
            or any(len(s['inputs']) != 3 for s in (a, b, c))
            or len({s['inputs'][0] for s in (a, b, c)}) != 1
            or ab['inputs'] != [a['output'], b['output']]
            or abc['inputs'] != [ab['output'], c['output']]
            or plan['outputs'] != [abc['output']]):
        return None
    names = [a['inputs'][0], *a['inputs'][1:], *b['inputs'][1:], *c['inputs'][1:]]
    specs = kernels.tensor_specs(plan)
    if any(specs[name]['shape'] != ([128] if i in (2, 4, 6) else [128, 128])
           for i, name in enumerate(names)):
        return None
    hidden = {s['output'] for s in steps[:-1]}
    if any(hidden.intersection(s['inputs']) for s in plan.get('backward', {}).get('steps', [])):
        return None
    return names, abc['output']


def recipe():
    sig = {'op': 'projection_sum3', 'shape': [128, 128], 'math': kernels_tensor.MATH_MODE,
           'sum_order': '(q+bq)+(k+bk), then +(v+bv)'}
    name = 'oa_projection_' + sha256(canonical(sig)).hexdigest()[:20]
    pa, pb, products = [name + suffix for suffix in ('_a', '_b', '_products')]
    a = kernels_tensor._pack((128, 128), False, 'a0', pa)
    b = kernels_tensor._pack((128, 128), False, 'a1', pb)
    text = b.ptx.replace(b.name, name + '_stack_weights')
    text = text.replace('.param .u64 input,.param .u64 output',
                        '.param .u64 input,.param .u64 input1,.param .u64 input2,.param .u64 output')
    text = text.replace('ld.param.u64 %d0,[input];', '''ld.param.u64 %d0,[input];
ld.param.u64 %d6,[input1]; ld.param.u64 %d7,[input2]; mov.u32 %r30,%ctaid.z;
setp.eq.u32 %p2,%r30,1; selp.u64 %d0,%d6,%d0,%p2;
setp.eq.u32 %p2,%r30,2; selp.u64 %d0,%d7,%d0,%p2;''')
    text = text.replace('mul.wide.u32 %d4,%r23,4;', 'mad.lo.u32 %r23,%r30,8192,%r23; mul.wide.u32 %d4,%r23,4;')
    b = replace(b, name=name+'_stack_weights', ptx=text, inputs=('a1', 'a3', 'a5'),
                signature={**b.signature, 'stacked': 3}, launch={**b.launch, 'grid': [1, 128, 3]})
    plan = GemmPlan(128, 384, 128, 'f16', tile='64x64')
    entry = name + '_gemm'
    gemm = TensorStage(entry, 'projection_product', (pa, pb), products,
                       generate(plan).replace('.entry gemm(', f'.entry {entry}('), plan.launch,
                       {'op': 'projection_product', 'plan': asdict(plan)}, plan)
    entry = name + '_sum'
    text = f'''.version 8.7
.target sm_80
.address_size 64
.visible .entry {entry}(.param .u64 products,.param .u64 bq,.param .u64 bk,.param .u64 bv,.param .u64 output) {{
.reg .b32 %r<8>; .reg .b64 %d<14>; .reg .f32 %f<10>; .reg .pred %p;
ld.param.u64 %d0,[products]; ld.param.u64 %d1,[bq]; ld.param.u64 %d2,[bk]; ld.param.u64 %d3,[bv]; ld.param.u64 %d4,[output];
mov.u32 %r0,%tid.x; mov.u32 %r1,%ctaid.x; mad.lo.u32 %r2,%r1,128,%r0;
shr.u32 %r3,%r2,7; and.b32 %r4,%r2,127; mad.lo.u32 %r5,%r3,384,%r4;
mul.wide.u32 %d5,%r5,4; add.u64 %d6,%d0,%d5;
ld.global.f32 %f0,[%d6]; ld.global.f32 %f1,[%d6+512]; ld.global.f32 %f2,[%d6+1024];
mul.wide.u32 %d7,%r4,4; add.u64 %d8,%d1,%d7; add.u64 %d9,%d2,%d7; add.u64 %d10,%d3,%d7;
ld.global.f32 %f3,[%d8]; ld.global.f32 %f4,[%d9]; ld.global.f32 %f5,[%d10];
add.rn.f32 %f0,%f0,%f3; add.rn.f32 %f1,%f1,%f4; add.rn.f32 %f2,%f2,%f5;
add.rn.f32 %f6,%f0,%f1; add.rn.f32 %f6,%f6,%f2;
mul.wide.u32 %d11,%r2,4; add.u64 %d12,%d4,%d11; st.global.f32 [%d12],%f6; ret; }}'''
    summed = TensorStage(entry, 'projection_sum', (products, 'a2', 'a4', 'a6'), 'output', text,
                         {'grid': [128, 1, 1], 'threads': 128, 'smem': 0}, sig)
    return TensorRecipe(name, 'projection_sum3', sig,
                        {pa: {'shape': [128, 128], 'dtype': 'f16', 'alignment': 16},
                         pb: {'shape': [384, 128], 'dtype': 'f16', 'alignment': 16},
                         products: {'shape': [128, 384], 'dtype': 'f32', 'alignment': 16}},
                        (a, b, gemm, summed))


def combined_recipes(plan, specs=None):
    result = recipes(plan, specs)
    if match(plan):
        composed = recipe()
        result[composed.name] = composed
    return result


def combined_prepare(plan):
    specs, steps, programs = prepare(plan)
    matched = match(plan)
    if matched:
        composed = recipe()
        names, output = matched
        steps = [{'kernel': 'projection_sum3', 'entry': composed.name, 'inputs': names,
                  'output': output, 'phase': 'forward', 'tensor': True},
                 *[s for s in steps if s['phase'] != 'forward']]
    return specs, steps, programs


def composed_workspaces(plan, steps, recs):
    result = workspace_plan(plan, steps, recs)
    if result is None and match(plan):
        # The production binding planner already supports FP32 workspaces.
        # Retain that planner's allocation result for this experimental recipe.
        fake = {key: replace(rec, stages=tuple(replace(stage, op='round_f16_f32')
                                              if stage.output.endswith('_products') else stage for stage in rec.stages))
                for key, rec in recs.items()}
        result = workspace_plan(plan, steps, fake)
    return result


def enable(stack):
    from exp.ml_suite import engines
    for module in (kernels, runtime, engines):
        stack.enter_context(patch.object(module, 'prepare', combined_prepare))
        stack.enter_context(patch.object(module, 'tensor_recipes', combined_recipes))
    stack.enter_context(patch.object(tensor_schedule, 'workspace_plan', composed_workspaces))

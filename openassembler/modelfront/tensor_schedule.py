"""Pure allocation and binding plan for per-iteration packed operand reuse."""
from collections import Counter
import hashlib
from .kernels import canonical


def workspace_plan(plan,steps,recipes):
    def source_name(step, stage):
        source = step['inputs'][int(stage.inputs[0][1:])]
        while source in plan.get('aliases', {}):
            source = plan['aliases'][source]['source']
        return source

    def separate_key(step, stage):
        # Reuse the admitted row-major projection pack within this forward.
        if (step['phase'] == 'forward' and stage.op == 'pack_f16_rne'
                and stage.signature['shape'] == [128, 128] and not stage.signature['transpose']):
            return source_name(step, stage), stage.name
        return None

    uses = Counter(key for step in steps if step.get('tensor')
                   for stage in recipes[step['entry']].stages
                   if (key := separate_key(step, stage)) is not None)
    allocations={};bindings={};pack_slots={}
    for step in steps:
        if not step.get('tensor'):continue
        recipe=recipes[step['entry']];local={};packs={}
        for stage in recipe.stages:
            separate = separate_key(step, stage)
            if stage.op=='pack_dual_f16_rne':
                key='operand_'+hashlib.sha256(canonical([source_name(step, stage),stage.signature])).hexdigest()[:24]
            elif uses[separate] > 1:
                key='separate_operand_'+hashlib.sha256(canonical(separate)).hexdigest()[:24]
            else:
                key=None
            if key is not None:
                spec=recipe.workspaces[stage.output]
                if key in allocations and allocations[key]!=spec:raise ValueError('packed operand allocation collision')
                allocations[key]=spec
                local[stage.output]={'allocation':key,'shape':spec['shape'],'offset':0}
                packs[stage.output]=key
            elif stage.output!='output':
                spec=recipe.workspaces[stage.output];key=stage.output
                allocations[key]=spec;local[key]={'allocation':key,'shape':spec['shape'],'offset':0}
        for name,spec in recipe.workspaces.items():
            if 'base' in spec:
                base=local[spec['base']]
                local[name]={'allocation':base['allocation'],'shape':spec['shape'],'offset':spec['offset']}
        bindings[step['output']]=local;pack_slots[step['output']]=packs
    if not any(pack_slots.values()) and not any(
            stage.op == 'round_f16_f32' for recipe in recipes.values() for stage in recipe.stages):
        return None
    return {'allocations':allocations,'bindings':bindings,'pack_slots':pack_slots}

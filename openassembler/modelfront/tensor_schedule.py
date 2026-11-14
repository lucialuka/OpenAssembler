"""Pure allocation/binding plan for per-iteration dual-layout operand reuse."""
import hashlib
from .kernels import canonical


def workspace_plan(plan,steps,recipes):
    allocations={};bindings={};pack_slots={}
    for step in steps:
        if not step.get('tensor'):continue
        recipe=recipes[step['entry']];local={};packs={}
        for stage in recipe.stages:
            if stage.op=='pack_dual_f16_rne':
                source=step['inputs'][int(stage.inputs[0][1:])]
                while source in plan.get('aliases',{}):source=plan['aliases'][source]['source']
                key='operand_'+hashlib.sha256(canonical([source,stage.signature])).hexdigest()[:24]
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
    return {'allocations':allocations,'bindings':bindings,'pack_slots':pack_slots}

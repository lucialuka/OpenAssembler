"""Reuse a separate FP16 pack when the same logical value and layout recur."""
from collections import Counter
from dataclasses import replace

from openassembler.modelfront.tensor_schedule import workspace_plan


def bind_candidate(runtime):
    aliases = runtime.plan.get('aliases', {})

    def identity(step, stage):
        source = step['inputs'][int(stage.inputs[0][1:])]
        while source in aliases:
            source = aliases[source]['source']
        return step['phase'], source, stage.name

    uses = Counter(identity(step, stage)
                   for step in runtime.steps if step.get('tensor')
                   for stage in runtime.recipes[step['entry']].stages
                   if stage.op == 'pack_f16_rne')
    if not any(count > 1 for count in uses.values()):
        return

    # Each step can reuse a different subset of a recipe's two packs. Give the
    # runtime a step-local recipe while retaining its original CUDA entries.
    for step in runtime.steps:
        if not step.get('tensor'):
            continue
        recipe = runtime.recipes[step['entry']]
        stages = tuple(replace(stage, op='pack_dual_f16_rne',
                               signature={**stage.signature, 'phase': step['phase']})
                       if stage.op == 'pack_f16_rne' and uses[identity(step, stage)] > 1 else stage
                       for stage in recipe.stages)
        if stages != recipe.stages:
            key = ('research-reuse', step['output'])
            runtime.recipes[key] = replace(recipe, stages=stages)
            step['entry'] = key
    runtime._tensor_schedule = workspace_plan(runtime.plan, runtime.steps, runtime.recipes)

"""Bounded dual-layout recipe geometry, reuse identity and opt-in contracts."""
from dataclasses import replace
import pytest
from openassembler.modelfront.kernels_tensor import emit_tensor_recipe,MATH_MODE
from openassembler.modelfront.tensor_schedule import workspace_plan


def spec(*shape):return {'shape':list(shape),'dtype':'f32'}
def dual(op,left,right,out,bias=False):
    return emit_tensor_recipe(op,[spec(*left),spec(*right)]+([spec(out[-1])] if bias else []),
                              spec(*out),{'math_mode':MATH_MODE,'packing':'dual'})


@pytest.mark.parametrize('op,left,right,out',[
    ('linear',(17,65),(33,65),(17,33)),
    ('linear_dx',(17,33),(33,65),(17,65)),
    ('linear_dw',(17,33),(17,65),(33,65)),
    ('linear',(129,130),(127,130),(129,127)),
])
def test_padded_recipe_has_bounded_owned_views_and_guarded_crop(op,left,right,out):
    recipe=dual(op,left,right,out,op=='linear')
    gemm=next(stage for stage in recipe.stages if stage.gemm_plan)
    assert gemm.gemm_plan.m>=out[0] and gemm.gemm_plan.n>=out[1]
    assert recipe.stages[-1].op=='crop_bias'
    assert all(stage.name.startswith('oa_') for stage in recipe.stages)
    for entry in recipe.workspaces.values():
        if 'base' in entry:
            from math import prod
            parent=recipe.workspaces[entry['base']]
            assert entry['offset']+prod(entry['shape'])<=prod(parent['shape'])
    pack=recipe.stages[0]
    assert 'tile[4224]' in pack.ptx and 'bar.sync 0' in pack.ptx
    assert pack.launch['threads']==256


def test_schedule_reuses_same_logical_operand_across_phases_and_shape_aliases():
    forward=dual('linear',(128,128),(128,128),(128,128))
    dx=dual('linear_dx',(128,128),(128,128),(128,128))
    plan={'aliases':{'x_view':{'source':'x','shape':[128,128],'phase':'forward'}}}
    steps=[{'tensor':True,'entry':forward.name,'inputs':['x_view','w'],'output':'y','phase':'forward'},
           {'tensor':True,'entry':dx.name,'inputs':['dy','w'],'output':'dx','phase':'backward'}]
    schedule=workspace_plan(plan,steps,{forward.name:forward,dx.name:dx})
    assert len(schedule['allocations'])==3
    first=schedule['bindings']['y'];second=schedule['bindings']['dx']
    wa=first[forward.stages[1].output]['allocation']
    wb=second[dx.stages[1].output]['allocation']
    assert wa==wb
    assert schedule['pack_slots']['y'][forward.stages[0].output]!=wa
    assert any(v['offset']>0 for v in second.values())


@pytest.mark.parametrize('left,right,out',[
    ((17,65),(33,64),(17,33)),((17,33),(33,65),(17,65)),
    ((513,128),(128,128),(513,128))])
def test_dual_rejects_incompatible_or_unbounded_shapes(left,right,out):
    with pytest.raises(ValueError):dual('linear',left,right,out)


def test_repository_dual_packing_requires_explicit_math(tmp_path):
    from openassembler.modelfront.frontend import load_config,ModelError
    (tmp_path/'model.py').write_text('def create_model(): pass\n')
    config=tmp_path/'openassembler.toml'
    config.write_text('[model]\nentrypoint="model.py"\ninput_shapes=[[17,65]]\ntensor_packing="dual"\n')
    with pytest.raises(ModelError,match='f16_rne_f32_accum'):load_config(tmp_path)
    config.write_text(config.read_text()+'math="f16_rne_f32_accum"\n')
    assert load_config(tmp_path)['tensor_packing']=='dual'

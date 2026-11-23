"""Repeated profiling must retain exact counts, contracts and invocation boundaries."""
from copy import deepcopy
import pytest

from openassembler.modelfront.profiling import audit_trace, kernel_distributions, profile_iterations


def observations():
    launches=[];audits=[]
    resources={'oa':{'signature':{'shape':[7]},'driver':{'num_regs':16,'local_bytes':0},
                     'module_cubin_bytes':128,'kernel_text_bytes':64}}
    for iteration,values in [(1,[1.,3.]),(2,[5.,7.]),(3,[9.,11.])]:
        ledger=[{'entry':'oa','cubin_sha256':'a'*64,'text_sha256':'b'*64,
                 'launch':{'grid':[1,1,1],'threads':32,'smem':0}} for _ in values]
        audit=audit_trace([{'cat':'kernel','name':'oa','dur':value} for value in values],ledger)
        audits.append({**audit,'iteration':iteration})
        launches.extend({**row,'iteration':iteration} for row in ledger)
    return audits,launches,resources


def test_multiple_launches_are_distinct_from_multiple_invocations():
    audits,launches,resources=observations()
    before=deepcopy((audits,launches,resources))
    report=kernel_distributions(audits,launches,resources)
    row=report['kernels'][0]
    assert row['launch_count']==6 and row['profiled_invocations']==3
    assert row['repeated_invocations'] is True
    assert row['summary_us']==pytest.approx({'median':6,'min':1,'max':11,'p10':2,'p90':10,'iqr':5})
    assert row['samples'][2]=={'iteration':3,'duration_us':[9.,11.],'launches':2}
    assert (audits,launches,resources)==before
    assert kernel_distributions(audits[:1],launches,resources)['kernels'][0]['repeated_invocations'] is False


@pytest.mark.parametrize('invalid', ['failed','duplicate','counts','missing_duration','nan','boolean','resources','identity'])
def test_distribution_rejects_unaccounted_or_inconsistent_evidence(invalid):
    audits,launches,resources=observations()
    if invalid=='failed':audits[0]['passed']=False
    elif invalid=='duplicate':audits[1]['iteration']=1
    elif invalid=='counts':audits[0]['kernel_counts']['oa']=7
    elif invalid=='missing_duration':audits[0]['kernel_duration_us']['oa'].pop()
    elif invalid=='nan':audits[0]['kernel_duration_us']['oa'][0]=float('nan')
    elif invalid=='boolean':audits[0]['kernel_duration_us']['oa'][0]=True
    elif invalid=='resources':resources.clear()
    elif invalid=='identity':launches[-1]['cubin_sha256']='c'*64
    with pytest.raises(ValueError):kernel_distributions(audits,launches,resources)


@pytest.mark.parametrize('steps,repetitions',[(4,3),(3,1),(1,1)])
def test_profile_selection_uses_only_existing_iterations(steps,repetitions):
    assert profile_iterations(steps,repetitions)==list(range(steps-repetitions,steps))


@pytest.mark.parametrize('steps,repetitions',[(0,1),(3,0),(3,4),(3,True),(True,1),(3,1.0)])
def test_profile_selection_rejects_bad_counts(steps,repetitions):
    with pytest.raises(ValueError):profile_iterations(steps,repetitions)


@pytest.mark.parametrize('value,valid',[('3',True),('1',True),('0',False),('4',False),('true',False),('1.5',False)])
def test_configuration_validates_profile_iteration_count(tmp_path,value,valid):
    from openassembler.modelfront.frontend import ModelError,load_config
    (tmp_path/'model.py').write_text('def create_model(): pass\n')
    (tmp_path/'openassembler.toml').write_text('[model]\nentrypoint="model.py"\ninput_shapes=[[7]]\nsteps=3\nprofile_iterations='+value+'\n')
    if valid:assert load_config(tmp_path)['profile_iterations']==int(value)
    else:
        with pytest.raises(ModelError,match='profile_iterations'):load_config(tmp_path)

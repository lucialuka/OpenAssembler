"""Explicit launch specifications preserve 64-bit values and argument bit patterns."""
import struct
from pathlib import Path
import pytest
import openassembler.runner as runner


def plan_spec(tmp_path,monkeypatch,spec):
    target=runner.Target({'cubin':'native.cubin'},{'name':'wide','params':[['u64','x'],['s64','offset'],['u64','other']],'compiled_by':'openassembler'},tmp_path/'wide.py')
    monkeypatch.setattr(runner,'select',lambda *a,**kw:([target],[]))
    monkeypatch.setattr(runner,'_native_bytes',lambda *a:b'owned-native')
    plan=runner.Plan(reference_comparison=False)
    runner.spec_case(spec,{},tmp_path,0,plan,'wide')
    return plan


def test_signed_unsigned_buffers_and_arguments_keep_all_bits(tmp_path,monkeypatch):
    spec={'buffers':{'x':{'dtype':'i64','count':3,'init':[-2**63,-1,2**63-1]},'y':{'dtype':'u64','count':2,'init':[0,2**64-1]}},
          'steps':[{'kernel':'wide','block':1,'args':[{'buf':'x'},{'i64':-2**63},{'u64':2**64-1}]}]}
    plan=plan_spec(tmp_path,monkeypatch,spec);job=plan.jobs[0]
    assert job['buffers']=={'x':24,'y':16}
    assert job['init']['x']==struct.pack('<3q',-2**63,-1,2**63-1)
    assert job['init']['y']==struct.pack('<2Q',0,2**64-1)
    assert job['steps'][0][-1]==[('buf','x',0),('u64',2**63),('u64',2**64-1)]


@pytest.mark.parametrize('dtype',['i64','u64'])
def test_random_wide_initialization_has_declared_byte_count(tmp_path,monkeypatch,dtype):
    spec={'buffers':{'x':{'dtype':dtype,'count':5}},'steps':[{'kernel':'wide','args':[{'buf':'x'},{'i64':-1},{'u64':0}]}]}
    job=plan_spec(tmp_path,monkeypatch,spec).jobs[0]
    assert len(job['init']['x'])==40


@pytest.mark.parametrize('dtype,value',[('i64',2**63),('i64',-2**63-1),('u64',-1),('u64',2**64),('i64',True),('u64',1.5)])
def test_wide_buffer_initializers_validate_exact_range(tmp_path,monkeypatch,dtype,value):
    spec={'buffers':{'x':{'dtype':dtype,'count':1,'init':[value]}},'steps':[{'kernel':'wide'}]}
    with pytest.raises(runner.RunError,match='initialization values must be integers'):plan_spec(tmp_path,monkeypatch,spec)


@pytest.mark.parametrize('argument',[{'i64':2**63},{'i64':-2**63-1},{'u64':2**64},{'u64':-1},{'i64':False},{'i64':1.5}])
def test_wide_scalar_arguments_validate_exact_range(tmp_path,monkeypatch,argument):
    spec={'buffers':{'x':{'dtype':'i64','count':1}},'steps':[{'kernel':'wide','args':[{'buf':'x'},argument,{'u64':0}]}]}
    with pytest.raises(runner.RunError,match='invalid .*64 argument'):plan_spec(tmp_path,monkeypatch,spec)

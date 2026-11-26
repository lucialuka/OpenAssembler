"""Native predicate guard readiness across loop and alternate predecessor paths."""
import pytest
from openassembler.backend.compile import _branch_pred_latency,PRED_TO_GUARD
from openassembler.isa.word import Control,control_valid

def item(text,stall=1):return text,Control(stall,False,7,7,0,0)

@pytest.mark.parametrize('predicate',['P0','P3','P6'])
def test_backward_quoted_branch_uses_latest_loop_predicate(predicate):
 source=[item(f'ISETP.LT.AND {predicate}, PT, R0, 0x3, PT'),
         item(f'@!{predicate} BRA `(done)',5),item('IADD3 R0, R0, 0x1, RZ'),
         item(f'ISETP.LT.AND {predicate}, PT, R0, 0x2, PT'),item('BRA `(again)',5),item('EXIT',5)]
 linear=_branch_pred_latency(source);assert sum(c.stall for _,c in linear[3:5])<PRED_TO_GUARD
 fixed=_branch_pred_latency(source,{'again':1,'done':5})
 assert sum(c.stall for _,c in fixed[3:5])>=PRED_TO_GUARD
 assert[t for t,c in fixed]==[t for t,c in source]
 assert all(control_valid(c)and c.stall>=old.stall for(_,c),(_,old)in zip(fixed,linear))

@pytest.mark.parametrize('conditional',['@P1 ','@!P1 '])
def test_alternate_predecessors_both_satisfy_guard_readiness(conditional):
 source=[item('ISETP.LT.AND P0, PT, R0, 0x3, PT'),item(conditional+'BRA `(skip)',5),
         item('ISETP.LT.AND P0, PT, R0, 0x2, PT'),item('BRA `(join)',5),item('MOV R0, 0x0'),
         item('@P0 IADD3 R2, R2, 0x1, RZ'),item('EXIT',5)]
 fixed=_branch_pred_latency(source,{'skip':4,'join':5})
 for path in((0,1,4),(2,3)):
  assert sum(fixed[index][1].stall for index in path)>=PRED_TO_GUARD
 assert all(control_valid(c)for _,c in fixed)

@pytest.mark.parametrize('producer',['ISETP.LT.AND P0, PT, R0, 0x2, PT','PLOP3.LUT P0, PT, P1, PT, PT, 0xf0, 0x0'])
def test_unguarded_exit_or_fallthrough_does_not_form_false_backward_edge(producer):
 source=[item(producer,13),item('@P0 IADD3 R1, R1, 0x1, RZ'),item('EXIT',5)]
 fixed=_branch_pred_latency(source,{'entry':0});assert fixed==_branch_pred_latency(source)

def test_predicated_definition_keeps_earlier_reaching_definition_requirement():
 source=[item('ISETP.LT.AND P0, PT, R0, 0x3, PT'),item('@P1 ISETP.LT.AND P0, PT, R0, 0x2, PT'),
         item('BRA `(guard)',5),item('@P0 IADD3 R2, R2, 0x1, RZ'),item('EXIT',5)]
 fixed=_branch_pred_latency(source,{'guard':3})
 assert sum(fixed[index][1].stall for index in(1,2))>=PRED_TO_GUARD
 assert sum(fixed[index][1].stall for index in(0,1,2))>=PRED_TO_GUARD
 assert all(control_valid(c)for _,c in fixed)

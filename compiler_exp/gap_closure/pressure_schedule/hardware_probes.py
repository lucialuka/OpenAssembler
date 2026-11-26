"""New scheduling workloads plus all prior scalar/group/predicate spill kernels."""
from probes import HEADER,cases as new_cases,expected as new_expected
import legacy_scalar_probes as scalar
import legacy_group_probes as group
import legacy_tight_probes as tight


def cases():
    for case in new_cases():yield dict(case,domain='new')
    for case in scalar.cases():yield dict(case,domain='prior_scalar',output_words=1)
    for case in tight.cases():yield dict(case,domain='prior_scalar',output_words=1)
    for case in group.cases():yield dict(case,domain='prior_group')


def expected(values,case,n,lanes):
    if case['domain']=='new':return new_expected(values,case,n,lanes)
    if case['domain']=='prior_group':return group.expected(values,case,n,lanes)
    result=scalar.expected(values,case['count'],n,lanes)
    if case.get('local'):
        for lane in range(n):result[lane]^=values[lane*case['count']]
    return result

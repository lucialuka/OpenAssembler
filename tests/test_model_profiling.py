"""Compute names cannot masquerade as storage activity in a native audit."""
import pytest
from openassembler.modelfront.profiling import audit_trace


@pytest.mark.parametrize('name', ['vendor_memcpy_compute', 'vendor_memset_compute', 'unknown'])
def test_profiler_trace_rejects_extra_compute_regardless_of_its_name(name):
    with pytest.raises(ValueError, match='audit failed'):
        audit_trace([{'cat':'kernel','name':'oa_owned'}, {'cat':'kernel','name':name}],
                    [{'entry':'oa_owned'}])


def test_profiler_trace_requires_exact_compute_counts_and_classifies_storage():
    trace = [{'cat':'kernel','name':'oa_memcpy_kernel'}]*2 + [
        {'cat':'gpu_memcpy','name':'Memcpy HtoD'}, {'cat':'gpu_memset','name':'Memset'},
        {'cat':'gpu_user_annotation','name':'training'}]
    ledger = [{'entry':'oa_memcpy_kernel'}]*2
    result = audit_trace({'traceEvents':trace},ledger)
    assert result['observed_compute_launches'] == 2
    assert result['storage_activities'] == {'gpu_memcpy':1,'gpu_memset':1}
    for bad in [trace[1:],trace[2:],[],trace+trace[:1]]:
        with pytest.raises(ValueError):
            audit_trace(bad,ledger)

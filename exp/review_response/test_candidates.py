from types import SimpleNamespace
import pytest

from exp.review_response.audit import audit_trace
from openassembler.modelfront.runtime import audit_profiler_events


@pytest.mark.parametrize('word', ['memcpy', 'memset'])
def test_existing_audit_has_the_reviewed_loophole(word):
    import ast
    from pathlib import Path
    source = (Path(__file__).parent/'input/baseline-runtime.py').read_text()
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'audit_profiler_events')
    baseline = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), 'baseline-audit', 'exec'), baseline)
    events = [SimpleNamespace(name=n, device_type='DeviceType.CUDA')
              for n in ['oa_owned', 'unexpected_' + word + '_compute']]
    # This explicitly records the baseline defect before production changes.
    assert baseline['audit_profiler_events'](events, [{'entry': 'oa_owned'}])['passed']
    with pytest.raises(ValueError):
        audit_profiler_events(events, [{'entry': 'oa_owned'}])


@pytest.mark.parametrize('word', ['memcpy', 'memset'])
def test_activity_based_candidate_rejects_disguised_compute(word):
    trace = [{'cat': 'kernel', 'name': 'oa_owned'},
             {'cat': 'kernel', 'name': 'unexpected_' + word + '_compute'}]
    with pytest.raises(ValueError, match='audit failed'):
        audit_trace(trace, [{'entry': 'oa_owned'}])


def test_activity_kinds_allow_real_storage_without_filtering_kernel_names():
    trace = [{'cat': 'kernel', 'name': 'oa_memcpy_semantics'},
             {'cat': 'gpu_memcpy', 'name': 'Memcpy HtoD'},
             {'cat': 'gpu_memset', 'name': 'Memset'},
             {'cat': 'gpu_user_annotation', 'name': 'region'}]
    result = audit_trace(trace, [{'entry': 'oa_memcpy_semantics'}])
    assert result['observed_compute_launches'] == 1
    assert result['storage_activities'] == {'gpu_memcpy': 1, 'gpu_memset': 1}
    with pytest.raises(ValueError):
        audit_trace(trace[1:], [{'entry': 'oa_memcpy_semantics'}])

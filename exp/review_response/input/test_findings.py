"""Desired invariants: failures are evidence, not patched production behavior."""
from types import SimpleNamespace
import pytest,torch
from openassembler.modelfront.runtime import audit_profiler_events

@pytest.mark.parametrize('name',['vendor_memcpy_named_compute','vendor_memset_named_compute'])
def test_compute_kernel_cannot_evade_audit_by_memory_substring(name):
    events=[SimpleNamespace(name=n,device_type='DeviceType.CUDA') for n in ['oa_known',name]]
    with pytest.raises(ValueError):audit_profiler_events(events,[{'entry':'oa_known'}])

from types import SimpleNamespace
import pytest
from openassembler import oracle
from openassembler.modelfront import model_cache
from exp.model_perf.cache_candidate import toolchain_identity


def test_binary_only_identity_works_without_disassembler_and_listing_keeps_dependency(monkeypatch):
    calls=[]
    def tool(name,*args):
        calls.append(name)
        if name=='nvdisasm':raise FileNotFoundError('no nvdisasm')
        return SimpleNamespace(stdout='ptxas pinned test')
    monkeypatch.setattr(oracle,'run_tool',tool)
    toolchain_identity.cache_clear();model_cache.toolchain_identity.cache_clear()
    with pytest.raises(FileNotFoundError):model_cache.toolchain_identity()
    calls.clear()
    assert toolchain_identity(False)=={'ptxas':'ptxas pinned test'}
    assert calls==['ptxas']
    with pytest.raises(FileNotFoundError):toolchain_identity(True)
    toolchain_identity.cache_clear();model_cache.toolchain_identity.cache_clear()

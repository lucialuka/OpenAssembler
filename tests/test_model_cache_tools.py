"""Optional disassembly must not become a binary-only cache dependency."""

from types import SimpleNamespace
import pytest
from openassembler import oracle
from openassembler.modelfront import model_cache


def test_binary_only_key_never_queries_missing_disassembler(monkeypatch):
    calls = []

    def tool(name, *args):
        calls.append(name)
        if name == "nvdisasm":
            raise FileNotFoundError("nvdisasm absent")
        return SimpleNamespace(stdout="ptxas pinned test")

    monkeypatch.setattr(oracle, "run_tool", tool)
    model_cache.toolchain_identity.cache_clear()
    try:
        assert model_cache.toolchain_identity(False) == {"ptxas": "ptxas pinned test"}
        assert calls == ["ptxas"]
        with pytest.raises(FileNotFoundError):
            model_cache.toolchain_identity(True)
        assert model_cache.toolchain_identity(False) == {"ptxas": "ptxas pinned test"}
    finally:
        model_cache.toolchain_identity.cache_clear()

"""Binary-only cache identities need only the assembler, not a disassembler."""
from functools import lru_cache


@lru_cache(maxsize=2)
def toolchain_identity(sass_listings=True):
    from openassembler import oracle
    names=('ptxas','nvdisasm') if sass_listings else ('ptxas',)
    return {name:oracle.run_tool(name,'--version').stdout for name in names}

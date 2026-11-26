from openassembler.pyfront import *
raise RuntimeError('Boolean storage helpers must not execute')
@device
def choose(value: bool, flag: bool) -> bool:
    if flag:
        return not value
    return value

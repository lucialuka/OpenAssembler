"""Byte and executable-function custody for standalone OA native modules.

This detects stale/replaced artifacts, not a cryptographic signature of origin:
the build records hashes only after comparing every executable word to OA output.
"""
from __future__ import annotations
import hashlib
from openassembler.container.elf import Elf,EM_CUDA


def inventory(binary):
    elf=Elf.parse(binary)
    if elf.machine!=EM_CUDA or elf.flags&255!=80:
        raise ValueError("native execution requires an SM80 CUDA cubin")
    sections={s.name[6:]:s for s in elf.sections if s.name.startswith('.text.') and s.data}
    if not sections:
        raise ValueError("native module has no executable kernel text")
    return {name:{'text_sha256':hashlib.sha256(section.data).hexdigest(),
                  'registers':section.info>>24,'text_bytes':len(section.data)}
            for name,section in sections.items()}


def record(binary,kernels,*,ptx_sha256):
    """Called only after the builder proves bytes equal its encoded native words."""
    texts=inventory(binary)
    if set(texts)!={k['name'] for k in kernels} or any(k.get('compiled_by')!='openassembler' for k in kernels):
        raise ValueError("native ownership requires complete OA executable text coverage")
    for kernel in kernels:
        texts[kernel['name']]['params']=kernel['params']
    return {'schema':1,'arch':'sm_80','cubin_sha256':hashlib.sha256(binary).hexdigest(),
            'ptx_sha256':ptx_sha256,'kernels':texts}


def validate(entry,binary):
    proof=entry.get('native_ownership')
    if not isinstance(proof,dict) or proof.get('schema')!=1 or proof.get('arch')!='sm_80':
        raise ValueError("native module has no current ownership ledger; recompile with OA")
    kernels=entry.get('kernels',[])
    if not kernels or any(k.get('compiled_by')!='openassembler' for k in kernels):
        raise ValueError("native ownership rejects vendor/fallback module entries")
    if hashlib.sha256(binary).hexdigest()!=proof.get('cubin_sha256'):
        raise ValueError("native cubin integrity differs from its OA build; recompile")
    texts=inventory(binary)
    for kernel in kernels:
        if kernel['name'] not in texts:
            raise ValueError("native module executable inventory differs from manifest")
        texts[kernel['name']]['params']=kernel['params']
    if set(texts)!={k['name'] for k in kernels} or texts!=proof.get('kernels'):
        raise ValueError("native module text/register/ABI inventory differs from its OA build")
    return binary

"""Byte and executable-function custody for standalone OA native modules.

This detects stale/replaced artifacts, not a cryptographic signature of origin:
the build records hashes only after comparing every executable word to OA output.
"""
from __future__ import annotations
import hashlib
from copy import deepcopy
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
    from openassembler.backend.storage import local_frame_inventory
    frames=local_frame_inventory(binary) if any(k.get('local_bytes',0) for k in kernels) else {}
    for kernel in kernels:
        if kernel.get('local_bytes',0):
            if frames.get(kernel['name'])!=kernel['local_bytes']:
                raise ValueError("native local frame metadata differs from generated contract")
            texts[kernel['name']]['local_bytes']=kernel['local_bytes']
        texts[kernel['name']]['params']=deepcopy(kernel['params'])
        if kernel.get('logical_parameter_contract') is not None:
            from openassembler.verify import _parameter_contract
            from openassembler.ptx.parse import Kernel
            _parameter_contract(Kernel(kernel['name'],kernel['params'],[],{}),kernel['logical_parameter_contract'])
            texts[kernel['name']]['logical_parameter_contract']=deepcopy(kernel['logical_parameter_contract'])
        if kernel.get('ptx_directives'):
            texts[kernel['name']]['ptx_directives']=deepcopy(kernel['ptx_directives'])
        if kernel.get('buffer_contract') is not None:
            texts[kernel['name']]['buffer_contract']=deepcopy(kernel['buffer_contract'])
        if kernel.get('parameter_attributes'):
            texts[kernel['name']]['parameter_attributes']=deepcopy(kernel['parameter_attributes'])
        if kernel.get('inlined_device_calls') is not None:
            texts[kernel['name']]['inlined_device_calls']=deepcopy(kernel['inlined_device_calls'])
        if kernel.get('device_link') is not None:
            texts[kernel['name']]['device_link']=deepcopy(kernel['device_link'])
        if kernel.get('python_source_graph') is not None:
            texts[kernel['name']]['python_source_graph']=deepcopy(kernel['python_source_graph'])
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
    from openassembler.backend.storage import local_frame_inventory
    frames=local_frame_inventory(binary) if any(k.get('local_bytes',0) for k in kernels) else {}
    for kernel in kernels:
        if kernel.get('local_bytes',0):
            if frames.get(kernel['name'])!=kernel['local_bytes']:
                raise ValueError("native local frame metadata differs from manifest")
            texts[kernel['name']]['local_bytes']=kernel['local_bytes']
        if kernel['name'] not in texts:
            raise ValueError("native module executable inventory differs from manifest")
        texts[kernel['name']]['params']=kernel['params']
        if kernel.get('logical_parameter_contract') is not None:
            from openassembler.verify import _parameter_contract
            from openassembler.ptx.parse import Kernel
            _parameter_contract(Kernel(kernel['name'],kernel['params'],[],{}),kernel['logical_parameter_contract'])
            texts[kernel['name']]['logical_parameter_contract']=kernel['logical_parameter_contract']
        if kernel.get('ptx_directives'):
            texts[kernel['name']]['ptx_directives']=kernel['ptx_directives']
        if kernel.get('buffer_contract') is not None:
            texts[kernel['name']]['buffer_contract']=kernel['buffer_contract']
        if kernel.get('parameter_attributes'):
            texts[kernel['name']]['parameter_attributes']=kernel['parameter_attributes']
        if kernel.get('inlined_device_calls') is not None:
            texts[kernel['name']]['inlined_device_calls']=kernel['inlined_device_calls']
        if kernel.get('device_link') is not None:
            texts[kernel['name']]['device_link']=kernel['device_link']
        if kernel.get('python_source_graph') is not None:
            texts[kernel['name']]['python_source_graph']=kernel['python_source_graph']
    if set(texts)!={k['name'] for k in kernels} or texts!=proof.get('kernels'):
        raise ValueError("native module text/register/ABI inventory differs from its OA build")
    return binary

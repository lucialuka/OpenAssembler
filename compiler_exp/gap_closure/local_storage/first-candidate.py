"""Isolated native local storage candidate; generated from frozen live sources."""
from pathlib import Path
import builtins
import hashlib
import json
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]


def once(text,old,new):
    assert text.count(old)==1,old
    return text.replace(old,new,1)


def generate():
    ra=(HERE/'baseline-ra.py').read_text()
    ra=once(ra,'        self.smem_bytes = soff','''        self.smem_bytes = soff
        from openassembler.backend.storage import local_frame
        try:self.local_frame=local_frame(k)
        except ValueError as error:raise NotSupported(str(error)) from None''')
    ra=once(ra,'        self.r1 = r1','''        self.r1 = r1
        if self.local_frame.bytes:
            self.emit(f"IADD3 {{d0}}, {{s0}}, -{self.local_frame.bytes:#x}, RZ",[r1],[r1],side=True)''')
    ra=once(ra,'        if op == "cvta":', '''        if op in ("ld","st") and "local" in m:
            return self.local(ins,t)
        if op=="mov" and isinstance(a[1],Var) and a[1].name in self.local_frame.offsets:
            if t not in("u32","b32"):
                raise NotSupported("local symbols require explicit 32-bit local addresses")
            destination=self.new()
            self.emit(f"IADD3 {{d0}}, {{s0}}, {self.local_frame.offsets[a[1].name]:#x}, RZ",[destination],[self.r1])
            self.bind(a[0].name,destination)
            return
        if op == "cvta":''')
    ra=once(ra,'    def shared(self, ins, t):','''    def local(self,ins,t):
        address=ins.args[1] if ins.op=="ld" else ins.args[0]
        if not self.local_frame.bytes:
            raise NotSupported("local access requires a declared native frame")
        if t not in("b32","u32","s32","f32") or address.off%4:
            raise NotSupported("local accesses require aligned scalar 32-bit elements")
        if address.base in self.local_frame.offsets:
            if address.off<0 or address.off+4>self.local_frame.sizes[address.base]:
                raise NotSupported("literal local symbol access is outside its declaration")
            base=self.r1;offset=self.local_frame.offsets[address.base]+address.off
        else:
            base=self.reg32(Var(address.base));offset=address.off
        if offset%4:
            raise NotSupported("local symbol layout is not aligned for a 32-bit access")
        if abs(offset)>65535:
            raise NotSupported("local literal address offset exceeds the admitted range")
        suffix=("+" if offset>0 else "-")+hex(abs(offset)) if offset else ""
        if ins.op=="ld":
            result=self.new()
            self.emit(f"LDL {{d0}}, [{{s0}}{suffix}]",[result],[base],var_lat=True)
            self.bind(ins.args[0].name,result)
        else:
            value=self.reg32(ins.args[1])
            self.emit(f"STL [{{s0}}{suffix}], {{s1}}",[],[base,value],side=True)

    def shared(self, ins, t):''')
    contract=(HERE/'baseline-contracts.py').read_text()
    contract=once(contract,"    for op in ('ret','bra'):","""    for dtype in ('b32','u32','s32','f32'):
        for op in ('ld','st'):add(op,'local.'+dtype,2)
    for op in ('ret','bra'):""")
    build=(HERE/'baseline-build.py').read_text()
    build=once(build,'def splice_kernels(base: bytes, ours: dict[str, tuple[list[int], int, list[int]]]) -> bytes:',
               'def splice_kernels(base: bytes, ours: dict[str, tuple[list[int], int, list[int]]], *, local_frames=None) -> bytes:')
    build=once(build,'    e.replace_data(".nv.info", nvinfo.serialize(recs))\n    return e.serialize()',
               '''    e.replace_data(".nv.info", nvinfo.serialize(recs))
    binary=e.serialize()
    if local_frames:
        if not set(local_frames)<=set(ours):raise ValueError("local frame reservation requires replaced native code")
        from openassembler.backend.storage import set_local_frames
        binary=set_local_frames(binary,local_frames)
    return binary''')
    build=once(build,'    buffer_contract: dict | None = None','    buffer_contract: dict | None = None\n    local_bytes: int = 0')
    build=once(build,'        if self.source_params:', '        if self.local_bytes:\n            r["local_bytes"]=self.local_bytes\n        if self.source_params:')
    build=once(build,'            k.instructions, k.registers = count_insns(code), nregs','''            k.instructions, k.registers = count_insns(code), nregs
            from openassembler.backend.storage import local_frame
            k.local_bytes=local_frame(module.kernels[kname]).bytes''')
    build=once(build,'        cubin = splice_kernels(ref, ours)',
               '        cubin = splice_kernels(ref, ours, local_frames={k.name:k.local_bytes for k in so.kept if k.compiled_by=="openassembler"})')
    ownership=(HERE/'baseline-native_ownership.py').read_text()
    ownership=once(ownership,"    for kernel in kernels:\n        texts[kernel['name']]['params']=deepcopy(kernel['params'])",'''    from openassembler.backend.storage import local_frame_inventory
    frames=local_frame_inventory(binary) if any(k.get('local_bytes',0) for k in kernels) else {}
    for kernel in kernels:
        if kernel.get('local_bytes',0):
            if frames.get(kernel['name'])!=kernel['local_bytes']:
                raise ValueError("native local frame metadata differs from generated contract")
            texts[kernel['name']]['local_bytes']=kernel['local_bytes']
        texts[kernel['name']]['params']=deepcopy(kernel['params'])''')
    ownership=once(ownership,"    for kernel in kernels:\n        if kernel['name'] not in texts:",'''    from openassembler.backend.storage import local_frame_inventory
    frames=local_frame_inventory(binary) if any(k.get('local_bytes',0) for k in kernels) else {}
    for kernel in kernels:
        if kernel.get('local_bytes',0):
            if frames.get(kernel['name'])!=kernel['local_bytes']:
                raise ValueError("native local frame metadata differs from manifest")
            texts[kernel['name']]['local_bytes']=kernel['local_bytes']
        if kernel['name'] not in texts:''')
    return {'openassembler/backend/ra.py':ra,'openassembler/backend/contracts.py':contract,
            'openassembler/build.py':build,'openassembler/native_ownership.py':ownership,
            'openassembler/backend/storage.py':(HERE/'storage.py').read_text()}


def install():
    import importlib.util
    name='openassembler.backend.storage';path=HERE/'storage.py'
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    import openassembler.backend.contracts as contracts
    import openassembler.backend.ra as ra
    import openassembler.build as build
    import openassembler.native_ownership as ownership
    for module in(contracts,ra,build,ownership):
        path=HERE/('candidate-'+Path(module.__file__).name)
        exec(builtins.compile(path.read_text(),str(path),'exec'),module.__dict__)


if __name__=='__main__':
    generated=generate()
    for name,source in generated.items():
        (HERE/('candidate-'+Path(name).name)).write_text(source)
    (HERE/'candidate-identities.json').write_text(json.dumps({name:hashlib.sha256(source.encode()).hexdigest()for name,source in generated.items()},indent=2)+'\n')

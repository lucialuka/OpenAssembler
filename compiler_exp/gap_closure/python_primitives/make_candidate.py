from pathlib import Path
HERE=Path(__file__).resolve().parent

def build():
 files={p.name:p.read_text()for p in(HERE/'before').glob('*.py')};files['primitives.py']=(HERE/'primitives.py').read_text()
 names='"atomic_add", "atomic_exchange", "atomic_compare_exchange"'
 s=files['compile.py'].replace('"device", "Array", "Buffer",','"device", "Array", "Buffer", '+names+',')
 s=s.replace('    def _call(self, node: ast.Call) -> tuple[_Reg, str]:','''    def _call(self, node: ast.Call) -> tuple[_Reg, str]:''')
 anchor='        if fname.startswith("pf.") and short in _SREGS:'
 s=s.replace(anchor,'''        if fname.startswith("pf.") and short in {"atomic_add", "atomic_exchange", "atomic_compare_exchange"}:
            from openassembler.pyfront.primitives import atomic
            return atomic(self,node,short)
'''+anchor)
 anchor='            if isinstance(n, ast.Call) and self.names.resolve(n.func) in {f"pf.{t}" for t in _THREAD_VARYING}:'
 s=s.replace(anchor,'            if isinstance(n, ast.Call) and self.names.resolve(n.func) in {f"pf.{t}" for t in _THREAD_VARYING | {"atomic_add", "atomic_exchange", "atomic_compare_exchange"}}:')
 # Effectful scalar returns may be discarded; still emit the RMW.
 anchor='        raise NotSupported("expression statement with no supported effect", node)'
 s=s.replace(anchor,'''        if isinstance(v,ast.Call) and self.names.resolve(v.func) in {"pf.atomic_add", "pf.atomic_exchange", "pf.atomic_compare_exchange"}:
            self._call(v)
            return
'''+anchor)
 files['compile.py']=s
 s=files['buffers.py'].replace('def access(emitter,sub,value=None,kind=None):','def access(emitter,sub,value=None,kind=None,atomic=None,index_values=None):')
 s=s.replace("    if value is not None and view.access=='read':", "    if (value is not None or atomic is not None) and view.access=='read':")
 s=s.replace('        loaded=emitter._load(array,f\'[{address.name}]\');emitter._mov(result,loaded,array.elem)', "        loaded=atomic(f'[{address.name}]')if atomic is not None else emitter._load(array,f'[{address.name}]');emitter._mov(result,loaded,array.elem)")
 s=s.replace('    for node in nodes:\n        register,index_kind=emitter._eval(node)', '    for position,node in enumerate(nodes):\n        register,index_kind=(index_values[position],"s32")if index_values is not None else emitter._eval(node)')
 files['buffers.py']=s
 s=files['intrinsics.py'];s+='\natomic_add = _stub("atomic_add", "GPU relaxed 32-bit integer atomic add; returns old value.")\natomic_exchange = _stub("atomic_exchange", "GPU relaxed u32 atomic exchange; returns old bits.")\natomic_compare_exchange = _stub("atomic_compare_exchange", "GPU relaxed u32 atomic CAS; returns old bits.")\n';files['intrinsics.py']=s
 s=files['__init__.py'];s+='\nfrom openassembler.pyfront.intrinsics import atomic_add, atomic_exchange, atomic_compare_exchange\n__all__ += ["atomic_add", "atomic_exchange", "atomic_compare_exchange"]\n';files['__init__.py']=s
 for name,text in files.items():(HERE/'candidate'/name).write_text(text)

if __name__=='__main__':build()

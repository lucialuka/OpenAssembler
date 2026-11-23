"""Generate isolated checked buffer candidate from pre-buffer source snapshots."""
from pathlib import Path
HERE=Path(__file__).resolve().parent

def build():
 files={p.name:p.read_text() for p in (HERE/'before').glob('*.py')}
 files['buffers.py']=(HERE/'checked.py').read_text()
 s=files['__init__.py'];s+='\nfrom openassembler.pyfront.buffers import Buffer, BufferType\n__all__ += ["Buffer", "BufferType"]\n';files['__init__.py']=s
 s=files['compile.py'].replace('"device", "Array",','"device", "Array", "Buffer",')
 anchor='    if isinstance(ann, ast.Call) and r == "pf.Array":'
 s=s.replace(anchor,'    if isinstance(ann, ast.Call) and r == "pf.Buffer":\n        from openassembler.pyfront.buffers import resolve_annotation\n        return resolve_annotation(ann,names)\n'+anchor)
 s=s.replace('    length: int | None = None','    length: int | None = None\n    view: object = None')
 anchor='        if isinstance(node.target, ast.Subscript):\n            arr, addr = self._address(node.target)'
 s=s.replace(anchor,'''        if isinstance(node.target, ast.Subscript):
            if isinstance(node.target.value,ast.Name) and getattr(self.s.arrays.get(node.target.value.id),"view",None) is not None:
                raise NotSupported("Buffer augmented stores require an explicit checked load and assignment",node)
            arr, addr = self._address(node.target)''')
 s=s.replace('        if isinstance(node, ast.Subscript):\n            arr, addr = self._address(node)', '''        if isinstance(node, ast.Subscript):
            if isinstance(node.value,ast.Name) and getattr(self.s.arrays.get(node.value.id),"view",None) is not None:
                from openassembler.pyfront.buffers import access
                return access(self,node)
            arr, addr = self._address(node)''')
 s=s.replace('    def _store(self, target: ast.Subscript, val: _Reg, kind: str) -> None:\n        arr, addr = self._address(target)', '''    def _store(self, target: ast.Subscript, val: _Reg, kind: str) -> None:
        if isinstance(target.value,ast.Name) and getattr(self.s.arrays.get(target.value.id),"view",None) is not None:
            from openassembler.pyfront.buffers import access
            access(self,target,val,kind)
            return
        arr, addr = self._address(target)''')
 s=s.replace('scope.arrays[arg.arg] = _Arr("global", t.elem.ptx, scope.new_reg("ptr"))', '''scope.arrays[arg.arg] = _Arr("global", t.elem.ptx, scope.new_reg("ptr"))
            from openassembler.pyfront.buffers import BufferType
            if isinstance(t,BufferType):scope.arrays[arg.arg].view=t''')
 s=s.replace('    if s.wide_integer:\n        contract =', '    if s.wide_integer or any(a.view is not None for a in s.arrays.values()):\n        contract =')
 anchor='    for sym, ty, count in s.shared:'
 s=s.replace(anchor,'''    views=[(pname,s.arrays[pname].view) for _,pname in s.params if pname in s.arrays and s.arrays[pname].view is not None]
    if views:
        from openassembler.pyfront.buffers import metadata
        out.insert(0,"// openassembler-pyfront-buffer-v1 "+json.dumps({"schema":1,"kernel":name,"buffers":[metadata(pname,view) for pname,view in views]},sort_keys=True,separators=(",",":")))
'''+anchor)
 files['compile.py']=s
 for name,text in files.items():(HERE/'candidate'/name).write_text(text)

if __name__=='__main__':build()

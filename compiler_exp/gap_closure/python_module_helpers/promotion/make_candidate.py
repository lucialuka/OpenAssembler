"""Stage per-module helper binding and graph custody without production mutation."""
from pathlib import Path
HERE=Path(__file__).resolve().parent

def once(text,old,new):
 assert text.count(old)==1,old
 return text.replace(old,new,1)

if __name__=='__main__':
 text=(HERE/'baseline-pyfront.py').read_text()
 text=once(text,'        for n in ast.walk(tree):\n            if isinstance(n, ast.Import):','        for n in tree.body if isinstance(tree, ast.Module) else ():\n            if isinstance(n, ast.Import):')
 text=once(text,'        from openassembler.pyfront.helpers import DeviceFunctions\n\n        self.devices = DeviceFunctions(tree, self)', '        from openassembler.pyfront.helpers import DeviceFunctions, _ModuleBindings\n        bindings = _ModuleBindings()\n        bindings.visit(tree)\n        self.defined.update(bindings.counts)\n        self.alias = {name: origin for name, origin in self.alias.items() if bindings.counts[name] == 1}\n        self.devices = DeviceFunctions(tree, self)')
 text=once(text,'  Device helpers: same-file @device functions with explicitly annotated scalar', '  Device helpers: same-file or statically resolved local-package @device functions\n          with explicitly annotated scalar')
 text=once(text,'          captures, recursion, early returns, runtime calls or cross-module helper imports are admitted.', '          captures, recursion, early returns or runtime calls are admitted. Local-package\n          imports use an explicit source root (or the enclosing package), retain exact\n          source/dependency/call identities, and execute no user module code.')
 text=once(text,'        self.alias: dict[str, str] = {}', '        self.source_tree = tree\n        self.alias: dict[str, str] = {}')
 text=once(text,'        self.path = path\n        super().__init__(self._render())','        self.path = path or getattr(node, "_oa_source_path", None)\n        super().__init__(self._render())')
 text=once(text,'        if fname is None:\n            raise NotSupported(f"call to {ast.unparse(node.func)!r} is not supported", node)', '''        if fname is None:
            graph = getattr(self.names, "graph", None)
            target = graph.resolve(self.names, node.func) if graph is not None else None
            if target is not None:
                target_names, target_name = target
                return target_names.devices.inline(self, node, name=target_name)
            raise NotSupported(f"call to {ast.unparse(node.func)!r} is not supported", node)''')
 text=once(text,'def compile_file(path: Path, arch: str = "sm_80") -> tuple[str, list[str]]:', 'def compile_file(path: Path, arch: str = "sm_80", *, source_root=None) -> tuple[str, list[str]]:')
 anchor='''        entries = [_compile_entry(k, names, arch) for k in kernels]
    except NotSupported as e:
        raise e.with_path(path) from None
    return HEADER.format(arch=arch) + "\\n".join(entries), [k.name for k in kernels]'''
 replacement='''        from .modules import SourceGraph
        graph = SourceGraph(path, tree, names, source_root)
        entries = []
        for k in kernels:
            graph.kernel = k.name
            entries.append(_compile_entry(k, names, arch))
        prefix = graph.prefix()
    except NotSupported as e:
        raise e.with_path(path) from None
    return prefix + HEADER.format(arch=arch) + "\\n".join(entries), [k.name for k in kernels]'''
 text=once(text,anchor,replacement)
 # KernelRef keeps source/root, and recompiles a fresh source snapshot per invocation.
 text=once(text,'        self.path, self.fdef, self.names = path, fdef, names','        self.path, self.fdef, self.names = path, fdef, names\n        self.source_root = None')
 text=once(text,'def find_kernel(path, entry: str | None) -> KernelRef:', 'def find_kernel(path, entry: str | None, *, source_root=None) -> KernelRef:')
 text=once(text,'    return KernelRef(path, fdef, names)','    reference = KernelRef(path, fdef, names)\n    reference.source_root = source_root\n    return reference')
 text=once(text,'def compile_kernel(fn, arch: str = "sm_80") -> str:', 'def compile_kernel(fn, arch: str = "sm_80", *, source_root=None) -> str:')
 anchor='''    if isinstance(fn, KernelRef):
        fdef, names, path = fn.fdef, fn.names, fn.path
    else:
        fdef, names, path = _live_source(getattr(fn, "__wrapped__", fn))
    try:
        return HEADER.format(arch=arch) + _compile_entry(fdef, names, arch)'''
 replacement='''    if isinstance(fn, KernelRef):
        reference = find_kernel(fn.path, fn.__name__, source_root=source_root or fn.source_root)
        fdef, names, path = reference.fdef, reference.names, reference.path
        source_root = reference.source_root
    else:
        fdef, names, path = _live_source(getattr(fn, "__wrapped__", fn))
    try:
        if path is not None:
            from .modules import SourceGraph
            graph = SourceGraph(path, names.source_tree, names, source_root)
            graph.kernel = fdef.name
            entry = _compile_entry(fdef, names, arch)
            return graph.prefix() + HEADER.format(arch=arch) + entry
        return HEADER.format(arch=arch) + _compile_entry(fdef, names, arch)'''
 text=once(text,anchor,replacement)
 (HERE/'candidate-pyfront.py').write_text(text)
 text=(HERE/'baseline-helpers.py').read_text()
 text=once(text,'    def inline(self, emitter, call):\n        name = call.func.id','    def inline(self, emitter, call, name=None):\n        name = name or call.func.id\n        graph = getattr(self.names, "graph", None)\n        active = graph.active if graph is not None else self.active\n        key = str(self.names.module_path)+":"+name if graph is not None else name')
 text=once(text,'        if name in self.active:\n            raise c.NotSupported(\'recursive device helper call: \' + \' -> \'.join([*self.active, name]), call)','        if key in active:\n            raise c.NotSupported(\'recursive device helper call: \' + \' -> \'.join(item.rsplit(\':\', 1)[-1] for item in [*active, key]), call)')
 text=once(text,'        if len(self.active) >= 32:','        if len(active) >= 32:')
 text=once(text,'        fn, kinds, result_kind, body = self._signature(name)', '        try:\n            fn, kinds, result_kind, body = self._signature(name)\n        except c.NotSupported as error:\n            if graph is not None:\n                raise error.with_path(self.names.module_path) from None\n            raise')
 text=once(text,'        self.active.append(name)','        active.append(key)\n        if graph is not None:\n            graph.note_call(self.names, name, fn, kinds, result_kind)')
 text=once(text,'            self.active.pop()','            active.pop()')
 (HERE/'candidate-helpers.py').write_text(text)

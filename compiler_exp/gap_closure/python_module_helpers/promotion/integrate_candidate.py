"""Stage normal build, ownership, CLI and remote graph integration."""
from pathlib import Path
HERE=Path(__file__).resolve().parent

def once(text,old,new):
 assert text.count(old)==1,old
 return text.replace(old,new,1)

if __name__=='__main__':
 text=(HERE/'baseline-build.py').read_text()
 text=once(text,'    device_library_options: list[dict] = field(default_factory=list)','    device_library_options: list[dict] = field(default_factory=list)\n    python_source_root: Path | None = None')
 text=once(text,'    device_link: dict | None = None\n\n    def record', '    device_link: dict | None = None\n    python_source_graph: dict | None = None\n\n    def record')
 text=once(text,"        if self.device_link is not None:r['device_link']=self.device_link", "        if self.device_link is not None:r['device_link']=self.device_link\n        if self.python_source_graph is not None:r['python_source_graph']=self.python_source_graph")
 anchor='''            ptx, names, notes = to_ptx_module(src, arch=opt.arch, include_dirs=opt.include_dirs,
                                              nvcc_flags=flags, **kwargs)'''
 new='''            if language == "python":
                from openassembler.pyfront.compile import compile_file
                ptx, names = compile_file(src, arch=opt.arch, source_root=opt.python_source_root)
                notes = []
                from openassembler.pyfront.modules import contract_from_ptx, validate_sources
                validate_sources(src, contract_from_ptx(ptx), opt.python_source_root)
            else:
                ptx, names, notes = to_ptx_module(src, arch=opt.arch, include_dirs=opt.include_dirs,
                                              nvcc_flags=flags, **kwargs)'''
 text=once(text,anchor,new)
 text=once(text,'        k.device_link = so.device_link','''        k.device_link = so.device_link
        from openassembler.pyfront.modules import contract_from_ptx
        try:
            k.python_source_graph = contract_from_ptx(ptx)
        except (ValueError, KeyError, TypeError) as error:
            k.error = f"invalid Python helper graph: {error}"
            so.kernels.append(k)
            continue''')
 anchor='''    p = so.paths(opt.no_sass)
    so.files = {p["cubin"]: cubin, p["ptx"]: ptx, p["reference"]: ref}'''
 new='''    if language == "python":
        try:
            from openassembler.pyfront.modules import contract_from_ptx, validate_sources
            validate_sources(src, contract_from_ptx(ptx), opt.python_source_root)
        except (OSError, ValueError) as error:
            so.error, so.rc = f"Python helper source validation failed: {error}", 1
            return so
    p = so.paths(opt.no_sass)
    so.files = {p["cubin"]: cubin, p["ptx"]: ptx, p["reference"]: ref}'''
 text=once(text,anchor,new)
 (HERE/'candidate-build.py').write_text(text)
 text=(HERE/'baseline-ownership.py').read_text()
 text=once(text,"        if kernel.get('device_link') is not None:\n            texts[kernel['name']]['device_link']=deepcopy(kernel['device_link'])", "        if kernel.get('device_link') is not None:\n            texts[kernel['name']]['device_link']=deepcopy(kernel['device_link'])\n        if kernel.get('python_source_graph') is not None:\n            texts[kernel['name']]['python_source_graph']=deepcopy(kernel['python_source_graph'])")
 text=once(text,"        if kernel.get('device_link') is not None:\n            texts[kernel['name']]['device_link']=kernel['device_link']", "        if kernel.get('device_link') is not None:\n            texts[kernel['name']]['device_link']=kernel['device_link']\n        if kernel.get('python_source_graph') is not None:\n            texts[kernel['name']]['python_source_graph']=kernel['python_source_graph']")
 (HERE/'candidate-ownership.py').write_text(text)
 text=(HERE/'baseline-cli.py').read_text()
 text=once(text,"device_libraries=[Path(p)for p in getattr(a,'device_library',[])])", "device_libraries=[Path(p)for p in getattr(a,'device_library',[])],\n                   python_source_root=Path(a.python_source_root)if getattr(a,'python_source_root',None)else None)")
 text=once(text,"or getattr(a,'device_library',[]):", "or getattr(a,'device_library',[]) or getattr(a,'python_source_root',None):")
 anchor="    pc.add_argument('--device-library',action='append',default=[],metavar='SOURCE',"
 text=once(text,anchor,"    pc.add_argument('--python-source-root',metavar='DIR',\n                    help='root for static local Python device-helper imports (default: enclosing package)')\n"+anchor)
 (HERE/'candidate-cli.py').write_text(text)
 text=(HERE/'baseline-remote.py').read_text()
 text=once(text,'    root = source if source.is_dir() else source.parent\n    files, _ = project.walk(root, output)', '''    root = source if source.is_dir() else source.parent
    from openassembler.pyfront.modules import source_root
    if options.python_source_root is not None:
        root = source_root(source if source.is_file() else source/'__entry__.py', options.python_source_root)
    elif source.is_file() and source.suffix == '.py':
        root = source_root(source)
    files, _ = project.walk(root, output)''')
 anchor="    from .compilation_database import discover as discover_commands\n    database, commands = discover_commands(source, options.compile_commands)"
 new='''    python_graphs = {}
    from openassembler.pyfront.compile import compile_file, NotSupported
    from openassembler.pyfront.modules import contract_from_ptx
    for path in files:
        if path.suffix != '.py' or not project.classify(path,path.name).language:
            continue
        try:
            ptx, _ = compile_file(path, source_root=options.python_source_root)
        except NotSupported:
            # Ordinary worker builds retain per-source failure/partial-output behavior.
            continue
        graph = contract_from_ptx(ptx)
        if graph is None:
            continue
        graph_root = source_root(path, options.python_source_root)
        for row in graph['files']:
            dependency = graph_root/row['path']
            if not dependency.is_relative_to(root):
                raise project.ProjectError('Python helper dependency is outside the bundled source root')
            add('source/'+dependency.relative_to(root).as_posix(),dependency)
            if project.sha256_file(dependency) != row['sha256']:
                raise project.ProjectError('Python helper dependency changed while bundling')
        python_graphs[path.relative_to(root).as_posix()] = graph
    if python_graphs:
        payload['.oa-python-graphs.json'] = json.dumps(python_graphs).encode()
    from .compilation_database import discover as discover_commands
    database, commands = discover_commands(source, options.compile_commands)'''
 text=once(text,anchor,new)
 text=once(text,"'path': 'source' if source.is_dir() else 'source/' + source.name,", "'path': 'source' if source == root else 'source/'+source.relative_to(root).as_posix(),")
 text=once(text,"'compile_command':None,", "'compile_command':None,\n                           'python_source_root':'source'if options.python_source_root is not None else None,")
 text=once(text,"            options['device_libraries']=[project.artifact_path(staging,rel)for rel in options.get('device_libraries',[])]", "            if options.get('python_source_root') is not None:\n                options['python_source_root']=project.artifact_path(staging,options['python_source_root'])\n            options['device_libraries']=[project.artifact_path(staging,rel)for rel in options.get('device_libraries',[])]")
 anchor="""            # The archive stores relative source records; local publication rebases them.
            files = {p.relative_to(output).as_posix(): p.read_bytes() for p in output.rglob('*')
                     if p.is_file() and p.name != 'writer.lock'}
            source_root = source if source.is_dir() else source.parent"""
 new="""            # Rebase package entry paths to the submitted root, including single
            # files whose containing package lives above the entry directory.
            manifest = project.load_manifest(output)
            source_root = staging/'source'
            graph_path = staging/'.oa-python-graphs.json'
            expected_graphs = json.loads(graph_path.read_text()) if graph_path.exists() else {}
            if manifest is not None:
                for entry in manifest['entries']:
                    original = project.entry_source_abs(manifest,entry)
                    relative = original.relative_to(source_root).as_posix()
                    expected = expected_graphs.get(relative)
                    if expected is not None:
                        for kernel in entry['kernels']:
                            if kernel.get('python_source_graph') != expected:
                                raise ValueError('remote Python helper graph differs from submitted source graph')
                    entry['source'] = relative
                manifest['root'] = str(source_root)
                (output/'manifest.json').write_text(json.dumps(manifest))
            files = {p.relative_to(output).as_posix(): p.read_bytes() for p in output.rglob('*')
                     if p.is_file() and p.name != 'writer.lock'}"""
 text=once(text,anchor,new)
 (HERE/'candidate-remote.py').write_text(text)

if __name__=='__main__':
    text=(HERE/'candidate-build.py').read_text()
    old='''            if language == "python":
                from openassembler.pyfront.compile import compile_file
                ptx, names = compile_file(src, arch=opt.arch, source_root=opt.python_source_root)
                notes = []
                from openassembler.pyfront.modules import contract_from_ptx, validate_sources
                validate_sources(src, contract_from_ptx(ptx), opt.python_source_root)
            else:
                ptx, names, notes = to_ptx_module(src, arch=opt.arch, include_dirs=opt.include_dirs,
                                              nvcc_flags=flags, **kwargs)'''
    new='''            if language == "python" and opt.python_source_root is not None:
                kwargs['python_source_root'] = opt.python_source_root
            ptx, names, notes = to_ptx_module(src, arch=opt.arch, include_dirs=opt.include_dirs,
                                              nvcc_flags=flags, **kwargs)
            if language == "python":
                from openassembler.pyfront.modules import contract_from_ptx, validate_sources
                validate_sources(src, contract_from_ptx(ptx), opt.python_source_root)'''
    (HERE/'candidate-build.py').write_text(once(text,old,new))
    text=(HERE/'baseline-frontend.py').read_text()
    text=once(text,'def _python_module(path: Path, arch: str)', 'def _python_module(path: Path, arch: str, source_root=None)')
    text=once(text,'    ptx, names = compile_file(path, arch)', '    ptx, names = compile_file(path, arch, source_root=source_root)')
    text=once(text,'source_argument: str | None = None) -> tuple[str, list[str], list[str]]:', 'source_argument: str | None = None, python_source_root: Path | None = None) -> tuple[str, list[str], list[str]]:')
    text=once(text,'        return _python_module(path, arch)','        return _python_module(path, arch, python_source_root)')
    (HERE/'candidate-frontend.py').write_text(text)
    text=(HERE/'candidate-remote.py').read_text()
    text=once(text,"or source.is_dir() and (root / rel.removeprefix('source/') in commands", "or source.is_dir() and (root / rel.removeprefix('source/')).is_relative_to(source) and (root / rel.removeprefix('source/') in commands")
    (HERE/'candidate-remote.py').write_text(text)

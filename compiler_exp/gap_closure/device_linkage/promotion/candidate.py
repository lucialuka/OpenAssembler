"""Generate isolated normal-path device-library candidates before promotion."""
from pathlib import Path
import importlib,builtins,sys
HERE=Path(__file__).resolve().parent

def once(source,old,new):
    assert source.count(old)==1,old
    return source.replace(old,new,1)

def generate():
    outputs={}
    text=(HERE/'baseline-ptx-inlining.py').read_text()
    anchor="        env=[dict(formal)];lines=[];returns=[];end_label=fresh()+'return';invocation=fresh()"
    replacement=(HERE/'debug-line.txt').read_text()+anchor
    text=once(text,anchor,replacement)
    outputs['inlining']=text

    text=(HERE/'baseline-cppfront.py').read_text()
    text=once(text,'source_argument: str | None = None) -> tuple[str, list[str]]:', 'source_argument: str | None = None, allow_device_library: bool = False) -> tuple[str, list[str]]:')
    text=once(text,'    if not names:\n','    if not names and not allow_device_library:\n')
    outputs['cppfront']=text
    text=(HERE/'baseline-build.py').read_text()
    text=once(text,'    compile_command: object = None','    compile_command: object = None\n    device_libraries: list[Path] = field(default_factory=list)\n    device_library_options: list[dict] = field(default_factory=list)')
    text=once(text,"            ptx, names, notes = to_ptx_module(src, arch=opt.arch, include_dirs=opt.include_dirs,","            if opt.device_libraries and language=='cuda':\n                from openassembler.ptx.link import relocatable_flags\n                flags=relocatable_flags(flags)\n            ptx, names, notes = to_ptx_module(src, arch=opt.arch, include_dirs=opt.include_dirs,")
    old="                if opt.compile_command is not None:\n                    from .compilation_database import dependencies"
    text=once(text,old,"                if opt.compile_command is not None and not opt.device_libraries:\n                    from .compilation_database import dependencies")
    old="                                              nvcc_flags=flags, **kwargs)"
    new=old+"\n            if opt.device_libraries and language=='cuda':\n                from .compilation_database import dependencies,CompileCommand\n                command=opt.compile_command\n                if command is None:\n                    command=CompileCommand(src.resolve(),str(src.resolve()),src.resolve().parent,(),\n                        ('nvcc',str(src.resolve())),src.resolve(),so.sha,{})\n                    extra=flags\n                else:\n                    from openassembler.ptx.link import relocatable_flags\n                    extra=relocatable_flags(opt.nvcc_flags)\n                dependency=dependencies(command,include_dirs=opt.include_dirs,extra_flags=extra)\n                so.device_link={'entry_compilation':{'source':str(src.resolve()),\n                    'source_sha256':so.sha,'device_flags':list(flags),'dependencies':dependency,\n                    'compile_contract':command.record()if opt.compile_command is not None else None}}"
    text=once(text,old,new)
    text=once(text,'    inlined_device_calls: dict | None = None','    inlined_device_calls: dict | None = None\n    device_link: dict | None = None')
    anchor="        if self.inlined_device_calls is not None:\n            r['inlined_device_calls']=self.inlined_device_calls"
    text=once(text,anchor,anchor+"\n        if self.device_link is not None:r['device_link']=self.device_link")
    text=once(text,'    build_contract: dict | None = None','    build_contract: dict | None = None\n    device_link: dict | None = None')
    anchor='        so.notes += notes'
    text=once(text,anchor,'''        if opt.device_libraries:
            if src.resolve() in {Path(p).resolve()for p in opt.device_libraries}:
                raise ValueError('the entry source cannot also be a device library')
            from openassembler.ptx.link import link_paths
            entry_contract=so.device_link
            ptx,so.device_link=link_paths(ptx,opt.device_libraries,arch=opt.arch,
                include_dirs=opt.include_dirs,nvcc_flags=opt.nvcc_flags,
                compile_commands=opt.compile_commands,library_options=opt.device_library_options)
            if entry_contract:so.device_link.update(entry_contract)
            if P.sha256_file(src)!=so.sha:raise ValueError('entry source changed during device linking')
        so.notes += notes''')
    text=once(text,'        k.parameter_attributes = module.kernels[kname].parameter_attributes','        k.parameter_attributes = module.kernels[kname].parameter_attributes\n        k.device_link = so.device_link')
    text=once(text,'            cls = [P.classify(f, f.relative_to(root).as_posix()) for f in files]','            library_paths={Path(p).resolve()for p in opt.device_libraries}\n            cls = [P.classify(f, f.relative_to(root).as_posix()) for f in files if f.resolve()not in library_paths]')
    anchor="        o = Options(**{**opt.__dict__, 'include_dirs':includes,'compile_command':command})"
    text=once(text,anchor,"        o = Options(**{**opt.__dict__, 'include_dirs':includes,'compile_command':command,\n                       'compile_commands':_database if opt.device_libraries and _database else opt.compile_commands})")
    anchor="            ptx, names, notes = to_ptx_module(src, arch=opt.arch, include_dirs=opt.include_dirs,"
    extra="""            if opt.device_libraries and language=='cuda':
                from .compilation_database import dependencies,CompileCommand
                check_command=opt.compile_command or CompileCommand(src.resolve(),str(src.resolve()),src.resolve().parent,(),
                    ('nvcc',str(src.resolve())),src.resolve(),so.sha,{})
                before_dependencies=dependencies(check_command,include_dirs=opt.include_dirs,
                    extra_flags=relocatable_flags(opt.nvcc_flags)if opt.compile_command else flags)
"""
    text=once(text,anchor,extra+anchor)
    anchor="                dependency=dependencies(command,include_dirs=opt.include_dirs,extra_flags=extra)"
    text=once(text,anchor,anchor+"\n                if before_dependencies!=dependency:raise ValueError('entry dependencies changed during compilation')")
    outputs['build']=text
    text=(HERE/'baseline-native_ownership.py').read_text()
    old="        if kernel.get('inlined_device_calls') is not None:\n            texts[kernel['name']]['inlined_device_calls']=deepcopy(kernel['inlined_device_calls'])"
    text=once(text,old,old+"\n        if kernel.get('device_link') is not None:\n            texts[kernel['name']]['device_link']=deepcopy(kernel['device_link'])")
    old="        if kernel.get('inlined_device_calls') is not None:\n            texts[kernel['name']]['inlined_device_calls']=kernel['inlined_device_calls']"
    text=once(text,old,old+"\n        if kernel.get('device_link') is not None:\n            texts[kernel['name']]['device_link']=kernel['device_link']")
    outputs['native_ownership']=text
    text=(HERE/'baseline-cli.py').read_text()
    text=once(text,"compile_commands=Path(a.compile_commands) if a.compile_commands else None)","compile_commands=Path(a.compile_commands) if a.compile_commands else None,\n                   device_libraries=[Path(p)for p in getattr(a,'device_library',[])])")
    text=once(text,"or a.compile_commands:\n", "or a.compile_commands or getattr(a,'device_library',[]):\n")
    anchor='    pc.add_argument("--dump-ptx", action="store_true", help="print the PTX each source compiled to")'
    text=once(text,anchor,"    pc.add_argument('--device-library',action='append',default=[],metavar='SOURCE',\n                    help='link scalar device functions from PTX or CUDA source (repeatable)')\n"+anchor)
    outputs['cli']=text
    text=(HERE/'baseline-remote_build.py').read_text()
    anchor='    return root, payload, includes'
    extra="""    if options.device_libraries:
        from dataclasses import replace
        from .compilation_database import load
        records=[]
        for index, library in enumerate(options.device_libraries):
            library=Path(library).resolve()
            if not library.is_file():raise project.ProjectError('device library missing: '+str(library))
            selected=load(options.compile_commands).get(library)if options.compile_commands else None
            database=options.compile_commands if selected else None
            library_options=replace(options,device_libraries=[],device_library_options=[],compile_commands=database)
            _library_root,library_files,library_includes=_sources(library,output,library_options)
            _,library_commands=discover_commands(library,database)
            if not library_commands:
                library_files.pop('source/compile_commands.json',None)
            prefix=f'device-libraries/{index}/'
            for relative,data in library_files.items():
                if len(payload)>=MAX_FILES or total+len(data)>MAX_BYTES:
                    raise project.ProjectError('device library source bundle exceeds 256 MiB / 20000 files')
                payload[prefix+relative]=data;total+=len(data)
            records.append({'path':prefix+'source/'+library.name,
                'source_origin':str(library),'source_sha256':project.sha256_file(library),
                'include_dirs':[prefix+name for name in library_includes],
                'compile_commands':prefix+'source/compile_commands.json'if library_commands else None})
        payload['.oa-device-libraries.json']=json.dumps(records).encode()
"""
    text=once(text,anchor,extra+anchor)
    anchor="    for path in files:\n        add('source/' + path.relative_to(root).as_posix(), path)"
    text=once(text,anchor,"    library_paths={Path(p).resolve()for p in options.device_libraries}\n    for path in files:\n        if path.resolve()not in library_paths:add('source/' + path.relative_to(root).as_posix(), path)")
    anchor='    database, commands = discover_commands(source, options.compile_commands)'
    text=once(text,anchor,anchor+"\n    commands={path:command for path,command in commands.items()if path not in library_paths}")

    anchor='                               root, payload, includes, candidates)'
    text=once(text,anchor,"                    root, payload, includes, candidates,\n                    json.loads(files.get('.oa-device-libraries.json',b'[]')))" )
    anchor='                    root, payload, includes, candidates):'
    text=once(text,anchor,'                    root, payload, includes, candidates, libraries=None):')
    anchor="    request = {'source': base64.b64encode(payload).decode(),"
    text=once(text,anchor,"    libraries=libraries or []\n"+anchor)
    anchor="                           'compile_command':None}, 'name': name}"
    text=once(text,anchor,"                           'compile_command':None,\n                           'device_libraries':[row['path']for row in libraries],\n                           'device_library_options':[{k:v for k,v in row.items()if k!='path'}for row in libraries]}, 'name': name}")
    anchor="    if not candidates:\n        raise project.ProjectError('nothing to compile: add supported GPU kernels, or configure '"
    text=once(text,anchor,"    library_paths={Path(p).resolve()for p in options.device_libraries}\n    candidates=[rel for rel in candidates if (root/rel).resolve()not in library_paths]\n"+anchor)
    anchor="        count = len({path for path in files if project.classify(path,path.name).language} | set(commands))"
    text=once(text,anchor,"        library_paths={Path(p).resolve()for p in options.device_libraries}\n        count = len(({path for path in files if project.classify(path,path.name).language} | set(commands))-library_paths)")
    anchor="            if options.get('compile_commands'):\n                options['compile_commands']=project.artifact_path(staging,options['compile_commands'])"
    text=once(text,anchor,anchor+"\n            options['device_libraries']=[project.artifact_path(staging,rel)for rel in options.get('device_libraries',[])]\n            for library in options.get('device_library_options',[]):\n                library['include_dirs']=[project.artifact_path(staging,rel)for rel in library['include_dirs']]\n                if library.get('compile_commands'):\n                    library['compile_commands']=project.artifact_path(staging,library['compile_commands'])")
    outputs['remote_build']=text
    return outputs

def install():
    from types import ModuleType
    module=importlib.import_module('openassembler.ptx.inlining');path=HERE/'candidate-inlining.py'
    exec(builtins.compile(path.read_bytes(),str(path),'exec'),module.__dict__)
    name='openassembler.ptx.link';module=ModuleType(name);module.__file__=str(HERE/'link.py');sys.modules[name]=module
    exec(builtins.compile((HERE/'link.py').read_bytes(),module.__file__,'exec'),module.__dict__)
    for name in('cppfront','native_ownership','build','cli','remote_build'):
        module=importlib.import_module('openassembler.'+name);path=HERE/('candidate-'+name+'.py')
        exec(builtins.compile(path.read_bytes(),str(path),'exec'),module.__dict__)

if __name__=='__main__':
    for name,source in generate().items():(HERE/('candidate-'+name+'.py')).write_text(source)

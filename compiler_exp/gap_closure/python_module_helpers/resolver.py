"""Isolated deterministic local source graph; never import or execute user modules."""
from __future__ import annotations
import ast,copy,hashlib
from pathlib import Path
from openassembler.pyfront import compile as c
from openassembler.pyfront.helpers import _ModuleBindings

class ResolutionError(ValueError):
    pass

class Resolver:
    def __init__(self, entry, root=None):
        self.entry=Path(entry).resolve()
        self.root=Path(root).resolve()if root is not None else self.entry.parent
        if not self.entry.is_relative_to(self.root):raise ResolutionError('entry outside explicit source root')
        self.modules={};self.functions={};self.active=[];self.total_bytes=0;self.origins={}
        self.entry_name='.'.join(self.entry.relative_to(self.root).with_suffix('').parts)

    def module(self,name):
        if name in self.modules:return self.modules[name]
        if not name or any(not p.isidentifier()for p in name.split('.')):raise ResolutionError('invalid local module '+name)
        relative=Path(*name.split('.'));choices=[self.root/relative.with_suffix('.py'),self.root/relative/'__init__.py']
        found=[p for p in choices if p.is_file()]
        if len(found)!=1:raise ResolutionError('local module missing or ambiguous: '+name)
        path=found[0]
        if path.is_symlink()or not path.resolve().is_relative_to(self.root):raise ResolutionError('module path escapes source root: '+name)
        if len(self.modules)>=64:raise ResolutionError('source graph exceeds 64 modules')
        raw=path.read_bytes();self.total_bytes+=len(raw)
        if self.total_bytes>8*1024*1024:raise ResolutionError('source graph exceeds 8 MiB')
        tree=ast.parse(raw,filename=str(path));names=c._Names(tree);bindings=_ModuleBindings();bindings.visit(tree)
        package=name if path.name=='__init__.py'else name.rpartition('.')[0]
        imports={};definitions={}
        for node in tree.body:
            if isinstance(node,ast.Import):
                for alias in node.names:
                    local=alias.asname or alias.name.split('.')[0]
                    imports[local]=('module',alias.name if alias.asname else local)
            elif isinstance(node,ast.ImportFrom):
                base=node.module or''
                if node.level:
                    pieces=package.split('.')if package else[]
                    if node.level>len(pieces):raise ResolutionError('relative import escapes source root')
                    base='.'.join(pieces[:len(pieces)-node.level+1]+([base]if base else[]))
                for alias in node.names:
                    if alias.name!='*':imports[alias.asname or alias.name]=('member',base,alias.name)
            elif isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)):
                definitions[node.name]=node
        info=dict(name=name,path=path,raw=raw,tree=tree,names=names,bindings=bindings.counts,imports=imports,definitions=definitions)
        self.modules[name]=info
        return info

    def function_target(self,module,expression):
        parts=[]
        while isinstance(expression,ast.Attribute):parts.insert(0,expression.attr);expression=expression.value
        if not isinstance(expression,ast.Name):raise ResolutionError('call target is not a static local name')
        name=expression.id
        if module['bindings'][name]!=1:raise ResolutionError('call target rebound or unresolved: '+name)
        if not parts and name in module['definitions']:return module['name'],name
        binding=module['imports'].get(name)
        if binding is None:raise ResolutionError('unknown source call target: '+name)
        if binding[0]=='module':
            if not parts:raise ResolutionError('module cannot be a device helper')
            return '.'.join([binding[1],*parts[:-1]]),parts[-1]
        base,member=binding[1:]
        if parts:return '.'.join([base,member,*parts[:-1]]),parts[-1]
        return base,member

    def add_function(self,module_name,name):
        key=(module_name,name)
        if key in self.active:raise ResolutionError('recursive source helper: '+' -> '.join('.'.join(x)for x in[*self.active,key]))
        if key in self.functions:return self.functions[key]
        if len(self.functions)>=1024 or len(self.active)>=32:raise ResolutionError('helper source expansion exceeds limit')
        module=self.module(module_name);fn=module['definitions'].get(name)
        if fn is None or name not in module['names'].devices.definitions:raise ResolutionError('import target is not a typed @device helper: '+'.'.join(key))
        if module['bindings'][name]!=1:raise ResolutionError('helper definition rebound: '+name)
        unique='__oa_module_'+str(len(self.functions))+'_'+name;self.functions[key]=unique;self.active.append(key)
        try:
            # Existing helper validation checks type/effect/flow before normalization.
            module['names'].devices._signature(name)
            revised=self.rewrite(module,fn);revised.name=unique
            self.origins[unique]=dict(path=str(module['path']),function=name,line=fn.lineno)
            self.functions[key]=(unique,revised)
        finally:self.active.pop()
        return self.functions[key]

    def rewrite(self,module,fn):
        result=copy.deepcopy(fn)
        local={a.arg for a in fn.args.args}|{n.id for n in ast.walk(fn)if isinstance(n,ast.Name)and isinstance(n.ctx,ast.Store)}
        owner=self
        class Rewrite(ast.NodeTransformer):
            def visit_Call(self,node):
                node.args=[self.visit(arg)for arg in node.args]
                node.keywords=[ast.keyword(arg=k.arg,value=self.visit(k.value))for k in node.keywords]
                root=node.func
                while isinstance(root,ast.Attribute):root=root.value
                if isinstance(root,ast.Name)and root.id in local:raise ResolutionError('call target is a local value: '+root.id)
                intrinsic=module['names'].resolve(node.func)
                if intrinsic:
                    if intrinsic.startswith('pf.'):
                        node.func=ast.Name(id=intrinsic[3:],ctx=ast.Load())
                    elif intrinsic.startswith('math.'):
                        node.func=ast.Attribute(value=ast.Name(id='math',ctx=ast.Load()),attr=intrinsic[5:],ctx=ast.Load())
                    return node
                target=owner.function_target(module,node.func);resolved=owner.add_function(*target)
                if isinstance(resolved,str):raise ResolutionError('recursive helper resolution')
                node.func=ast.Name(id=resolved[0],ctx=ast.Load());return node
            def visit_Name(self,node):
                if node.id in local:return node
                resolved=module['names'].resolve(node)
                if resolved and resolved.startswith('pf.'):return ast.copy_location(ast.Name(id=resolved[3:],ctx=node.ctx),node)
                return node
            def visit_Attribute(self,node):
                resolved=module['names'].resolve(node)
                if resolved and resolved.startswith('pf.'):return ast.copy_location(ast.Name(id=resolved[3:],ctx=ast.Load()),node)
                return self.generic_visit(node)
        # Decorators/annotations resolve in their defining module before canonical imports.
        return Rewrite().visit(result)

    def resolve(self):
        entry=self.module(self.entry_name)
        kernels=c._file_kernels(entry['tree'],entry['names'],self.entry)
        rewritten=[]
        for fn in kernels:
            revised=self.rewrite(entry,fn);rewritten.append(revised)
            self.origins[fn.name]=dict(path=str(self.entry),function=fn.name,line=fn.lineno)
        imports=ast.parse('from openassembler.pyfront import *\nimport math\n').body
        helpers=[value[1]for value in self.functions.values()if isinstance(value,tuple)]
        tree=ast.fix_missing_locations(ast.Module(body=imports+helpers+rewritten,type_ignores=[]))
        dependencies=[dict(path=str(info['path'].relative_to(self.root)),sha256=hashlib.sha256(info['raw']).hexdigest())for _,info in sorted(self.modules.items())]
        return tree,dict(schema=1,strategy='static-local-python-helper-source-graph',root=str(self.root),dependencies=dependencies,origins=self.origins)

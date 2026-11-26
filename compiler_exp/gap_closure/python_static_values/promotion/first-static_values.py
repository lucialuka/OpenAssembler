"""Explicit immutable source values for the static typed device language."""
from __future__ import annotations
import ast
import copy
import math
import operator
import struct

MISSING = object()
BINARY = {ast.Add:operator.add, ast.Sub:operator.sub, ast.Mult:operator.mul,
          ast.Div:operator.truediv, ast.FloorDiv:operator.floordiv, ast.Mod:operator.mod,
          ast.LShift:operator.lshift, ast.RShift:operator.rshift,
          ast.BitAnd:operator.and_, ast.BitOr:operator.or_, ast.BitXor:operator.xor}
UNARY = {ast.UAdd:operator.pos, ast.USub:operator.neg, ast.Invert:operator.invert, ast.Not:operator.not_}
COMPARE = {ast.Eq:operator.eq,ast.NotEq:operator.ne,ast.Lt:operator.lt,ast.LtE:operator.le,ast.Gt:operator.gt,ast.GtE:operator.ge}


def encoded(value):
    if type(value) is tuple:
        return {'type':'tuple','items':[encoded(item) for item in value]}
    if type(value) is float:
        return {'type':'float','bits':struct.pack('>d',value).hex()}
    if type(value) is bool:
        return {'type':'bool','value':value}
    return {'type':'int','value':str(value)}


class StaticValues:
    def __init__(self, graph):
        self.graph=graph
        self.cache={}
        self.active=[]
        self.records={}
        self.declarations={}
        self.uses={}
        self.steps=0

    def fail(self,message,node=None,path=None):
        self.graph.fail('static value: '+message,node,path)

    def final(self,info,annotation):
        names=info['names']
        if isinstance(annotation,ast.Constant) and isinstance(annotation.value,str):
            try:annotation=ast.parse(annotation.value,mode='eval').body
            except (SyntaxError,ValueError):return None
        base=annotation.value if isinstance(annotation,ast.Subscript)else annotation
        dotted=names._dotted(base)
        if dotted not in ('typing.Final','typing_extensions.Final'):
            return None
        root=base
        while isinstance(root,ast.Attribute):root=root.value
        if not isinstance(root,ast.Name)or info['bindings'][root.id]!=1:
            self.fail('Final marker is rebound',annotation,info['path'])
        return annotation.slice if isinstance(annotation,ast.Subscript)else True

    def declared(self,info):
        key=info['name']
        if key not in self.declarations:
            declarations={}
            for node in info['tree'].body:
                if isinstance(node,ast.AnnAssign)and isinstance(node.target,ast.Name):
                    annotation=self.final(info,node.annotation)
                    if annotation is not None:declarations[node.target.id]=(node,annotation)
            self.declarations[key]=declarations
        return self.declarations[key]

    def target(self,info,expression,seen=()):
        parts=[];head=expression
        while isinstance(head,ast.Attribute):parts.insert(0,head.attr);head=head.value
        if not isinstance(head,ast.Name):return None
        name=head.id
        if not parts and name in self.declared(info):return info,name
        binding=info['imports'].get(name)
        if binding is None:return None
        # Framework/type imports and runtime symbols are not constant sources.
        origin=binding[1]
        if origin.startswith(('openassembler','typing','math')):return None
        if info['bindings'][name]!=1 or name in info['mutations']:
            self.fail('import target is rebound or mutated: '+name,expression)
        if info['stars']:self.fail('wildcard imports require explicit bindings',expression)
        if binding[0]=='module':
            if not parts:return None
            module,member='.'.join([origin,*parts[:-1]]),parts[-1]
        else:
            module,member=('.'.join([origin,binding[2],*parts[:-1]]),parts[-1])if parts else(origin,binding[2])
        key=(module,member)
        if key in seen or len(seen)>=32:self.fail('cyclic or excessive constant re-export',expression)
        module_info=self.graph._load(module,expression)
        if member in self.declared(module_info):return module_info,member
        if member in module_info['imports']:
            ref=ast.copy_location(ast.Name(id=member,ctx=ast.Load()),expression)
            return self.target(module_info,ref,(*seen,key))
        return None

    def definition(self,info,name,use):
        key=(info['name'],name)
        declaration,annotation=self.declared(info)[name]
        if info['bindings'][name]!=1 or name in info['mutations']:
            self.fail('Final binding is rebound or mutated: '+name,declaration,info['path'])
        if declaration.value is None:self.fail('Final declaration needs a value',declaration,info['path'])
        if key in self.active or len(self.active)>=32:self.fail('cyclic or excessive constant dependency',use,info['path'])
        if key not in self.cache:
            self.active.append(key)
            try:
                value=self.evaluate(declaration.value,info,required=True)
                self.check_annotation(value,annotation,declaration,info)
                self.cache[key]=value
            finally:self.active.pop()
        value=self.cache[key]
        relative=info['path'].relative_to(self.graph.root).as_posix()
        record=self.records.setdefault(relative+':'+name,dict(path=relative,name=name,line=declaration.lineno,
                        value=encoded(value),kernels={}))
        kernel=self.graph.kernel
        if kernel is not None:
            uses=self.uses.setdefault((relative+':'+name,kernel),set())
            uses.add((getattr(use,'_oa_source_path',str(info['path'])),getattr(use,'lineno',0),getattr(use,'col_offset',0)))
            record['kernels'][kernel]=len(uses)
        return value

    def check_annotation(self,value,annotation,node,info,depth=0):
        if annotation is True:return
        if depth>32:self.fail('annotation exceeds depth 32',node,info['path'])
        if isinstance(annotation,ast.Name):
            expected={'int':int,'float':float,'bool':bool}.get(annotation.id)
            if info['bindings'][annotation.id]:self.fail('Final annotation type is rebound',node,info['path'])
            if expected is None or type(value)is not expected:
                self.fail('Final annotation and literal type disagree',node,info['path'])
            return
        if isinstance(annotation,ast.Subscript)and isinstance(annotation.value,ast.Name)and annotation.value.id=='tuple':
            if info['bindings']['tuple']:self.fail('Final tuple annotation is rebound',node,info['path'])
            if type(value)is not tuple:self.fail('Final tuple annotation needs a tuple',node,info['path'])
            kinds=list(annotation.slice.elts)if isinstance(annotation.slice,ast.Tuple)else[annotation.slice]
            if len(kinds)==2 and isinstance(kinds[1],ast.Constant)and kinds[1].value is Ellipsis:kinds=[kinds[0]]*len(value)
            if len(kinds)!=len(value):self.fail('Final tuple annotation has different arity',node,info['path'])
            for item,kind in zip(value,kinds):self.check_annotation(item,kind,node,info,depth+1)
            return
        self.fail('Final annotation requires int, float, bool or a finite tuple',node,info['path'])

    def bounded(self,value,node,depth=0):
        if type(value)is int:
            if value.bit_length()>2048:self.fail('integer exceeds 2048 bits',node)
        elif type(value)is float:
            if not math.isfinite(value):self.fail('nonfinite compile-time float',node)
        elif type(value)is tuple:
            if depth>32 or len(value)>256:self.fail('tuple exceeds depth/256-element bound',node)
            for child in value:self.bounded(child,node,depth+1)
        elif type(value)is not bool:self.fail('only numeric scalars and immutable tuples are admitted',node)
        return value

    def evaluate(self,node,info,required=False,locals_=(),depth=0):
        self.steps+=1
        if self.steps>65536:self.fail('evaluation exceeds 65536 expression nodes',node)
        if depth>32:self.fail('expression exceeds depth 32',node)
        recurse=lambda item:self.evaluate(item,info,required,locals_,depth+1)
        try:
            if isinstance(node,ast.Constant)and type(node.value)in(int,float,bool):value=node.value
            elif isinstance(node,(ast.Name,ast.Attribute)):
                root=node
                while isinstance(root,ast.Attribute):root=root.value
                if isinstance(root,ast.Name)and root.id in locals_:return MISSING
                target=self.target(info,node)
                if target is None:
                    if required:self.fail('value is not an explicit Final declaration',node,info['path'])
                    return MISSING
                value=self.definition(*target,node)
            elif isinstance(node,ast.Tuple):
                if len(node.elts)>256:self.fail('tuple exceeds 256 elements',node)
                values=[recurse(item)for item in node.elts]
                if any(item is MISSING for item in values):return MISSING
                value=tuple(values)
            elif isinstance(node,ast.UnaryOp)and type(node.op)in UNARY:
                item=recurse(node.operand)
                if item is MISSING:return MISSING
                value=UNARY[type(node.op)](item)
            elif isinstance(node,ast.BinOp)and type(node.op)in BINARY:
                left,right=recurse(node.left),recurse(node.right)
                if left is MISSING or right is MISSING:return MISSING
                if type(left)not in(int,float)or type(right)not in(int,float):self.fail('numeric operands required',node)
                if isinstance(node.op,(ast.LShift,ast.RShift))and(type(right)is not int or not 0<=right<=2048):self.fail('shift must be in [0,2048]',node)
                value=BINARY[type(node.op)](left,right)
            elif isinstance(node,ast.Compare)and all(type(op)in COMPARE for op in node.ops):
                values=[recurse(item)for item in[node.left,*node.comparators]]
                if any(item is MISSING for item in values):return MISSING
                value=all(COMPARE[type(op)](left,right)for op,left,right in zip(node.ops,values,values[1:]))
            elif isinstance(node,ast.Subscript):
                container=recurse(node.value)
                if container is MISSING:return MISSING
                if type(container)is not tuple:
                    if required:self.fail('compile-time index requires a tuple',node)
                    return MISSING
                if isinstance(node.slice,ast.Slice):
                    parts=[None if item is None else recurse(item)for item in(node.slice.lower,node.slice.upper,node.slice.step)]
                    if any(item is MISSING for item in parts):self.fail('tuple slice must be static',node)
                    if any(item is not None and type(item)is not int for item in parts):self.fail('tuple slice indices must be integers',node)
                    value=container[slice(*parts)]
                else:
                    index=recurse(node.slice)
                    if index is MISSING:self.fail('tuple index must be static',node)
                    if type(index)is not int:self.fail('tuple index must be an integer',node)
                    value=container[index]
            elif isinstance(node,ast.Call)and isinstance(node.func,ast.Name)and node.func.id=='len'and len(node.args)==1 and not node.keywords and info['bindings']['len']==0:
                container=recurse(node.args[0])
                if container is MISSING:return MISSING
                if type(container)is not tuple:self.fail('compile-time len requires a tuple',node)
                value=len(container)
            else:
                if required:self.fail('unsupported expression in Final declaration',node,info['path'])
                return MISSING
        except (ArithmeticError,TypeError,IndexError,ValueError)as error:
            from .compile import NotSupported
            if isinstance(error,NotSupported):raise
            self.fail('invalid compile-time operation: '+str(error),node,info['path'])
        return self.bounded(value,node)

    def fold(self,names,fn):
        info=self.graph.modules[names.module_name]
        declarations=self.declared(info)
        if not declarations and not info['imports']:return fn
        local={arg.arg for arg in fn.args.args}|{node.id for node in ast.walk(fn)if isinstance(node,ast.Name)and isinstance(node.ctx,(ast.Store,ast.Del))}
        owner=self
        class Fold(ast.NodeTransformer):
            def visit(self,node):
                if not isinstance(node,ast.expr):return super().visit(node)
                # Only expressions involving an explicit source constant are folded.
                refs=[item for item in ast.walk(node)if isinstance(item,(ast.Name,ast.Attribute))]
                static=False
                for item in refs:
                    root=item
                    while isinstance(root,ast.Attribute):root=root.value
                    if isinstance(root,ast.Name)and root.id in local:continue
                    if owner.target(info,item)is not None:static=True;break
                if static:
                    value=owner.evaluate(node,info,locals_=local)
                    if value is not MISSING:
                        if type(value)is tuple:owner.fail('tuple value must be indexed, sliced or used by len at compile time',node,info['path'])
                        result=ast.copy_location(ast.Constant(value=value),node)
                        if hasattr(node,'_oa_source_path'):result._oa_source_path=node._oa_source_path
                        return result
                return super().visit(node)
            def visit_Call(self,node):
                node.args=[self.visit(arg)for arg in node.args]
                node.keywords=[ast.keyword(arg=kw.arg,value=self.visit(kw.value))for kw in node.keywords]
                return node
        result=copy.deepcopy(fn)
        result.body=[Fold().visit(node)for node in result.body]
        return ast.fix_missing_locations(result)


def validate_records(records,paths):
    if not isinstance(records,dict)or not records or len(records)>1024:
        raise ValueError('invalid static value inventory')
    def check(value,depth=0):
        if depth>32 or not isinstance(value,dict):raise ValueError('invalid static value encoding')
        kind=value.get('type')
        if kind=='tuple':
            if set(value)!={'type','items'}or not isinstance(value['items'],list)or len(value['items'])>256:raise ValueError('invalid static tuple')
            for item in value['items']:check(item,depth+1)
        elif kind=='int':
            text=value.get('value');number=int(text)
            if set(value)!={'type','value'}or str(number)!=text or number.bit_length()>2048:raise ValueError('invalid static integer')
        elif kind=='float':
            bits=value.get('bits')
            if set(value)!={'type','bits'}or not isinstance(bits,str)or len(bits)!=16 or not math.isfinite(struct.unpack('>d',bytes.fromhex(bits))[0]):raise ValueError('invalid static float')
        elif kind=='bool':
            if set(value)!={'type','value'}or type(value.get('value'))is not bool:raise ValueError('invalid static Boolean')
        else:raise ValueError('invalid static value type')
    for key,row in records.items():
        if row['path']not in paths or key!=row['path']+':'+row['name']or type(row['line'])is not int or row['line']<=0:raise ValueError('invalid static source identity')
        if not isinstance(row['kernels'],dict)or not row['kernels']or any(type(count)is not int or count<=0 for count in row['kernels'].values()):raise ValueError('invalid static use inventory')
        check(row['value'])

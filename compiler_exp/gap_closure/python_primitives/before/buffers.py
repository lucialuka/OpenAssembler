"""Candidate pure compile-time buffer contracts and checked scalar accesses."""
from __future__ import annotations
import ast,math
from dataclasses import dataclass
from openassembler.pyfront.types import ArrayType,Scalar

@dataclass(frozen=True)
class BufferType(ArrayType):
    shape: tuple[int,...]
    strides: tuple[int,...]
    alignment: int
    access: str

    @property
    def storage_elements(self):return 1+sum((n-1)*s for n,s in zip(self.shape,self.strides))


def Buffer(elem,shape,*,strides=None,alignment=None,access='readwrite'):
    if not isinstance(elem,Scalar) or elem.ptx not in ('s32','u32','f32','s64','u64'):raise ValueError('Buffer requires a supported scalar element type')
    if isinstance(shape,int):shape=(shape,)
    if not isinstance(shape,(tuple,list)) or not 1<=len(shape)<=4 or any(type(n)is not int or not 1<=n<=2**31-1 for n in shape):raise ValueError('Buffer shape needs one to four positive int32 dimensions')
    shape=tuple(shape);size=8 if elem.ptx.endswith('64') else 4
    if strides is None:strides=tuple(math.prod(shape[i+1:]) for i in range(len(shape)))
    if not isinstance(strides,(tuple,list)) or len(strides)!=len(shape) or any(type(s)is not int or not 0<=s<=2**31-1 for s in strides):raise ValueError('Buffer strides need matching nonnegative integer element strides')
    strides=tuple(strides);alignment=size if alignment is None else alignment
    if type(alignment)is not int or alignment<size or alignment>256 or alignment&(alignment-1):raise ValueError('Buffer alignment must be a power of two from element size through 256 bytes')
    if access not in ('read','readwrite'):raise ValueError('Buffer access must be read or readwrite')
    result=BufferType(elem,shape,strides,alignment,access)
    if result.storage_elements*size>2**31-1:raise ValueError('Buffer storage byte span exceeds positive int32 offset domain')
    if access=='readwrite':
        span=1
        for stride,dim in sorted(zip(strides,shape)):
            if dim>1:
                if stride<span:raise ValueError('writable Buffer layout may overlap; use read access for broadcast views')
                span+=(dim-1)*stride
    return result


def resolve_annotation(ann,names):
    from openassembler.pyfront import compile as c
    if len(ann.args)!=2:raise c.NotSupported('Buffer(dtype, shape) takes two positional arguments',ann)
    elem=c._resolve_type(ann.args[0],names)
    def literal(node):
        if isinstance(node,(ast.Tuple,ast.List)):return tuple(literal(x) for x in node.elts)
        if isinstance(node,ast.Constant) and type(node.value)in (int,str):return node.value
        raise c.NotSupported('Buffer shape/stride/alignment/access must be literal constants',node)
    kwargs={}
    for keyword in ann.keywords:
        if keyword.arg not in ('strides','alignment','access') or keyword.arg in kwargs:raise c.NotSupported('unsupported or repeated Buffer option',keyword)
        kwargs[keyword.arg]=literal(keyword.value)
    try:return Buffer(elem,literal(ann.args[1]),**kwargs)
    except ValueError as error:raise c.NotSupported(str(error),ann) from None


def metadata(name,view):
    return {'name':name,'dtype':view.elem.ptx,'shape':list(view.shape),'strides':list(view.strides),'alignment':view.alignment,
            'access':view.access,'element_bytes':8 if view.elem.ptx.endswith('64') else 4,'storage_elements':view.storage_elements}


def access(emitter,sub,value=None,kind=None):
    from openassembler.pyfront import compile as c
    array=emitter.s.arrays[sub.value.id];view=array.view
    if value is not None and view.access=='read':raise c.NotSupported('cannot write a read-only Buffer',sub)
    nodes=sub.slice.elts if isinstance(sub.slice,ast.Tuple) else [sub.slice]
    if len(nodes)!=len(view.shape):raise c.NotSupported(f'Buffer requires {len(view.shape)} scalar indices',sub)
    indices=[]
    for node in nodes:
        register,index_kind=emitter._eval(node)
        if index_kind!='s32':raise c.NotSupported('Buffer indices must be int32 expressions',node)
        snapshot=emitter.s.new_reg('s32');emitter._mov(snapshot,register,'s32');indices.append(snapshot)
    result=None
    if value is None:
        result=emitter.s.new_reg(array.elem)
        zero,_=emitter._lit_as(0.0 if array.elem=='f32' else 0,array.elem,sub)
        emitter._mov(result,zero,array.elem)
    elif kind!=array.elem:raise c.NotSupported(f'storing {kind} into Buffer of {array.elem} requires an explicit conversion',sub)
    finish=emitter.s.new_label('buffer_checked')
    for index,dimension in zip(indices,view.shape):
        predicate=emitter.s.new_reg('pred')
        emitter.s.emit(f' setp.lt.s32 {predicate.name}, {index.name}, 0;')
        emitter.s.emit(f' @{predicate.name} bra {finish};')
        emitter.s.emit(f' setp.ge.s32 {predicate.name}, {index.name}, {dimension};')
        emitter.s.emit(f' @{predicate.name} bra {finish};')
    offset=emitter.s.new_reg('s32');emitter.s.emit(f' mov.s32 {offset.name}, 0;')
    for index,stride in zip(indices,view.strides):
        product=emitter.s.new_reg('s32')
        emitter.s.emit(f' mul.lo.s32 {product.name}, {index.name}, {stride};')
        emitter.s.emit(f' add.s32 {offset.name}, {offset.name}, {product.name};')
    address=emitter.s.new_reg('ptr');size=8 if array.elem.endswith('64') else 4
    emitter.s.emit(f' mul.wide.u32 {address.name}, {offset.name}, {size};')
    emitter.s.emit(f' add.u64 {address.name}, {array.base.name}, {address.name};')
    if value is None:
        loaded=emitter._load(array,f'[{address.name}]');emitter._mov(result,loaded,array.elem)
    else:emitter._store_at(array,f'[{address.name}]',value,kind,sub)
    emitter.s.emit(f'{finish}:')
    return result,array.elem


def contract_from_ptx(source,kernel):
    """Validate shaped-buffer metadata against the entry ABI and storage rules."""
    import json
    from openassembler.pyfront.types import Scalar
    marker='// openassembler-pyfront-buffer-v1 '
    records={}
    for line in source.splitlines():
        text=line.strip()
        if not text.startswith(marker.rstrip()):continue
        if not text.startswith(marker):raise ValueError('malformed checked Buffer metadata')
        try:record=json.loads(text[len(marker):])
        except ValueError:raise ValueError('malformed checked Buffer JSON') from None
        if not isinstance(record,dict) or set(record)!={'schema','kernel','buffers'} or record.get('schema')!=1 or type(record['schema'])is not int or not isinstance(record['kernel'],str) or not isinstance(record['buffers'],list):raise ValueError('invalid checked Buffer schema')
        if record['kernel'] in records:raise ValueError('duplicate checked Buffer entry')
        records[record['kernel']]=record
    result=records.get(kernel.name)
    if result is None:return None
    params={name:kind for kind,name in kernel.params};seen=set()
    for row in result['buffers']:
        keys={'name','dtype','shape','strides','alignment','access','element_bytes','storage_elements'}
        if not isinstance(row,dict) or set(row)!=keys or not isinstance(row['name'],str) or row['name'] in seen or params.get(row['name'])!='u64':raise ValueError('checked Buffer does not match PTX pointer ABI')
        seen.add(row['name'])
        if type(row['element_bytes'])is not int or row['element_bytes'] not in (4,8):raise ValueError('invalid checked Buffer element width')
        if row['dtype'] not in ('f32','s32','u32','s64','u64') or (8 if row['dtype'].endswith('64') else 4)!=row['element_bytes']:raise ValueError('checked Buffer dtype and width mismatch')
        view=Buffer(Scalar(row['dtype']),row['shape'],strides=row['strides'],alignment=row['alignment'],access=row['access'])
        if type(row['storage_elements'])is not int or row['storage_elements']!=view.storage_elements:raise ValueError('checked Buffer storage span mismatch')
    if not seen:raise ValueError('checked Buffer metadata has no buffers')
    return result

"""Typed GPU-scope relaxed 32-bit atomic primitives for kernel source."""
from __future__ import annotations
import ast

ATOMIC_FORMS={'atomic_add':('add',3),'atomic_exchange':('exch',3),'atomic_compare_exchange':('cas',4)}


def atomic(emitter,node,name):
    from openassembler.pyfront import compile as c
    operation,arity=ATOMIC_FORMS[name]
    if len(node.args)!=arity or node.keywords:raise c.NotSupported(f'{name} expects {arity} positional arguments',node)
    target,index,*arguments=node.args
    if not isinstance(target,ast.Name)or target.id not in emitter.s.arrays:raise c.NotSupported('atomic target must be an Array or Buffer parameter',target)
    array=emitter.s.arrays[target.id]
    if array.space!='global'or array.elem not in('s32','u32'):raise c.NotSupported('atomics require global i32/u32 elements',node)
    if operation!='add'and array.elem!='u32':raise c.NotSupported('exchange and compare_exchange require u32 bit values',node)
    nodes=index.elts if isinstance(index,ast.Tuple) else [index]
    if array.view is None and len(nodes)!=1:raise c.NotSupported('flat Array atomic index must be one int32 expression',index)
    indices=[]
    for expression in nodes:
        coordinate,kind=emitter._eval(expression)
        if kind!='s32':raise c.NotSupported('atomic indices must be int32 expressions',expression)
        snapshot=emitter.s.new_reg('s32');emitter._mov(snapshot,coordinate,'s32');indices.append(snapshot)
    values=[]
    for argument in arguments:
        literal=c._const_value(argument)
        value,kind=emitter._lit_as(literal,array.elem,argument)if literal is not None else emitter._eval(argument)
        if kind!=array.elem:raise c.NotSupported(f'atomic value type {kind} differs from target {array.elem}',argument)
        snapshot=emitter.s.new_reg(kind);emitter._mov(snapshot,value,kind);values.append(snapshot)
    def perform(address):
        old=emitter.s.new_reg(array.elem)
        dtype=array.elem if operation=='add'else'b32'
        emitter.s.emit(f' atom.relaxed.gpu.global.{operation}.{dtype} {old.name}, {address}, '+', '.join(value.name for value in values)+';')
        return old
    sub=ast.copy_location(ast.Subscript(value=target,slice=index,ctx=ast.Load()),node)
    if array.view is not None:
        from openassembler.pyfront.buffers import access
        return access(emitter,sub,atomic=perform,index_values=indices)
    address=emitter.s.new_reg('ptr')
    emitter.s.emit(f' mul.wide.u32 {address.name}, {indices[0].name}, 4;')
    emitter.s.emit(f' add.u64 {address.name}, {array.base.name}, {address.name};')
    return perform(f'[{address.name}]'),array.elem

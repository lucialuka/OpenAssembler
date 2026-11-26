"""Exact scalar PTX admission contracts, separate from instruction selection.

An accepted mnemonic is insufficient: modifiers, types, widths and operand
shapes must all describe a lowering the selector actually implements. Tensor
instructions retain their separate exact checks in backend.tensor.
"""
from functools import lru_cache
from itertools import product

from openassembler.ptx.parse import Addr, ImmF, ImmI, Var, Vec


class ContractError(ValueError):
    pass


@lru_cache(maxsize=1)
def scalar_forms():
    forms = {}
    def add(op, modifiers, arity):
        forms[(op, tuple(modifiers.split('.')) if modifiers else ())] = arity
    for t in ('b32','u32','s32','f32','b64','u64','s64','f64'):
        add('mov',t,2)
        add('ld','param.'+t,2)
    for t in ('s32','u32','s64','u64'):
        for op in ('add','sub'): add(op,t,3)
    for op in ('add','sub'): add(op,'sat.s32',3)
    for t in ('b32','b64'):
        for op in ('and','or','xor'): add(op,t,3)
    for op in ('and','or','xor'): add(op,'pred',3)
    for t in ('b32','pred'): add('not',t,2)
    add('mov','pred',2)
    for t in ('s32','u32'):
        for op in ('min','max'): add(op,t,3)
        for variant in ('lo','hi','wide'): add('mul',variant+'.'+t,3)
        add('mad','lo.'+t,4)
        add('neg',t,2)
    for t in ('b32','b64'): add('shl',t,3)
    for t in ('b32','u32','s32','b64','u64','s64'): add('shr',t,3)
    for dst,src in product(('s32','u32','s64','u64'), repeat=2): add('cvt',dst+'.'+src,2)
    for mode in ('global','to.global'):
        for t in ('u64','u32'): add('cvta',mode+'.'+t,2)
    for t in ('u32','s32','b32','f32'): add('selp',t,4)
    for op in ('add','sub','mul','mad','fma'):
        for rounding in ('','rn.'):
            add(op,rounding+'f32',4 if op in ('mad','fma') else 3)
    for op in ('neg','abs','min','max'): add(op,'f32',2 if op in ('neg','abs') else 3)
    for t in ('s32','u32','b32'):
        for cmp in ('eq','ne','lt','le','gt','ge','lo','ls','hi','hs'): add('setp',cmp+'.'+t,3)
    for t in ('s64','u64'):
        for cmp in ('eq','ne','lt','le','gt','ge'): add('setp',cmp+'.'+t,3)
    for cmp in ('lo','ls','hi','hs'): add('setp',cmp+'.u64',3)
    for cmp in ('eq','ne','lt','le','gt','ge','equ','neu','ltu','leu','gtu','geu','num','nan'):
        add('setp',cmp+'.f32',3)
        add('setp',cmp+'.ftz.f32',3)
        add('setp','ftz.'+cmp+'.f32',3)
    for t in ('b32','u32','s32','f32'):
        for width in ('','v2.','v4.'):
            for op in ('ld','st'): add(op,'shared.'+width+t,2)
        for cache in ('','wb.','cg.','cs.','wt.'):
            for width in ('','v2.','v4.'): add('st','global.'+cache+width+t,2)
    for t in ('b8','u8','s8','b16','u16','s16','b64','u64','s64','f64'):
        for cache in ('','wb.','cg.','cs.','wt.'): add('st','global.'+cache+t,2)
    for t in ('u8','s8','u16','s16','b32','u32','s32','f32','b64','u64','s64','f64'):
        for cache in ('','nc.','cg.','cs.','lu.','ca.'):
            add('ld','global.'+cache+t,2)
            if t in ('b32','u32','s32','f32'):
                for width in ('v2.','v4.'): add('ld','global.'+cache+width+t,2)
    for prefix in ('global.','relaxed.gpu.global.'):
        for dtype in ('u32','s32'):
            add('atom',prefix+'add.'+dtype,3)
            add('red',prefix+'add.'+dtype,2)
        add('atom',prefix+'exch.b32',3)
        add('atom',prefix+'cas.b32',4)
    for semantics in ('acquire','release','acq_rel'):
        prefix=semantics+'.gpu.global.'
        for dtype in ('u32','s32'):add('atom',prefix+'add.'+dtype,3)
        add('atom',prefix+'exch.b32',3)
        add('atom',prefix+'cas.b32',4)
    for prefix in ('shared.','relaxed.cta.shared.'):
        for dtype in ('u32','s32'):add('atom',prefix+'add.'+dtype,3)
        add('atom',prefix+'exch.b32',3)
    for dtype in ('b32','u32','s32','f32'):
        for op in ('ld','st'):add(op,'local.'+dtype,2)
    for op in ('ret','bra'):
        for m in ('','uni'): add(op,m,0 if op=='ret' else 1)
    add('bar','sync',1)
    add('bar','',1)
    for mode in ('bfly','idx','up','down'):add('shfl','sync.'+mode+'.b32',5)
    from .fragments import library
    for key in library():
        op, *mods = key.split('.')
        add(op,'.'.join(mods),3 if op in ('div','rem') else 2)
    return forms


def validate_scalar(ins):
    key = (ins.op, ins.mods)
    name = '.'.join((ins.op,*ins.mods))
    arity = scalar_forms().get(key)
    if arity is None:
        raise ContractError(f'unsupported PTX contract {name}: modifiers/types have no faithful scalar lowering')
    if len(ins.args) != arity:
        raise ContractError(f'{name}: expected {arity} operands, got {len(ins.args)}')
    a = ins.args
    if not a:
        return
    if ins.op == 'mov' and any(isinstance(x,Vec) for x in a):
        vectors = [x for x in a if isinstance(x,Vec)]
        if ins.mods != ('b64',) or len(vectors)!=1 or len(vectors[0].elems)!=2:
            raise ContractError(f'{name}: packed/unpacked moves require .b64 and two 32-bit registers')
    if ins.op in ('atom','red'):
        returning = ins.op == 'atom'
        address = a[1] if returning else a[0]
        sources = a[2:] if returning else a[1:]
        if (returning and not isinstance(a[0],Var) or not isinstance(address,Addr)
                or address.off % 4 or any(not isinstance(value,(Var,ImmI)) for value in sources)):
            raise ContractError(f'{name}: requires a register result, naturally aligned address and scalar integer sources')
    elif ins.op in ('ld','st'):
        address, value = (a[1], a[0]) if ins.op=='ld' else (a[0], a[1])
        if not isinstance(address, Addr):
            raise ContractError(f'{name}: memory operand must be an address')
        width = 4 if 'v4' in ins.mods else 2 if 'v2' in ins.mods else 1
        if width > 1:
            if not isinstance(value,Vec) or len(value.elems)!=width or any(
                    not isinstance(v,Var) for v in value.elems if v is not None):
                raise ContractError(f'{name}: requires a vector of {width} registers')
        elif isinstance(value,Vec):
            raise ContractError(f'{name}: scalar memory operation cannot use a vector')
    elif ins.op not in ('bar','shfl'):
        if not isinstance(a[0],(Var,Vec)):
            raise ContractError(f'{name}: destination must be a register')
        if ins.op != 'mov' and any(not isinstance(x,(Var,ImmI,ImmF)) for x in a[1:]):
            raise ContractError(f'{name}: requires scalar source operands')
    if ins.op in ('shl','shr') and isinstance(a[2],ImmI) and a[2].value < 0:
        raise ContractError(f'{name}: negative shift count')

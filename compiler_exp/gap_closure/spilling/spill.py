"""Candidate scalar virtual-value spilling into the validated private frame."""
from dataclasses import replace


def allocate_kernel(selector,ops):
    from openassembler.backend import ra
    limit=selector.k.directives.get('maxnreg',(252,))[0]
    if limit<16:raise ra.NotSupported('native register limit must be at least 16')
    max_reg=min(250,limit-2)
    original=list(ops)
    groups=ra._groups(original)
    starts,ends=ra._intervals(original)
    defined={value for op in original for value in ra._flat(op.defs)if isinstance(value,ra.V)}
    scalar=[value for value,group in groups.items()if len(group)==1 and value!=selector.r1 and value in defined]
    scalar.sort(key=lambda value:(-(ends[value]-starts[value]),value.n))
    spilled={}
    base=selector.local_frame.bytes
    prologue=next(op for op in original if op.defs==[selector.r1]and op.fmt.startswith('MOV'))
    adjustment=next((op for op in original if op.defs==[selector.r1]and op.fmt.startswith('IADD3')),None)
    next_virtual=max(v.n for op in original for v in ra._flat(op.defs+op.uses)if isinstance(v,ra.V))
    def rebuild():
        nonlocal next_virtual
        next_virtual=max(v.n for op in original for v in ra._flat(op.defs+op.uses)if isinstance(v,ra.V))
        result=[];frame=(base+len(spilled)*4+15)//16*16
        if frame>65536:raise ra.NotSupported('spill frame exceeds 65536 bytes')
        for op in original:
            if op is adjustment:continue
            if op is prologue:
                result.append(op)
                if frame:result.append(ra.Op(f'IADD3 {{d0}}, {{s0}}, -{frame:#x}, RZ',[selector.r1],[selector.r1],side=True))
                continue
            reads=set(ra._flat(op.uses))&spilled.keys()
            writes=set(ra._flat(op.defs))&spilled.keys()
            if op.guard is not None:reads|=writes
            replacements={}
            for value in sorted(reads|writes,key=lambda value:value.n):
                next_virtual+=1;replacements[value]=ra.V(next_virtual)
            for value in sorted(reads,key=lambda value:value.n):
                result.append(ra.Op(f'LDL {{d0}}, [{{s0}}+{spilled[value]:#x}]',[replacements[value]],[selector.r1],var_lat=True))
            def subst(value):
                if isinstance(value,tuple):return tuple(subst(x)for x in value)
                return replacements.get(value,value)
            result.append(replace(op,defs=[subst(x)for x in op.defs],uses=[subst(x)for x in op.uses]))
            for value in sorted(writes,key=lambda value:value.n):
                result.append(ra.Op(f'STL [{{s0}}+{spilled[value]:#x}], {{s1}}',[],[selector.r1,replacements[value]],side=True))
        return result,frame
    while True:
        try:
            phys=ra.allocate(ops,selector.r1,max_reg=max_reg)
            return ops,phys,dict(local_bytes=base if not spilled else frame,spill_slots=len(spilled),register_limit=limit)
        except ra.NotSupported as error:
            if str(error)!='out of registers':raise
            candidates=[value for value in scalar if value not in spilled]
            if not candidates or len(spilled)>=256:
                raise ra.NotSupported('register pressure exceeds scalar spill candidate domain; wide/predicate spilling remains unsupported')from None
            value=candidates[0];spilled[value]=base+len(spilled)*4
            ops,frame=rebuild()

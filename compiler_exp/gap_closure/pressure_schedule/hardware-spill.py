"""Bounded private-frame spilling for scalar, aligned and predicate values."""
from dataclasses import replace


def _allocate_kernel(selector,ops):
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
    # Preserve the proven scalar policy first. Only a failed scalar plan retries
    # with entire aligned groups as candidates; tuple members are never split.
    wide={group for group in groups.values()if len(group)in(2,4)and selector.r1 not in group and all(value in defined for value in group)}
    combined=[(value,)for value in scalar]+list(wide)
    combined.sort(key=lambda group:(-(max(ends[v]for v in group)-min(starts[v]for v in group)),-len(group),tuple(v.n for v in group)))
    predicate_span={}
    predicate_defined={value for op in original for value in op.defs if isinstance(value,ra.PV)}
    live=ra.cfg_liveness(original)if any(op.target for op in original)else None
    for index,op in enumerate(original):
        values=ra._defs(op)|set(ra._all_uses(op))
        if live is not None:values|=live[index]
        for value in values:
            if isinstance(value,ra.PV)and value in predicate_defined:
                first,last=predicate_span.get(value,(index,index))
                predicate_span[value]=(min(first,index),max(last,index))
    predicates=sorted(predicate_span,key=lambda value:(-(predicate_span[value][1]-predicate_span[value][0]),value.n))
    registers=[];predicate_slots=[];group_plan=False
    base=selector.local_frame.bytes
    prologue=next(op for op in original if op.defs==[selector.r1]and op.fmt.startswith('MOV'))
    adjustment=next((op for op in original if op.defs==[selector.r1]and op.fmt.startswith('IADD3')),None)
    max_virtual=max(v.n for op in original for v in ra._flat(op.defs+op.uses)if isinstance(v,ra.V))
    max_predicate=max((v.n for op in original for v in ra._defs(op)|set(ra._all_uses(op))if isinstance(v,ra.PV)),default=0)

    def rebuild():
        next_virtual=max_virtual;next_predicate=max_predicate
        spilled={value:base+4*index for index,value in enumerate(registers+predicate_slots)}
        frame=(base+len(spilled)*4+15)//16*16
        if frame>65536:raise ra.NotSupported('spill frame exceeds 65536 bytes')
        result=[]
        for op in original:
            if op is adjustment:continue
            if op is prologue:
                result.append(op)
                if frame:result.append(ra.Op(f'IADD3 {{d0}}, {{s0}}, -{frame:#x}, RZ',[selector.r1],[selector.r1],side=True))
                continue
            reads=set(ra._all_uses(op))&spilled.keys()
            writes=ra._defs(op)&spilled.keys()
            # A false guard preserves the old value. Reload/store unconditionally
            # so a temporary never becomes live through a loop's back edge.
            if op.guard is not None:reads|=writes
            replacements={}
            for value in sorted(reads|writes,key=lambda value:(isinstance(value,ra.PV),value.n)):
                if isinstance(value,ra.PV):
                    next_predicate+=1;replacements[value]=ra.PV(next_predicate)
                else:
                    next_virtual+=1;replacements[value]=ra.V(next_virtual)
            for value in sorted(reads,key=lambda value:(isinstance(value,ra.PV),value.n)):
                destination=replacements[value]
                if isinstance(value,ra.PV):
                    next_virtual+=1;temporary=ra.V(next_virtual)
                    result.append(ra.Op(f'LDL {{d0}}, [{{s0}}+{spilled[value]:#x}]',[temporary],[selector.r1],var_lat=True))
                    result.append(ra.Op('ISETP.EQ.U32.AND {p0}, PT, {s0}, RZ, PT',[destination],[temporary]))
                else:result.append(ra.Op(f'LDL {{d0}}, [{{s0}}+{spilled[value]:#x}]',[destination],[selector.r1],var_lat=True))
            def subst(value):
                if isinstance(value,tuple):return tuple(subst(x)for x in value)
                return replacements.get(value,value)
            result.append(replace(op,defs=[subst(x)for x in op.defs],uses=[subst(x)for x in op.uses],
                                  guard=subst(op.guard)if op.guard is not None else None))
            for value in sorted(writes,key=lambda value:(isinstance(value,ra.PV),value.n)):
                source=replacements[value]
                if isinstance(value,ra.PV):
                    next_virtual+=1;temporary=ra.V(next_virtual)
                    # The recovered SEL form permits an immediate second source.
                    # Slot convention is zero=true, one=false; restore compares 0.
                    result.append(ra.Op('SEL {d0}, RZ, 0x1, {q0}',[temporary],[(source,False)]))
                    source=temporary
                result.append(ra.Op(f'STL [{{s0}}+{spilled[value]:#x}], {{s1}}',[],[selector.r1,source],side=True))
        return result,frame

    while True:
        try:
            phys=ra.allocate(ops,selector.r1,max_reg=max_reg)
            ra.allocate_preds(ops)
            return ops,phys,dict(local_bytes=base if not registers and not predicate_slots else frame,
                                 spill_slots=len(registers)+len(predicate_slots),register_limit=limit)
        except ra.NotSupported as error:
            if str(error)=='out of predicate registers':
                candidates=[value for value in predicates if value not in predicate_slots]
                if not candidates or len(registers)+len(predicate_slots)>=256:
                    raise ra.NotSupported('predicate pressure exceeds bounded spill candidate domain')from None
                predicate_slots.append(candidates[0])
            elif str(error)=='out of registers':
                plans=combined if group_plan else[(value,)for value in scalar]
                candidates=[group for group in plans if not any(value in registers for value in group)
                            and len(registers)+len(predicate_slots)+len(group)<=256]
                if not candidates and not group_plan:
                    group_plan=True;registers=[]
                    continue
                if not candidates:raise ra.NotSupported('register pressure exceeds bounded scalar/group spill candidate domain')from None
                registers.extend(candidates[0])
            else:raise
            ops,frame=rebuild()


def allocate_kernel(selector,ops):
    from openassembler.backend import ra
    from openassembler.backend.pressure import schedule
    from collections import Counter
    def transfers(result):
        counts=Counter();block=0
        for op in result[0]:
            if op.label:block+=1
            opcode=op.fmt.split()[0].split('.')[0]if op.fmt else ''
            if opcode in('LDL','STL'):counts[block,opcode]+=1
            if op.target:block+=1
        return counts
    failure=None
    try:
        baseline=_allocate_kernel(selector,ops)
        if not baseline[2]['spill_slots']:return baseline
    except ra.NotSupported as error:
        if not str(error).startswith(('register pressure exceeds','predicate pressure exceeds','spill frame exceeds')):raise
        failure=error
    ordered=schedule(ops)
    if any(a is not b for a,b in zip(ops,ordered)):
        try:
            candidate=_allocate_kernel(selector,ordered)
            if failure is not None:return candidate
            before,after=transfers(baseline),transfers(candidate)
            if(candidate[2]['spill_slots']<baseline[2]['spill_slots']and candidate[2]['local_bytes']<=baseline[2]['local_bytes']
               and all(count<=before[key]for key,count in after.items())):return candidate
        except ra.NotSupported:pass
    if failure is not None:raise failure
    return baseline

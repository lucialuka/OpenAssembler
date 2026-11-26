"""Pressure-oriented virtual scheduling with explicit register dependencies."""
from collections import Counter

# Unknown operations and all stores/fences terminate a region. Loads retain
# their source order; arithmetic cannot cross any store or synchronization.
MOVABLE={'MOV','IMAD','IADD3','LOP3','SHF','SGXT','LEA','PRMT','BMSK','FLO','POPC','BREV',
         'IABS','IMNMX','FADD','FMUL','FFMA','FSEL','FMNMX','MUFU','F2F','F2I','I2F','I2I',
         'ISETP','FSETP','PLOP3','SEL','LDG','LDS','LDL','LDC'}
LOADS={'LDG','LDS','LDL','LDC'}


def schedule(ops):
    from openassembler.backend import ra
    import re
    live=ra.cfg_liveness(ops)
    groups=ra._groups(ops)
    def cost(values):
        expanded=set()
        for value in values:
            expanded.update(groups.get(value,(value,)))
        return sum(4 if isinstance(value,ra.PV)else 1 for value in expanded)
    def region(items,live_out):
        if len(items)<3:return items
        reads=[ra._all_uses(op)|(ra._defs(op)if op.guard is not None else set())for op in items]
        writes=[ra._defs(op)for op in items]
        succ=[set()for _ in items];last_write={};readers={};last_load=None
        for index,op in enumerate(items):
            for value in reads[index]:
                if value in last_write:succ[last_write[value]].add(index)
            for value in writes[index]:
                if value in last_write:succ[last_write[value]].add(index)
                for reader in readers.get(value,()):succ[reader].add(index)
            opcode=op.fmt.split()[0].split('.')[0]
            if opcode in LOADS:
                if last_load is not None:succ[last_load].add(index)
                last_load=index
            for value in reads[index]:readers.setdefault(value,set()).add(index)
            for value in writes[index]:last_write[value]=index;readers[value]=set()
        pending=[0]*len(items);priority=[0]*len(items)
        for index in range(len(items)-1,-1,-1):
            priority[index]=max((priority[target]+1 for target in succ[index]),default=0)
            for target in succ[index]:pending[target]+=1
        remaining=Counter(value for values in reads for value in values)
        current=set(live_out)
        for index in range(len(items)-1,-1,-1):
            if items[index].guard is None:current-=writes[index]
            current|=reads[index]
        ready=[i for i,n in enumerate(pending)if n==0];result=[]
        while ready:
            def after(index):
                dying={value for value in reads[index]if remaining[value]==1 and value not in live_out}
                born={value for value in writes[index]if remaining[value]>int(value in reads[index])or value in live_out}
                return(current-dying)|born
            index=min(ready,key=lambda index:(cost(after(index)),-priority[index],index))
            ready.remove(index);current=after(index);result.append(items[index])
            remaining.subtract(reads[index])
            for target in sorted(succ[index]):
                pending[target]-=1
                if pending[target]==0:ready.append(target)
        assert len(result)==len(items),'virtual dependency cycle'
        return result
    result=[];chunk=[]
    for index,op in enumerate(ops):
        opcode=op.fmt.split()[0].split('.')[0]if op.fmt else ''
        movable=(not op.label and not op.target and not op.side and opcode in MOVABLE
                 and not re.search(r'\b(?:R|P|UR)\d+\b',op.fmt))
        if movable:chunk.append(op)
        else:
            if chunk:result.extend(region(chunk,live[index-1]));chunk=[]
            result.append(op)
    if chunk:result.extend(region(chunk,live[-1]))
    return result

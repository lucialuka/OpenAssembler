"""Source-level direct PTX function specialization with lexical parameter slots.

This lowers calls to the ordinary entry-local backend. It defines no native
call stack or linker ABI. All generated names, storage and returns are explicit.
"""
from dataclasses import dataclass
import hashlib
import re

_SYMBOL=r'[%A-Za-z_$][\w$\.]*'
_TYPE=re.compile(r'\.(b32|u32|s32|f32|b64|u64|s64|f64)\b')

@dataclass
class Slot:
    name: str
    dtype: str
    formal: bool=False


def inline_module(source, parser):
    """Return entry-only source and exact definition/call inventories."""
    stripped=parser._strip_comments(source);masked=parser._mask_strings(stripped)
    functions={};spans=[];position=0
    pattern=re.compile(r'(?<![\w$.])\.func\b')
    def balanced(start,opening='(',closing=')'):
        if masked[start]!=opening:raise SyntaxError('function signature requires '+opening)
        depth=1;end=start+1
        while end<len(masked)and depth:
            depth+=(masked[end]==opening)-(masked[end]==closing);end+=1
        if depth:raise SyntaxError('unbalanced device function signature')
        return end
    while match:=pattern.search(masked,position):
        begin=match.end()
        while begin<len(masked)and masked[begin].isspace():begin+=1
        returns=''
        if begin<len(masked)and masked[begin]=='(':
            end=balanced(begin);returns=stripped[begin+1:end-1];begin=end
        name_match=re.match(r'\s*('+_SYMBOL+r')\s*\(',masked[begin:])
        if not name_match:raise SyntaxError('unsupported device function signature')
        name=name_match[1];start_params=begin+name_match.end()-1;end_params=balanced(start_params)
        end=end_params
        while end<len(masked)and masked[end].isspace():end+=1
        if end>=len(masked)or masked[end]not in'{;':raise SyntaxError('unsupported device function header '+name)
        definition=masked[end]=='{';finish=parser._match_brace(masked,end)+1 if definition else end+1
        linkage=re.search(r'(?:\.(?:visible|weak|extern)\s+)+$',masked[:match.start()])
        row=dict(name=name,returns=returns,params=stripped[start_params+1:end_params-1],body=stripped[end+1:finish-1]if definition else None,line=stripped.count('\n',0,end)+1,external=bool(linkage and '.extern'in linkage[0]))
        if name in functions:
            previous=functions[name]
            if (previous['external']or row['external'])and(previous['body']is not None or definition):raise SyntaxError('device definition conflicts with external declaration '+name)
            if previous['body']is not None and definition:raise SyntaxError('duplicate device function '+name)
            if [t for t,n in parser._parameters(previous['params'],name)]!=[t for t,n in parser._parameters(row['params'],name)]or [t for t,n in parser._parameters(previous['returns'],name)]!=[t for t,n in parser._parameters(row['returns'],name)]:raise SyntaxError('conflicting device function declaration '+name)
            if definition:functions[name]=row
        else:functions[name]=row
        # Strip linkage tokens too; retain newlines for source diagnostics.
        start=match.start();prefix=re.search(r'(?:\.(?:visible|weak|extern)\s+)+$',masked[:start])
        if prefix:start=prefix.start()
        spans.append((start,finish));position=finish
    if not functions:return source,{},{},{}
    reserved=stripped;prefix='__oa_inline_'
    while prefix in reserved:prefix+='x'
    serial=0;calls={};budget=0;all_slots=set()
    def fresh():
        nonlocal serial
        serial+=1;return prefix+str(serial)+'_'
    def params(text,name):
        values=parser._parameters(text,'device '+name)
        if len(values)>32:raise SyntaxError('device function has more than 32 scalar slots')
        return values
    def tokens(body):
        current='';depth=0;line=1;start=1
        for ch in body:
            if not current.strip()and not ch.isspace():start=line
            if ch=='\n':line+=1
            if ch=='{'and not current.strip():yield('{',line);continue
            if ch=='}'and depth==0:
                if current.strip():yield(current.strip(),start)
                current='';yield('}',line);continue
            if ch=='{':depth+=1
            elif ch=='}':depth-=1
            if ch==';'and depth==0:
                yield(current.strip(),start);current=''
            else:current+=ch
        if current.strip():yield(current.strip(),start)
    def lower(body, formal, output_names, stack, line_offset, is_entry=False):
        nonlocal budget
        body=re.sub(r'(?<![\w$.])\.loc[ \t]+[0-9]+[ \t]+[0-9]+[ \t]+[0-9]+',lambda m:' '*len(m[0]),body)
        env=[dict(formal)];lines=[];returns=[];end_label=fresh()+'return';invocation=fresh()
        # Labels are function-scoped, distinct from lexical call argument blocks.
        labels={m[1]:invocation+m[1].replace('.','_')for m in re.finditer(r'(?m)(?:^|[;{}])\s*([A-Za-z_$][\w$.]*):',body)}
        def lookup(name):
            for scope in reversed(env):
                if name in scope:return scope[name]
            return None
        def rename(text):
            def sub(match):
                name=match[0];binding=lookup(name)
                return binding.name if isinstance(binding,Slot)else binding if binding is not None else labels.get(name,name)
            return re.sub(_SYMBOL,sub,text)
        def emit(text,line):
            nonlocal budget
            budget+=1
            if budget>100000:raise SyntaxError('device inlining exceeds 100000 statements')
            lines.append((text,line))
        for statement,relative in tokens(body):
            line=line_offset+relative-1
            if statement=='{':env.append({});continue
            if statement=='}':
                if len(env)==1:raise SyntaxError('unbalanced function lexical scope')
                env.pop();continue
            while label:=re.match(r'^([A-Za-z_$][\w$.]*):\s*(.*)$',statement,re.S):
                emit(labels[label[1]]+':',line);statement=label[2]
            if not statement:continue
            if statement.startswith('.reg '):
                dtype=re.search(r'\.(pred|[busf](?:8|16|32|64))\b',statement)
                if not dtype:raise SyntaxError('unsupported device register declaration')
                names=statement[dtype.end():].strip().split(',')
                for name in names:
                    name=name.strip();ranged=re.fullmatch(r'('+_SYMBOL+r')<(\d+)>',name)
                    if ranged and int(ranged[2])>65536:raise SyntaxError('device register declaration too large')
                    expanded=[ranged[1]+str(i)for i in range(int(ranged[2]))]if ranged else[name]
                    if len(expanded)>65536:raise SyntaxError('device register declaration too large')
                    for symbol in expanded:
                        if not re.fullmatch(_SYMBOL,symbol):raise SyntaxError('invalid device register name')
                        if symbol in env[-1]:raise SyntaxError('duplicate lexical symbol '+symbol)
                        env[-1][symbol]='%'+fresh()+symbol.lstrip('%').replace('.','_')
                continue
            if statement.startswith('.param '):
                values=params(statement,'call scope')
                if len(values)!=1:raise SyntaxError('scalar call slot declaration required')
                dtype,name=values[0]
                if name in env[-1]:raise SyntaxError('duplicate call parameter slot '+name)
                env[-1][name]=Slot('%'+fresh()+'slot',dtype)
                all_slots.add(env[-1][name].name)
                continue
            storage=re.match(r'\.(local|shared|global|const)\b',statement)
            if storage:
                if storage[1]!='local'and not is_entry:raise SyntaxError('inlined device storage requires per-call static local objects')
                declaration=parser._VAR_DECL.fullmatch(statement)
                if not declaration:raise SyntaxError('unsupported device storage declaration')
                name=declaration[4]
                if name in env[-1]:raise SyntaxError('duplicate storage symbol '+name)
                env[-1][name]=fresh()+name;emit(rename(statement),line);continue
            if statement.startswith('.'):
                if statement.split()[0]not in('.loc','.pragma','.file'):raise SyntaxError('unsupported device body directive '+statement.split()[0])
                continue
            guard='';core=statement
            if core.startswith('@'):guard,core=core.split(None,1)
            call=re.match(r'call(?:\.(uni))?\s+(.+)$',core,re.S)
            if call:
                operands=parser._split_top(call[2]);out=[]
                if operands and operands[0].startswith('('):out=parser._split_top(operands.pop(0)[1:-1])
                if len(operands)not in(1,2)or not re.fullmatch(_SYMBOL,operands[0]):raise SyntaxError('only direct scalar device calls are supported')
                target=operands[0];inputs=parser._split_top(operands[1][1:-1])if len(operands)==2 and operands[1].startswith('(')and operands[1].endswith(')')else[]
                if len(operands)==2 and not(operands[1].startswith('(')and operands[1].endswith(')')):raise SyntaxError('call arguments require a parameter tuple')
                if target in stack:raise SyntaxError('recursive device call: '+' -> '.join((*stack,target)))
                if len(stack)>=32:raise SyntaxError('device call expansion exceeds depth 32')
                if target not in functions or functions[target]['body']is None:raise SyntaxError('unresolved device function '+target)
                definition=functions[target];arguments=params(definition['params'],target);results=params(definition['returns'],target)
                if len(results)>1:raise SyntaxError('device function ABI permits at most one scalar return slot')
                if len(inputs)!=len(arguments)or len(out)!=len(results):raise SyntaxError('device call arity mismatch for '+target)
                bindings={};skip=fresh()+'call_end'
                if guard:emit(rename('@'+guard[2:]if guard.startswith('@!')else'@!'+guard[1:])+' bra '+skip,line)
                outputs=[]
                for names,signature,result in((inputs,arguments,False),(out,results,True)):
                    for actual,(dtype,name)in zip(names,signature):
                        slot=lookup(actual)
                        if not isinstance(slot,Slot)or slot.dtype[-2:]!=dtype[-2:]:raise SyntaxError('device call requires matching-width scalar parameter slots: '+actual)
                        local=Slot('%'+fresh()+'formal',dtype,not result);bindings[name]=local
                        all_slots.add(local.name)
                        if result:outputs.append((slot,local))
                        else:emit('mov.'+dtype+' '+local.name+', '+slot.name,line)
                expanded=lower(definition['body'],bindings,{name for _,name in results},(*stack,target),definition['line'])
                for text,origin in expanded:emit(text,origin)
                for destination,local in outputs:emit('mov.'+local.dtype+' '+destination.name+', '+local.name,line)
                if guard:emit(skip+':',line)
                calls[target]=calls.get(target,0)+1
                continue
            if core.split()[0].split('.')[0]=='call':raise SyntaxError('only direct scalar call or call.uni forms are supported')
            memory=re.match(r'(ld|st)\.param\.(b32|u32|s32|f32|b64|u64|s64|f64)\s+(.+)$',core,re.S)
            if memory:
                operands=parser._split_top(memory[3]);address=parser.parse_operand(operands[1 if memory[1]=='ld'else 0])if len(operands)==2 else None
                slot=lookup(address.base)if isinstance(address,parser.Addr)and address.off==0 else None
                if slot is not None:
                    if not isinstance(slot,Slot)or slot.dtype[-2:]!=memory[2][-2:]:raise SyntaxError('device parameter access requires matching scalar width and zero offset')
                    if memory[1]=='st':
                        if slot.formal:raise SyntaxError('device input parameter is read-only')
                        lhs,rhs=slot.name,rename(operands[1])
                    else:lhs,rhs=rename(operands[0]),slot.name
                    emit((rename(guard)+' 'if guard else'')+'mov.'+memory[2]+' '+lhs+', '+rhs,line)
                    continue
                if not is_entry:raise SyntaxError('unknown device parameter or nonzero offset')
            if core=='ret'and not is_entry:
                emit((rename(guard)+' 'if guard else'')+'bra '+end_label,line);returns.append(len(lines)-1);continue
            if core.split()[0]in('exit','trap')and not is_entry:raise SyntaxError('device function must return to its caller')
            # Parameters cannot escape or be used as ordinary address/register values.
            for symbol in re.findall(_SYMBOL,core):
                if isinstance(lookup(symbol),Slot):raise SyntaxError('device parameter slot used outside ld/st.param or call')
            emit((rename(guard)+' 'if guard else'')+rename(core),line)
        if len(env)!=1:raise SyntaxError('unbalanced function lexical scope')
        if not is_entry:
            emit(end_label+':',line_offset)
            # Definite return-slot initialization on every reachable return and
            # rejection of fallthrough protect against silent stale returns.
            ir=parser._parse_body('inline',[],render(lines))
            return_index=ir.labels[end_label];required={formal[name].name for name in output_names}
            incoming={0:set()};work=[0]
            while work:
                index=work.pop();defined=set(incoming[index])
                if index==return_index:
                    if not required<=defined:raise SyntaxError('device return slot is not definitely initialized')
                    continue
                ins=ir.body[index]
                if ins.op=='mov'and ins.guard is None and isinstance(ins.args[0],parser.Var):defined.add(ins.args[0].name)
                edges=[ir.labels[ins.args[0].name]]if ins.op=='bra'else[index+1]
                if ins.op=='bra'and ins.guard is not None:edges.append(index+1)
                for edge in edges:
                    if edge==return_index and not(ins.op=='bra'and ins.args[0].name==end_label and (ins.guard is None or edge!=index+1)):
                        raise SyntaxError('device function falls through without return')
                    if edge>return_index:raise SyntaxError('device branch escapes function')
                    state=defined if edge not in incoming else incoming[edge]&defined
                    if edge not in incoming or state!=incoming[edge]:incoming[edge]=state;work.append(edge)
        return lines
    def render(lines):
        return '\n'.join(text if text.endswith(':')else text+';'for text,line in lines)
    def validate_slots(lines):
        ir=parser._parse_body('inline_slots',[],render(lines))
        if not ir.body:return
        incoming={0:set()};work=[0]
        while work:
            index=work.pop();ins=ir.body[index];defined=set(incoming[index]);reads=[];write=None
            if ins.op=='mov'and isinstance(ins.args[0],parser.Var):
                write=ins.args[0].name;reads=ins.args[1:]
            else:reads=ins.args
            for operand in reads:
                if isinstance(operand,parser.Var)and operand.name in all_slots and operand.name not in defined:
                    raise SyntaxError('device parameter slot read before definite initialization')
            if write is not None and ins.guard is None:defined.add(write)
            if ins.op in('ret','exit'):continue
            if ins.op=='bra':
                if not isinstance(ins.args[0],parser.Var)or ins.args[0].name not in ir.labels:raise SyntaxError('unresolved device branch target')
                edges=[ir.labels[ins.args[0].name]]
                if ins.guard is not None:edges.append(index+1)
            else:edges=[index+1]
            for edge in edges:
                if edge>=len(ir.body):continue
                state=defined if edge not in incoming else incoming[edge]&defined
                if edge not in incoming or state!=incoming[edge]:incoming[edge]=state;work.append(edge)
    entries=list(parser.entry_definitions(stripped));replacements=[];inventories={};origins={}
    for name,raw_params,header,body,start,end in entries:
        if not re.search(r'\bcall(?:\.[a-z_]+)*\s',body):continue
        before=dict(calls)
        lines=lower(body,{},set(),(),stripped.count('\n',0,end-len(body)-1)+1,True)
        validate_slots(lines)
        flat=[]
        for text,line in lines:flat.extend([line]*len((text if text.endswith(':')else text+';').splitlines()))
        probe=parser._parse_body(name,[],render(lines))
        origins[name]=[flat[instruction.line-1]for instruction in probe.body]
        replacements.append((start,end,'.entry '+name+'('+raw_params+') '+header+' {\n'+render(lines)+'\n}'))
        inventories[name]={target:calls[target]-before.get(target,0)for target in calls if calls[target]>before.get(target,0)}
    for start,end in spans:replacements.append((start,end,'\n'*stripped.count('\n',start,end)))
    result=stripped
    for start,end,replacement in sorted(replacements,reverse=True):result=result[:start]+replacement+result[end:]
    inventory={name:dict(parameters=params(row['params'],name),returns=params(row['returns'],name),defined=row['body']is not None,body_sha256=hashlib.sha256(row['body'].encode()).hexdigest()if row['body']is not None else None)for name,row in functions.items()}
    return result,inventory,inventories,origins

"""Deterministic scalar PTX source-library resolution before direct inlining."""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re


class LinkError(ValueError):
    pass


@dataclass
class Function:
    name: str
    linkage: str
    params: list
    returns: list
    body: str | None
    text: str
    start: int
    end: int
    name_start: int
    name_end: int


def digest(data):
    return hashlib.sha256(data).hexdigest()


def definitions(source):
    from . import parse as parser
    stripped=parser._strip_comments(source);masked=parser._mask_strings(stripped)
    rows=[];cursor=0
    def balanced(start):
        depth=1;end=start+1
        while end<len(masked)and depth:
            depth+=(masked[end]=='(')-(masked[end]==')');end+=1
        if depth:raise LinkError('unbalanced device-library function signature')
        return end
    while match:=re.search(r'(?<![\w$.])\.func\b',masked[cursor:]):
        pos=cursor+match.start();index=cursor+match.end()
        while index<len(masked)and masked[index].isspace():index+=1
        returns=''
        if index<len(masked)and masked[index]=='(':
            end=balanced(index);returns=stripped[index+1:end-1];index=end
        name=re.match(r'\s*([A-Za-z_$][\w$.]*)\s*\(',masked[index:])
        if name is None:raise LinkError('unsupported device-library function signature')
        name_start=index+name.start(1);name_end=index+name.end(1);begin=index+name.end()-1;end=balanced(begin)
        index=end
        while index<len(masked)and masked[index].isspace():index+=1
        if index>=len(masked)or masked[index]not in'{;':raise LinkError('unsupported device-library function header '+name[1])
        finish=parser._match_brace(masked,index)+1 if masked[index]=='{'else index+1
        prefix=re.search(r'(?:\.(?:visible|weak|extern)\s+)+$',masked[:pos]);start=prefix.start()if prefix else pos
        qualifiers=re.findall(r'\.(visible|weak|extern)',prefix[0])if prefix else[]
        if len(qualifiers)>1:raise LinkError('conflicting function linkage for '+name[1])
        linkage=qualifiers[0]if qualifiers else'private'
        body=stripped[index+1:finish-1]if masked[index]=='{'else None
        if body is not None and linkage=='extern':raise LinkError('external device function cannot define a body')
        if '.ptr'in stripped[begin+1:end-1]or '.ptr'in returns:raise LinkError('device-library function slots do not admit pointer annotations')
        try:args=parser._parameters(stripped[begin+1:end-1],name[1]);out=parser._parameters(returns,name[1])
        except SyntaxError as error:raise LinkError(str(error))from None
        if len(args)>32 or len(out)>1:raise LinkError('device libraries require at most 32 scalar inputs and one scalar return')
        rows.append(Function(name[1],linkage,args,out,body,stripped[start:finish],start,finish,name_start-start,name_end-start))
        cursor=finish
    return stripped,rows


def _signature(function):
    return tuple(t for t,n in function.params),tuple(t for t,n in function.returns)


def _calls(text):
    from .parse import _split_top,_mask_strings
    found=[]
    for match in re.finditer(r'(?<![\w$.])call(?:\.uni)?\s+([^;]+);',_mask_strings(text)):
        operands=_split_top(match[1])
        if operands and operands[0].startswith('('):operands.pop(0)
        if not operands or not re.fullmatch(r'[A-Za-z_$][\w$.]*',operands[0]):raise LinkError('device library resolution requires direct named calls')
        raw=match[1];offset=0
        if raw.lstrip().startswith('('):
            close=raw.index(')');offset=close+1
            while offset<len(raw)and(raw[offset].isspace()or raw[offset]==','):offset+=1
        else:offset=len(raw)-len(raw.lstrip())
        found.append((match.start(1)+offset,match.start(1)+offset+len(operands[0]),operands[0]))
    return found


def resolve(units):
    """Resolve [(identity, PTX)] with unit0 supplying the executable entries."""
    from .parse import entry_definitions,_strip_comments,_mask_strings
    if not units or len(units)>65:raise LinkError('device link requires one root and at most 64 libraries')
    parsed=[];exported={};entries={};prefix='__oa_link_'
    while any(prefix in text for identity,text in units):prefix+='x'
    versions=[];target=None
    for index,(identity,text)in enumerate(units):
        source,functions=definitions(text);masked=_mask_strings(source)
        version=re.findall(r'(?m)^\s*\.version\s+(\d+)\.(\d+)\s*$',masked)
        targets=re.findall(r'(?m)^\s*\.target\s+([^\n]+)',masked)
        sizes=re.findall(r'(?m)^\s*\.address_size\s+(\d+)\s*$',masked)
        if len(version)!=1 or len(targets)!=1 or sizes!=['64']:raise LinkError('device-link inputs require one version/target and address_size 64')
        if targets[0].strip()!='sm_80':raise LinkError('device-library target must be exactly sm_80')
        versions.append(tuple(map(int,version[0])))
        local={};spans=[]
        for function in functions:
            local.setdefault(function.name,[]).append(function);spans.append((function.start,function.end))
            if function.body is not None and function.linkage in('visible','weak'):exported.setdefault(function.name,[]).append((index,function))
        raw_entries=list(entry_definitions(source))
        if index and raw_entries:raise LinkError('device libraries must contain functions, not kernel entries')
        for name,params,header,body,start,end in raw_entries:
            if name in entries:raise LinkError('duplicate linked entry '+name)
            entries[name]=index
            qualifier=re.search(r'\.(?:visible|weak)\s*$',source[:start])
            spans.append((qualifier.start()if qualifier else start,end))
        residue=source
        for start,end in sorted(spans,reverse=True):residue=residue[:start]+' '*(end-start)+residue[end:]
        residue=re.sub(r'(?m)^\s*\.(?:version|target|address_size|file|loc)\b[^\n]*','',residue)
        if residue.strip():raise LinkError('device libraries do not yet relocate module data or unsupported module directives: '+residue.strip()[:80])
        for name,rows in local.items():
            if len({ _signature(row)for row in rows})!=1:raise LinkError('conflicting local function signatures: '+name)
            definitions_=[row for row in rows if row.body is not None]
            if len(definitions_)>1:raise LinkError('duplicate device definition in one module: '+name)
            if definitions_ and any(row.linkage=='extern'for row in rows):raise LinkError('device definition conflicts with local extern declaration: '+name)
            if any(row.linkage=='private'for row in definitions_)and any(row.linkage in('visible','weak')for row in rows):raise LinkError('private/exported visibility conflict: '+name)
        parsed.append(dict(identity=identity,source=source,functions=functions,local=local,entries=raw_entries))
    selected={};origins={}
    for name,choices in exported.items():
        if name in entries:raise LinkError('device function conflicts with entry '+name)
        if len({_signature(function)for index,function in choices})!=1:raise LinkError('conflicting exported function signatures: '+name)
        strong=[choice for choice in choices if choice[1].linkage=='visible']
        if len(strong)>1:raise LinkError('duplicate strong device symbol '+name)
        if strong:choice=strong[0]
        else:
            if len({function.body for index,function in choices})!=1:raise LinkError('ambiguous nonidentical weak device definitions: '+name)
            choice=choices[0]
        selected[name]=choice;origins[name]=dict(unit=choice[0],linkage=choice[1].linkage,body_sha256=digest(choice[1].body.encode()))
    bindings=[]
    for index,unit in enumerate(parsed):
        table={}
        for name,rows in unit['local'].items():
            private=next((f for f in rows if f.body is not None and f.linkage=='private'),None)
            if private:table[name]=(index,private,prefix+str(index)+'_'+str(len(table)))
            elif name in selected:
                owner,definition=selected[name]
                if _signature(rows[0])!=_signature(definition):raise LinkError('external device ABI mismatch: '+name)
                table[name]=(owner,definition,name)
            elif any(f.body is None for f in rows):table[name]=None
        bindings.append(table)
    retained={};visiting=[];used_edges=[]
    def visit(index,name):
        binding=bindings[index].get(name)
        if binding is None:raise LinkError('unresolved external device function '+name)
        owner,function,linked_name=binding
        key=(owner,function.name)
        if key in visiting:raise LinkError('recursive linked device call: '+' -> '.join(n for i,n in[*visiting,key]))
        if key in retained:return linked_name
        if len(visiting)>=32:raise LinkError('linked device call depth exceeds 32')
        visiting.append(key);text=function.text
        edits=[(function.name_start,function.name_end,linked_name)]
        for start,end,target in _calls(text):
            destination=visit(owner,target);edits.append((start,end,destination));used_edges.append(dict(unit=owner,caller=linked_name,callee=destination))
        for start,end,value in sorted(edits,reverse=True):text=text[:start]+value+text[end:]
        # Only selected reachable definitions are emitted. No external prototype
        # can conflict with a now-defined body in the normalized closed module.
        retained[key]=text;visiting.pop();return linked_name
    root=parsed[0];entry_texts=[]
    for name,params,header,body,start,end in root['entries']:
        text=root['source'][start:end];edits=[]
        for a,b,target_ in _calls(text):
            destination=visit(0,target_);edits.append((a,b,destination));used_edges.append(dict(unit=0,caller=name,callee=destination))
        for a,b,value in sorted(edits,reverse=True):text=text[:a]+value+text[b:]
        # Entry scanner starts at .entry; retain its source linkage explicitly.
        preceding=root['source'][:start];visibility=re.search(r'\.(visible|weak)\s*$',preceding)
        if visibility:text='.'+visibility[1]+' '+text
        entry_texts.append(text)
    if not entry_texts:raise LinkError('device link root has no kernel entries')
    maximum=max(versions);header=f'.version {maximum[0]}.{maximum[1]}\n.target sm_80\n.address_size 64\n'
    # Debug file identifiers are module-local. Rewrite each unit's file and loc
    # indices together, preserving file records in the linked module.
    file_lines=[];file_maps=[];next_file=1
    for unit in parsed:
        mapping={}
        for match in re.finditer(r'(?m)^\s*\.file\s+(\d+)\s+([^\n]+)',unit['source']):
            identifier=int(match[1])
            if identifier in mapping:raise LinkError('duplicate module debug file id')
            mapping[identifier]=next_file;file_lines.append('.file '+str(next_file)+' '+match[2]);next_file+=1
        file_maps.append(mapping)
    def debug(text,owner):
        def replace(match):
            old=int(match[1])
            if old not in file_maps[owner]:raise LinkError('unresolved module debug file index')
            return '.loc '+str(file_maps[owner][old])
        return re.sub(r'\.loc\s+(\d+)\b',replace,text)
    merged=header+'\n'.join(file_lines)+'\n'+'\n'.join(debug(text,owner)for(owner,name),text in retained.items())+'\n'+'\n'.join(debug(text,0)for text in entry_texts)+'\n'
    contract=dict(schema=1,strategy='scalar-ptx-source-link',units=[dict(identity=identity,ptx_sha256=digest(text.encode()))for identity,text in units],
                  exports=origins,retained_functions=[dict(unit=index,name=name)for index,name in retained],edges=used_edges,
                  version=list(maximum),target='sm_80',address_size=64,linked_ptx_sha256=digest(merged.encode()))
    return merged,contract


def link_paths(source, libraries, *, arch, include_dirs=(), nvcc_flags=(), compile_commands=None, library_options=()):
    """Compile explicit source libraries, then resolve their PTX definitions."""
    from ..project import ProjectError
    from .. import cppfront,oracle
    from ..compilation_database import load,dependencies
    units=[('entry',source)];records=[];paths=[Path(p).resolve()for p in libraries]
    if len(set(paths))!=len(paths):raise LinkError('device library paths must be unique')
    if len(paths)>64:raise LinkError('at most 64 device libraries are supported')
    if library_options and len(library_options)!=len(paths):raise LinkError('device library transport options mismatch')
    for index,path in enumerate(paths):
        if not path.is_file():raise LinkError('device library does not exist: '+str(path))
        before=path.read_bytes();options=library_options[index]if library_options else{}
        if options and options.get('source_sha256')!=digest(before):raise LinkError('transported device library source changed')
        includes=[Path(p)for p in options.get('include_dirs',include_dirs)];database=options.get('compile_commands',compile_commands)
        command=None
        if path.suffix.lower()!='.ptx':
            if database is not None:command=load(database).get(path)
            if command is None and(path.parent/'compile_commands.json').is_file():command=load(path.parent/'compile_commands.json').get(path)
        flags=list(nvcc_flags);kwargs={};dependency={str(path):digest(before)}
        if path.suffix.lower()=='.ptx':text=before.decode();language='ptx';settings=None
        elif path.suffix.lower()in('.cu','.cuh','.cpp','.cc','.cxx'):
            language='cuda'
            if command:
                flags=[*command.flags,*flags];kwargs=dict(working_directory=command.directory,source_argument=command.source_argument)
            flags=relocatable_flags(flags)
            if command:before_dependencies=dependencies(command,include_dirs=includes,extra_flags=relocatable_flags(nvcc_flags))
            else:
                from ..compilation_database import CompileCommand
                synthetic=CompileCommand(path,str(path),path.parent,(),('nvcc',str(path)),path,digest(before),{})
                before_dependencies=dependencies(synthetic,include_dirs=includes,extra_flags=flags)
            text,names=cppfront.compile_to_ptx_module(path,arch=arch,include_dirs=includes,extra_flags=flags,allow_device_library=True,**kwargs)
            if names:raise LinkError('device libraries must not define kernel entries: '+str(path))
            if command:dependency=dependencies(command,include_dirs=includes,extra_flags=relocatable_flags(nvcc_flags))
            else:
                # Ask the actual compiler for dependencies even without a database.
                from ..compilation_database import CompileCommand
                synthetic=CompileCommand(path,str(path),path.parent,(),('nvcc',str(path)),path, digest(before),{})
                dependency=dependencies(synthetic,include_dirs=includes,extra_flags=flags)
            if before_dependencies!=dependency:raise LinkError('device-library dependencies changed during compilation: '+str(path))
            settings=command.record()if command else dict(source=str(path),device_flags=flags)
            settings['effective_device_flags']=list(flags)
        else:raise LinkError('device library requires PTX or CUDA source: '+str(path))
        if path.read_bytes()!=before:raise LinkError('device library changed during compilation: '+str(path))
        identity=options.get('source_origin',str(path));units.append((identity,text))
        records.append(dict(source=identity,source_sha256=digest(before),language=language,dependencies=dependency,compile_contract=settings))
    merged,contract=resolve(units);contract['libraries']=records
    if any(record['language']=='cuda'for record in records):contract['toolchain']=oracle.run_tool('nvcc','--version').stdout
    return merged,contract


def relocatable_flags(flags):
    """Library input requests separate device compilation; refuse a conflicting opt-out."""
    flags=list(flags);present=False
    for index,flag in enumerate(flags):
        key,separator,value=flag.partition('=')
        if key not in('-rdc','--relocatable-device-code'):continue
        setting=value if separator else flags[index+1]if index+1<len(flags)else''
        if setting!='true':raise LinkError('device libraries require relocatable device code; conflicting '+flag)
        present=True
    return flags if present else[*flags,'-rdc=true']

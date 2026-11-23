"""Representative cross-stage scalar contract ledger; never a coverage percentage.

Each declared form gets its own retained PTX fixture and stage outcomes. A passed
encoding check applies only to that fixture's operand domain; CPU semantics,
packaged modules, toolkit and hardware remain explicitly separate evidence.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path
import platform
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
HERE=Path(__file__).resolve().parent
FIXTURES={}

from openassembler.backend.contracts import scalar_forms,validate_scalar
from openassembler.backend import ra
from openassembler.isa.encode import Encoder
from openassembler.ptx.parse import parse_module

HEADER='.version 8.7\n.target sm_80\n.address_size 64\n'
PARAMS='.param .u64 input, .param .u64 output, .param .u32 a, .param .u32 b, .param .u32 c, .param .u64 aa, .param .u64 bb, .param .f32 fa, .param .f32 fb, .param .f32 fc, .param .f64 da, .param .f64 db'
SETUP=['ld.param.u64 %input,[input];','ld.param.u64 %output,[output];',
       'ld.param.u32 %r0,[a];','ld.param.u32 %r1,[b];','ld.param.u32 %r2,[c];',
       'ld.param.u64 %d0,[aa];','ld.param.u64 %d1,[bb];',
       'ld.param.f32 %f0,[fa];','ld.param.f32 %f1,[fb];','ld.param.f32 %f2,[fc];',
       'ld.param.f64 %fd0,[da];','ld.param.f64 %fd1,[db];',
       'setp.eq.u32 %p0,%r0,%r1;','setp.ne.u32 %p1,%r0,%r1;']


def registers(kind):
    if kind=='pred':return ('%p0','%p1','%p0'),'%pout'
    if kind=='f32':return ('%f0','%f1','%f2'),'%fout'
    if kind=='f64':return ('%fd0','%fd1','%fd0'),'%dout'
    if kind.endswith('64'):return ('%d0','%d1','%d0'),'%dout'
    return ('%r0','%r1','%r2'),'%rout'


def fixture(op,mods,arity,register_backed=True):
    form='.'.join((op,*mods));t=mods[-1] if mods else 'u32'
    sources,dest=registers(t);prefix=list(SETUP);suffix=[]
    if register_backed:
        prefix += [f'ld.global.u32 %r{i},[%input+{i*4}];' for i in range(3)]
        prefix += [f'ld.global.f32 %f{i},[%input+{16+i*4}];' for i in range(3)]
        prefix += [f'ld.global.u64 %d{i},[%input+{32+i*8}];' for i in range(2)]
        prefix += [f'ld.global.f64 %fd{i},[%input+{48+i*8}];' for i in range(2)]
        prefix += ['setp.eq.u32 %p0,%r0,%r1;','setp.ne.u32 %p1,%r0,%r1;']
    domain={'probe':'single unpredicated operation, result kept live by global store',
            'operand_storage':'global-load registers' if register_backed else 'parameter constant bank',
            'source_type':t,'numerical_values':'not executed; parameter bytes unconstrained by encoding checks',
            'architecture':'sm_80','aliasing':'distinct input/output pointer bindings',
            'alignment':'16-byte global input/output and shared storage','launch':'not executed'}
    if op in ('ld','st'):
        width=4 if 'v4' in mods else 2 if 'v2' in mods else 1
        vector_prefix='%w' if t.endswith('64') else '%v'
        value='{'+','.join(vector_prefix+str(i) for i in range(width))+'}' if width>1 else dest
        if op=='st':
            value='{'+','.join(sources[i%len(sources)] for i in range(width))+'}' if width>1 else sources[0]
        space=next(x for x in ('param','global','shared') if x in mods)
        address=('[input]' if space=='param' and t.endswith('64') else
                 '[a]' if space=='param' and t not in ('f32','f64') else
                 '[fa]' if space=='param' else '[storage]' if space=='shared' else '[%input]')
        if space=='param':
            # Give ld.param exactly the parameter type it names.
            address='[probe_value]'
        if space=='shared' and op=='ld':prefix+=['st.shared.v4.u32 [storage],{%r0,%r1,%r2,%r0};','bar.sync 0;']
        target=f'{form} {value},{address};' if op=='ld' else f'{form} {address},{value};'
        domain.update({'memory_space':space,'element_bits':int(t[1:]),'vector_width':width,
                       'access_bytes':int(t[1:])//8*width,'address_offset':0})
        if op=='ld':
            if width>1 and t.endswith('64'):
                for i in range(width):
                    suffix += [f'mov.b64 {{%low,%high}},%w{i};',f'st.global.v2.u32 [%output+{i*8}],{{%low,%high}};']
            elif width>1:suffix += [f'st.global.u32 [%output+{i*4}],%v{i};' for i in range(width)]
            elif t.endswith('64'):suffix+=['mov.b64 {%low,%high},'+dest+';','st.global.v2.u32 [%output],{%low,%high};']
            else:suffix+=['st.global.b32 [%output],'+dest+';']
    elif op in ('ret','bra','bar'):
        target=form+(' after;' if op=='bra' else ' 0;' if op=='bar' else ';')
        if op=='bra':suffix+=['after:']
        domain['probe']='unpredicated control/synchronization form; no numerical operation'
        if op=='bar':domain['barrier_id']=0
    elif op=='shfl':
        target=f'{form} %rout,%r0,1,31,0xffffffff;'
        suffix+=['st.global.u32 [%output],%rout;']
        domain.update({'participation':'full warp','clamp':31,'membermask':'0xffffffff','lane_delta':1})
    else:
        if op=='setp':dest='%pout'
        if op=='cvt':
            dst_type,src_type=mods[-2:];sources,_=registers(src_type);_,dest=registers(dst_type)
            domain.update({'source_type':src_type,'destination_type':dst_type})
        if op=='mul' and 'wide' in mods:dest='%dout'
        args=[dest,*sources[:arity-1]]
        if op=='selp':args=[dest,sources[0],sources[1],'%p0']
        if op in ('shl','shr'):
            args=[dest,sources[0],'7'];domain['shift_count']=7
        target=form+' '+','.join(args)+';'
        out_type=(mods[-2] if op=='cvt' else 'u64' if op=='mul' and 'wide' in mods else t)
        if op=='setp' or out_type=='pred':suffix+=['selp.u32 %rout,1,0,%pout;','st.global.u32 [%output],%rout;']
        elif out_type.endswith('64'):suffix+=['mov.b64 {%low,%high},'+dest+';','st.global.v2.u32 [%output],{%low,%high};']
        else:suffix+=['st.global.b32 [%output],'+dest+';']
    params=PARAMS+(', .param .'+t+' probe_value' if op=='ld' and 'param' in mods else '')
    lines=[HEADER.rstrip(),f'.visible .entry probe({params}) {{',
           '.reg .b32 %r<8>,%v<4>,%rout,%low,%high;',
           '.reg .b64 %input,%output,%d<4>,%dout,%fd<4>,%w<4>;',
           '.reg .f32 %f<4>,%fout; .reg .pred %p<4>,%pout;',
           '.shared .align 16 .b8 storage[64];',*prefix]
    source='\n'.join(lines)+'\n'+target+'\n'+'\n'.join(suffix+['ret;','}'])+'\n'
    target_line=len('\n'.join(lines).splitlines())+1
    return source,target_line,domain


def stage_result(action):
    try:
        value=action()
        return {'status':'passed'},value
    except Exception as exc:
        return {'status':'rejected','exception':type(exc).__name__,'reason':str(exc)},None


def evaluate(op,mods,arity,encoder,*,declared=True,mutator=None,identifier=None,register_backed=True):
    form='.'.join((op,*mods));ident=identifier or form.replace('.','_')
    ident += '_register' if register_backed else '_parameter'
    try:source,target_line,domain=fixture(op,mods,arity,register_backed)
    except Exception as exc:
        return {'form':form,'declared':declared,'stages':{'fixture':{'status':'not-run','reason':str(exc)}}}
    if mutator:source,target_line,domain=mutator(source,target_line,domain)
    FIXTURES[ident]=source
    row={'form':form,'declared':declared,'arity':arity,'fixture_id':ident,
         'fixture_artifact':'compiler_exp/gap_closure/contracts/fixtures.json',
         'fixture_sha256':hashlib.sha256(source.encode()).hexdigest(),'target_line':target_line,
         'domain':domain,'stages':{}}
    stages=row['stages']
    stages['parse'],module=stage_result(lambda:parse_module(source))
    if module is not None:
        kernel=module.kernels['probe']
        ins=next(i for i in kernel.body if i.line==target_line)
        stages['admission'],_=stage_result(lambda:validate_scalar(ins))
        # Selection has its own whole-fixture admission too. A rejection there
        # is retained even when the target itself passed admission.
        if stages['admission']['status']=='passed':
            selector=ra.Selector(kernel)
            stages['selection'],ops=stage_result(selector.run)
            if ops is not None:
                ops=ra.coalesce(ra.dce(ops))
                stages['register_allocation'],phys=stage_result(lambda:ra.allocate(ops,selector.r1))
                if phys is not None:
                    stages['predicate_allocation_and_lowering'],lowered=stage_result(lambda:ra.lower(ops,phys))
                    if lowered is not None:
                        stages['register_allocation']['registers']=max(phys.values())+1
                        stages['scheduling'],compiled=stage_result(lambda:ra.compile_kernel(kernel))
                        if compiled is not None:
                            code,stats=compiled
                            stages['encoding'],words=stage_result(lambda:ra.encode_code(encoder,code,stats['labels']))
                            if words is not None:
                                stages['encoding'].update({'instruction_count':len(words),
                                    'machine_words_sha256':hashlib.sha256(b''.join(w.to_bytes(16,'little') for w in words)).hexdigest()})
    preceding='fixture or preceding stage did not pass'
    for stage in ('parse','admission','selection','register_allocation','predicate_allocation_and_lowering','scheduling','encoding'):
        stages.setdefault(stage,{'status':'not-run','reason':preceding})
    for stage in ('independent_cpu_semantics','toolkit','packaging','hardware','performance'):
        stages[stage]={'status':'not-run','reason':'this representative ledger does not execute '+stage.replace('_',' ')+' gates'}
    return row


def main():
    encoder=Encoder('sm_80');forms=scalar_forms()
    rows=[evaluate(op,mods,arity,encoder) for (op,mods),arity in sorted(forms.items())]
    for row,((op,mods),arity) in zip(rows,sorted(forms.items())):
        row['parameter_backed_variant']=evaluate(op,mods,arity,encoder,register_backed=False)
    negatives=[]
    for op,mods in [('st',('global','v4','u32')),('st',('global','u8')),('st',('global','u16')),('st',('global','u64')),
                    ('ld',('global','v2','u64')),('ld',('global','volatile','u32')),('st',('global','release','u32')),
                    ('ld',('shared','u8')),('ld',('shared','u64')),('add',('cc','u32')),('add',('rz','f32'))]:
        negatives.append(evaluate(op,mods,2 if op in ('ld','st') else 3,encoder,declared=False))
    def shift_variant(value):
        def mutate(source,line,domain):
            lines=source.splitlines();lines[line-1]=lines[line-1].replace(',7;',','+value+';')
            return '\n'.join(lines)+'\n',line,{**domain,'shift_count':value,'purpose':'operand-domain boundary control'}
        return mutate
    for op,mods in [('shl',('b32',)),('shr',('u32',)),('shl',('b64',)),('shr',('u64',))]:
        for value in ('%r1','-1','32','63'):
            negatives.append(evaluate(op,mods,3,encoder,mutator=shift_variant(value),identifier='_'.join((op,*mods,'count',value.replace('%','reg').replace('-','minus')))))
    memory=[]
    for op in ('ld','st'):
        for space in ('global','shared'):
            for kind in ('u8','s8','u16','s16','b32','u32','s32','f32','b64','u64','s64','f64'):
                for width in (1,2,4):
                    mods=(space,)+(('v'+str(width),) if width>1 else ())+ (kind,)
                    memory.append(evaluate(op,mods,2,encoder,declared=(op,mods) in forms))
    counts={stage:dict(Counter(row['stages'][stage]['status'] for row in rows))
            for stage in rows[0]['stages']}
    package=ROOT/'openassembler'
    inputs=[package/'backend/contracts.py',package/'backend/ra.py',package/'backend/compile.py',
            package/'backend/fragments.py',package/'backend/ptxas_fragments.json',
            package/'ptx/parse.py',package/'isa/encode.py',package/'isa/word.py',Path(__file__),
            ROOT/'corpus/sm_80/pairs.jsonl',*(ROOT/'specs/sm_80').glob('*.json')]
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(inputs)}
    result={'schema':1,'target':'sm_80','python':platform.python_version(),'declared_forms':len(forms),
            'policy':'one representative fixture per exact declaration; accepted is not fully supported or hardware proven',
            'scope':'parse/admission/selection/register allocation/scheduling/encoding only',
            'source_hashes':hashes,'counts':counts,'forms':rows,'boundary_and_negative_controls':negatives,
            'memory_width_matrix':memory,
            'fixture_artifact':'compiler_exp/gap_closure/contracts/fixtures.json'}
    (HERE/'fixtures.json').write_text(json.dumps(FIXTURES,indent=2)+'\n')
    result['fixtures_sha256']=hashlib.sha256((HERE/'fixtures.json').read_bytes()).hexdigest()
    (HERE/'ledger.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'declared_forms':len(forms),'counts':counts,'controls':len(negatives),'memory_matrix':len(memory)},indent=2))
    assert len(rows)==len(forms) and len({r['form'] for r in rows})==len(forms)
    assert all(r['stages']['admission']['status']=='passed' for r in rows)
    assert all(r['stages']['admission']['status']=='rejected' for r in negatives[:11])
    assert all(r['fixture_id'] in FIXTURES for r in rows+negatives+memory)


if __name__=='__main__':main()

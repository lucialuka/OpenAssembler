"""Recover typed PTX constant expression behavior and failure boundaries."""
import hashlib
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT))
from openassembler import oracle


def cases():
    groups={
        'integer': ['1+2*3','(1+2)*3','8/3','-8/3','8/-3','-8/-3','-8%3','8%-3','-8%-3',
            '1<<63','1<<64','1<<65','-1>>63','-1>>64','-1>>65','1<<-1','1>>-1',
            '0xffffffffffffffff+1','9223372036854775807+1','-9223372036854775808/-1',
            '0xffffffffffffffff/2','(.s64)0xffffffffffffffff/2','(.u64)-1/2',
            '0123','0b1010','+123','42U','0x1234U','(~0)>>63','(-2&-1)>0',
            '(-2|-1)>0','(-2^-1)>0','~0>0','(-8%3)>0','(-8%3)<0',
            '1<2','-1<1U','-1<1','1==1','1!=1','!0','!!3','1&&0','1||0',
            '(1<2)?3:4','0?3:4','0?1:0xffffffffffffffff','(0?1:0xffffffffffffffff)>0',
            '8-3-2','16/4/2','2+3<<1','1|2&4','1^3&2','1<<2+1','8>>1+1',
            '1||0&&0','0?1:0?2:3','1?0?2:3:4','(.s64)(-1U)<0','(.u64)(-1)<0',
            '18446744073709551615','18446744073709551616','0x10000000000000000',
            '1/0','0/0','1%0','0&&1/0','1||1/0','1?3:1/0','0?1/0:3'],
        'floating': ['(-1.5)','1.5+0.5','1.5+0.5*4.0','(1.5+0.5)*4.0',
            '1.5-0.5-0.5','8.0/4.0/2.0','1.0/3.0','((1.0e20+1.0)-1.0e20)',
            '1.0e308*2.0','1.0e-307/2.0','1.0e-307/1.0e308','1.0/0.0','0.0/0.0',
            '(-0.0)','-(-0.0)','(-0.0)+(-0.0)','0.0-0.0','-0.0*2.0',
            '0d3ff0000000000000+1.0','(0d7ff0000000000000+1.0)',
            '(0d7ff0000000000000-0d7ff0000000000000)',
            '(0d7ff8000000000001+1.0)','(0dfff8000000000001+1.0)',
            '1?1.5:2.5','(1 ? (1.5) : (2.5))','(1==1)?1.5:2.5',
            '1.5<2.0','1.5==1.5','1.5!=1.5','0d7ff8000000000000!=1.0',
            '0d7ff8000000000000<1.0','1.5+1','1+1.5','(.s64)1.5','0f3f800000+1.0',
            '(1?1.5:2.5)+1.0','1?1.5:1','(1.5<2.0)?1.5:2.5'],
    }
    for kind,expressions in groups.items():
        for index,expression in enumerate(expressions):
            # .b64 accepts either f64 raw representations or integer values.
            yield dict(name=f'{kind}_{index}',kind=kind,expression=expression)


def source(name,expression):
    return f'''.version 8.7
.target sm_80
.address_size 64
.visible .entry {name}(.param .u64 output){{
.reg .u64 %out;
.reg .b64 %value;
ld.param.u64 %out,[output];
mov.b64 %value,({expression});
st.global.b64 [%out],%value;ret;
}}
'''


if __name__=='__main__':
    folder=HERE/'oracle';folder.mkdir(exist_ok=True);records=[]
    for case in cases():
        name=case['name'];text=source(name,case['expression']);path=folder/(name+'.ptx');path.write_text(text);binary=folder/(name+'.cubin')
        result=oracle.run_tool('ptxas','-arch=sm_80','-o',oracle.tool_path(binary),oracle.tool_path(path),check=False)
        raw=result.stdout+result.stderr;(folder/(name+'.tool.log')).write_text(raw)
        row=dict(case,accepted=result.returncode==0,returncode=result.returncode,
                 source_sha256=hashlib.sha256(text.encode()).hexdigest(),tool_log_sha256=hashlib.sha256(raw.encode()).hexdigest())
        if row['accepted']:
            listing=oracle.disasm_text(binary);(folder/(name+'.sass')).write_text(listing)
            row.update(cubin_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),sass_sha256=hashlib.sha256(listing.encode()).hexdigest())
        records.append(row);print(row,flush=True)
    result=dict(schema=1,scope='Typed integer/floating constant-expression toolkit acceptance and retained executable artifacts; CPU/GPU execution are independent gates',
                results=records,ptxas=oracle.run_tool('ptxas','--version').stdout,
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (HERE/'oracle-probe.json').write_text(json.dumps(result,indent=2)+'\n')

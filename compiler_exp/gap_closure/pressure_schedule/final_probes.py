"""Final source admission covers the maximum frame and guarded early return."""
from probes import HEADER,SENTINEL,source,expected as integer_expected
import resource_probes


def cases():
    yield from resource_probes.cases()
    name='early_exit96';text=source(name,96,32)
    marker='xor.b32 %tmp,%r48,49;'
    assert text.count(marker)==1
    text=text.replace(marker,'@%even ret;\n'+marker)
    yield dict(name=name,count=96,limit=32,width=1,predicates=False,local=False,loop=False,output_words=1,
               domain='guarded_early_exit',source=text,may_spill=True)


def expected(data,case,n,lanes):
    result=integer_expected(data,case,n,lanes)
    if case['name']=='early_exit96':
        for lane in range(0,n,2):result[lane]=SENTINEL
    return result

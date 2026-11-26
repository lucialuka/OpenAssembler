"""A valid maximum local frame previously rejected by added spill slots."""
from probes import HEADER,source,expected


def cases():
    name='recover_max_frame'
    text=source(name,48,32,1,False,True,True).replace('preserve[32]','preserve[65536]')
    yield dict(name=name,count=48,limit=32,width=1,predicates=False,local=True,loop=True,output_words=1,
               domain='resource_recovery',source=text)

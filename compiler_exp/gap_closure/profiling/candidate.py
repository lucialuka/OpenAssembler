"""Independent pre-integration accounting model for repeated kernel observations."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics


def summarize(observations):
    groups=defaultdict(list)
    for iteration,entry,contract,duration in observations:
        identity=hashlib.sha256(json.dumps(contract,sort_keys=True).encode()).hexdigest()
        groups[(entry,identity)].append((iteration,duration))
    return [{'entry':name,'contract_sha256':identity,'launches':len(rows),
             'iterations':sorted({i for i,_ in rows}),
             'median_us':statistics.median(value for _,value in rows),
             'min_us':min(value for _,value in rows),'max_us':max(value for _,value in rows)}
            for (name,identity),rows in sorted(groups.items())]


if __name__=='__main__':
    observations=[(i,'oa_shared',{'shape':[7]},t) for i,t in [(1,1.0),(1,3.0),(2,5.0),(2,7.0),(3,9.0),(3,11.0)]]
    observations += [(i,'oa_other',{'shape':[3]},t) for i,t in [(1,4.0),(2,2.0),(3,3.0)]]
    rows=summarize(observations)
    assert rows[0]['launches']==3 and rows[0]['median_us']==3
    assert rows[1]['launches']==6 and rows[1]['iterations']==[1,2,3] and rows[1]['median_us']==6
    report={'schema':1,'scope':'synthetic pre-integration accounting, not measured GPU timing','observations':observations,'results':rows,'passed':True,
            'gap':'G48 single-invocation audits do not establish repeated per-contract distributions',
            'solution':'aggregate only independently audited kernel activities keyed by executable/launch/shape identity; retain per-iteration samples and distinguish profiler-free timings'}
    path=Path(__file__).with_name('candidate-results.json')
    with path.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    print('candidate accounting passed')

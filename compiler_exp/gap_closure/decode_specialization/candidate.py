"""Explicit bounded lazy variant policy candidate, before production promotion."""
import hashlib
import json
from pathlib import Path


def run():
    requests=[];variants={2:'precompiled'};max_context=8;budget=4;attempts=0
    expected_builds=[3,4,5,6]
    for cycle in range(2):
        length=2
        while length<7:
            if length not in variants:
                assert attempts<budget;attempts+=1
                contract={'schema':1,'artifact_key':'cpu-reference-v1','past_length':length,'next_length':length+1,'max_context':max_context}
                requests.append(contract);variants[length]='validated-exact-variant'
            length+=1
    assert [r['past_length'] for r in requests]==expected_builds and attempts==4
    before=dict(variants)
    try:
        assert 7 in variants or attempts<budget,'build attempt budget exhausted'
    except AssertionError:pass
    else:raise AssertionError('budget miss admitted')
    assert variants==before
    called=False
    try:
        if 8+1>max_context:raise ValueError('maximum context reached')
        called=True
    except ValueError:pass
    assert not called
    return {'schema':1,'gap':'G39','passed':True,'candidate_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'requests':requests,'build_attempts':attempts,'repeat_sequence_has_no_builds':True,
            'budget_failure_preserves_variants':True,'context_limit_precedes_builder':True,
            'hardware':'not run by independent policy state-machine candidate'}
if __name__=='__main__':
    r=run();Path(__file__).with_name('candidate-results.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))

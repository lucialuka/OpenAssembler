"""Audit retained archives and summarize frozen policy replay; no GPU work."""
from pathlib import Path
import hashlib
import json
import statistics
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from openassembler.gemm_tune import _verify_archive, choose_policy
from openassembler.gemm_dispatch import TABLE


def main(directory):
    directory = Path(directory)
    protocol = json.loads((directory / 'protocol.json').read_text())
    frozen = json.loads((directory / 'frozen-selection.json').read_text())
    installed = json.loads(TABLE.read_text())
    cases = []
    for case in protocol['cases']:
        plan = case['plan']
        key = 'x'.join(str(plan[d]) for d in ('m','n','k'))
        policy = frozen[key]
        matches = [e for e in installed['entries'] if all(e['contract'].get(d) == plan[d] for d in ('m','n','k','dtype'))]
        incumbent = matches[0]['policy'] if len(matches) == 1 else 'baseline'
        sessions = []
        for session in range(2):
            path = directory / f'{key}-session{session}.json'
            result = json.loads(path.read_text())
            _verify_archive(path, result)
            nominal = choose_policy(result)
            if session == 0 and nominal != policy:
                raise ValueError('frozen selection differs from first-session rule')
            medians = [r['median_ms'] for r in result['timings']]
            ratios = lambda reference, candidate: [r[reference] / r[candidate] for r in medians]
            all_policies = {name: {'baseline_speedup': ratios('baseline',name),
                                   'incumbent_speedup': ratios(incumbent,name),
                                   'native_resources': candidate['artifacts']['native']['resources'],
                                   'native_instruction_counts': candidate['artifacts']['native']['counts']}
                            for name,candidate in result['candidates'].items()}
            sessions.append({'session':session,'record':path.name,'record_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                             'archive':result['artifact_archive'],'device':result['device'],
                             'compiler_identity':result['compiler_identity'], 'policies':len(result['candidates']),
                             'correctness_checks':sum(len(c['checks']) for c in result['candidates'].values()),
                             'nominal_selected':nominal,'frozen_vs_baseline':ratios('baseline',policy),
                             'frozen_vs_incumbent':ratios(incumbent,policy),'all_policies':all_policies})
        cases.append({'shape':key,'frozen':policy,'incumbent':incumbent,'sessions':sessions,
                      'heldout_at_least_2pct_over_incumbent':min(sessions[1]['frozen_vs_incumbent'])>=1.02})
    summary = {'schema':1,'scope':protocol['scope'],'cases':cases,'promotion':'none',
               'limits':['Two sequential sessions may reuse a worker or GPU',
                         'No counters or ptxas performance comparison',
                         'Resident kernels only; no packing/host/backward/optimizer timing',
                         'Final compiler continuity audit separately establishes unchanged retained policy instructions']}
    (directory/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps([{'shape':c['shape'],'frozen':c['frozen'],'incumbent':c['incumbent'],
                      'heldout_vs_baseline':c['sessions'][1]['frozen_vs_baseline'],
                      'heldout_vs_incumbent':c['sessions'][1]['frozen_vs_incumbent'],
                      'policies':c['sessions'][1]['policies']} for c in cases],indent=2))


if __name__ == '__main__':
    main(sys.argv[1])

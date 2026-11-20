"""Promote only evidence admitted by the unchanged production auditor.

The default is read-only. --install appends admitted exact contracts; it never
replaces an incumbent and never weakens timing, provenance or binary gates.
"""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))

from openassembler.gemm_dispatch import TABLE, compiler_identity
from openassembler.gemm_promotion import promote
from openassembler.project import write_atomic


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('campaign',type=Path)
    parser.add_argument('--install',action='store_true')
    args=parser.parse_args()
    decisions=[]
    for key in ('128x512x1024','256x768x1024'):
        value=promote(args.campaign/f'{key}-selection.json',args.campaign/f'{key}-heldout.json',
                      args.campaign/f'{key}-frozen.json')
        if not value['production_promoted'] or value['status']!='promoted':
            raise ValueError(f'{key}: candidate not admitted: {value["reasons"]}')
        decisions.append(value)
    path=args.campaign/'promotion-decisions.json'
    payload=(json.dumps({'schema':1,'decisions':decisions},indent=2)+'\n').encode()
    write_atomic(path,payload)
    if args.install:
        table=json.loads(TABLE.read_text())
        if table['compiler_identity']!=compiler_identity():
            raise ValueError('existing table continuity must be established first')
        for decision in decisions:
            if any(e['contract']==decision['entry']['contract'] for e in table['entries']):
                raise ValueError('refusing to replace an existing exact contract')
        # Existing entries retain all original evidence and continuity records.
        table['entries'].extend(d['entry'] for d in decisions)
        table['incremental_promotions']=table.get('incremental_promotions',[])+[
            {'decisions':str(path),'sha256':hashlib.sha256(payload).hexdigest(),
             'contracts_added':len(decisions),'installed_utc':datetime.now(timezone.utc).isoformat()}]
        table['campaign_counts']={'promoted':sum(e['policy']!='baseline' for e in table['entries']),
                                  'measured-baseline':sum(e['policy']=='baseline' for e in table['entries'])}
        write_atomic(TABLE,json.dumps(table,indent=2)+'\n')
    print(path,'admitted',len(decisions),'installed',args.install)


if __name__=='__main__':main()

"""Compare entry-header approaches before production promotion; CPU only."""
import json
from datetime import datetime, timezone
from pathlib import Path
import re
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from openassembler.ptx.parse import parse_module

CASES = {
    'visible': ('.visible .entry a() { ret; }', ['a']),
    'bare': ('.entry b() { ret; }', ['b']),
    'weak': ('.weak .entry c() { ret; }', ['c']),
    'bounds': ('.entry d() .maxntid 128, 1, 1 { ret; }', ['d']),
    'mixed': ('.visible .entry a() { ret; } .entry b() { ret; }', ['a','b']),
    'comment': ('// .entry fake() { ret; }\n.entry real() { ret; }', ['real']),
    'quoted': ('.file 1 "source.entry fake() { x; }"\n.entry real() { ret; }', ['real']),
    'dollar': ('.visible .entry $kernel$1() { ret; }', ['$kernel$1']),
    'prototype': ('.extern .entry proto(); .visible .entry real() { ret; }', ['real']),
}

def regex_extension(text):
    text = re.sub(r'/\*.*?\*/|//[^\n]*', '', text, flags=re.S)
    return re.findall(r'(?:\.(?:visible|weak)\s+)?\.entry\s+([\w$]+)\s*\([^)]*\)[^{;]*\{', text)

if __name__ == '__main__':
    rows=[]
    for name,(text,expected) in CASES.items():
        for method,fn in [('current_parser',lambda t:list(parse_module(t).kernels)),('regex_extension',regex_extension)]:
            try: actual=fn(text);error=None
            except Exception as exc: actual=None;error=str(exc)
            rows.append(dict(case=name,method=method,expected=expected,actual=actual,error=error,passed=actual==expected))
    target=Path(__file__).with_name(sys.argv[1] if len(sys.argv)>1 else 'parser-candidates-replay-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '.json')
    if target.exists():
        raise ValueError('refusing to overwrite retained evidence')
    target.write_text(json.dumps(rows,indent=2)+'\n')
    print(target, sum(r['passed'] for r in rows), '/',len(rows))

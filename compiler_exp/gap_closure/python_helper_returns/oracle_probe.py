"""Actual toolkit check of isolated normalized helper-return source, not original integration."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;sys.path[:0]=[str(ROOT),str(HERE)]
from openassembler.pyfront import compile_file
from openassembler import oracle
if __name__=='__main__':
 text,names=compile_file(HERE/'normalized.py');source=HERE/'candidate.ptx';source.write_text(text);binary=HERE/'candidate.cubin'
 result=oracle.run_tool('ptxas','-arch=sm_80','-o',oracle.tool_path(binary),oracle.tool_path(source),check=False);(HERE/'oracle-tool.log').write_text(result.stdout+result.stderr);assert result.returncode==0,result.stderr
 listing=oracle.disasm_text(binary);(HERE/'candidate.sass').write_text(listing)
 files={name:hashlib.sha256((HERE/name).read_bytes()).hexdigest()for name in('normalized.py','candidate.ptx','candidate.cubin','candidate.sass','oracle-tool.log')}
 (HERE/'oracle-probe.json').write_text(json.dumps(dict(schema=1,passed=True,kernels=names,artifact_hashes=files,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),scope='Actual toolkit acceptance/disassembly of three normalized helper-return kernels; original source/diagnostic/return-path/ownership/control integration and native hardware remain unvalidated'),indent=2)+'\n');print('Accepted three normalized kernels; five source/tool artifacts retained')

"""Actual toolkit check of the explicitly normalized static-value hypothesis."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT))
from openassembler import oracle
if __name__=='__main__':
 source=HERE/'candidate.ptx';binary=HERE/'candidate.cubin'
 result=oracle.run_tool('ptxas','-arch=sm_80','-o',oracle.tool_path(binary),oracle.tool_path(source),check=False)
 (HERE/'oracle-tool.log').write_text(result.stdout+result.stderr);assert result.returncode==0,result.stderr
 listing=oracle.disasm_text(binary);(HERE/'candidate.sass').write_text(listing)
 files={name:hashlib.sha256((HERE/name).read_bytes()).hexdigest()for name in('candidate.ptx','candidate.cubin','candidate.sass','oracle-tool.log')}
 report=dict(schema=1,passed=True,artifact_hashes=files,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),scope='Actual toolkit acceptance/listing of explicitly normalized source hypothesis; original source graph/dependency/host effects and hardware remain unvalidated')
 (HERE/'oracle-probe.json').write_text(json.dumps(report,indent=2)+'\n');print('Accepted normalized static-value PTX; four raw artifact hashes')

"""Independent toolkit check of normalized source hypothesis, not graph integration."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT))
from openassembler import oracle
if __name__=='__main__':
 source=HERE/'candidate.ptx';binary=HERE/'candidate.cubin'
 result=oracle.run_tool('ptxas','-arch=sm_80','-o',oracle.tool_path(binary),oracle.tool_path(source),check=False)
 (HERE/'oracle-tool.log').write_text(result.stdout+result.stderr)
 rows=dict(schema=1,accepted=result.returncode==0,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),scope='Actual ptxas acceptance of isolated normalized three-entry PTX; does not validate original-source graph/dependency/remote/diagnostic integration or hardware execution')
 if rows['accepted']:
  listing=oracle.disasm_text(binary);(HERE/'candidate.sass').write_text(listing)
  rows.update(cubin_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),listing_sha256=hashlib.sha256(listing.encode()).hexdigest())
 (HERE/'oracle-probe.json').write_text(json.dumps(rows,indent=2)+'\n');print('Normalized PTX accepted',rows['accepted'])
 if not rows['accepted']:raise SystemExit(1)

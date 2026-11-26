"""Original Boolean ABI rejections and actual independent CUDA parameter/storage form."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT))
from openassembler.pyfront import compile_kernel,find_kernel,NotSupported
from openassembler import oracle
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
if __name__=='__main__':
 rows=[]
 for name in('scalar_flag','boolean_arrays','boolean_helper'):
  try:compile_kernel(find_kernel(HERE/'source.py',name))
  except NotSupported as error:
   assert error.path and error.lineno;rows.append(dict(entry=name,rejected=True,diagnostic=str(error)))
  else:raise AssertionError(name)
 ptx=HERE/'source.ptx';cubin=HERE/'reference.cubin';result=oracle.run_tool('nvcc','-ptx','-arch=sm_80','-o',oracle.tool_path(ptx),oracle.tool_path(HERE/'source.cu'),check=False);(HERE/'nvcc.log').write_text(result.stdout+result.stderr);assert result.returncode==0,result.stderr
 text=ptx.read_text();assert '.param .u8 scalar_flag_param_0'in text
 result=oracle.run_tool('ptxas','-arch=sm_80','-o',oracle.tool_path(cubin),oracle.tool_path(ptx),check=False);(HERE/'ptxas.log').write_text(result.stdout+result.stderr);assert result.returncode==0,result.stderr;(HERE/'reference.sass').write_text(oracle.disasm_text(cubin))
 report=dict(schema=1,passed=True,production_fix=False,results=rows,artifact_hashes={p.name:sha(p)for p in(HERE/'source.py',HERE/'source.cu',ptx,cubin,HERE/'reference.sass',HERE/'nvcc.log',HERE/'ptxas.log')},source_hashes={n:sha(ROOT/n)for n in('openassembler/pyfront/compile.py','openassembler/pyfront/types.py','openassembler/backend/ra.py','openassembler/pyfront/modules.py')},script_sha256=sha(Path(__file__)),scope='Three original Boolean scalar/array/helper ABI rejections with original diagnostics and no user execution; independent actual nvcc/ptxas control establishes this toolkit emitted .param.u8 Boolean scalar and byte-storage operations. No OA candidate, GPU execution or universal ABI claim')
 (HERE/'baseline-toolkit.json').write_text(json.dumps(report,indent=2)+'\n');print('Recorded three Boolean ABI rejections and actual CUDA byte-parameter/storage controls')

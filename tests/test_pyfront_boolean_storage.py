"""Original Boolean storage kernels against independent frozen byte/effect references."""
import json
from pathlib import Path
import pytest
from openassembler import build
from openassembler.pyfront import compile_file
from openassembler.pyfront.modules import validate_sources,contract_from_ptx
from openassembler.ptx.parse import parse_module
from openassembler.verify import parameter_contract_from_ptx
from openassembler.ptx.interp import Interp,Launch as PLaunch
from openassembler.emu.machine import Memory,Machine,Launch
from openassembler.emu.program import load_listing
from tests.backend_harness import listing
DATA=json.loads((Path(__file__).parent/'data/boolean_storage_fixtures.json').read_text())
CASES=[(configuration,case)for suite in DATA['suites']for configuration in suite['configurations']for case in suite['cases']]
@pytest.mark.parametrize('configuration,case',CASES,ids=[configuration['name']+'-'+case['name']for configuration,case in CASES])
@pytest.mark.parametrize('level',['safe','opt'])
def test_original_boolean_storage_complete_bytes_and_effects(configuration,case,level,tmp_path):
 for name,text in configuration['files'].items():
  path=tmp_path/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(text)
 source=tmp_path/configuration['entry'];text,_=compile_file(source,source_root=tmp_path if configuration.get('explicit')else None);graph=contract_from_ptx(text);validate_sources(source,graph)
 module=parse_module(text);kernel=module.kernels[case['name']];contract=parameter_contract_from_ptx(text,kernel,allow_buffers=True);assert contract['parameters'][0]['dtype']=='bool'
 code,labels,_,_,_=build.generate(kernel,'sm_80',level);program=load_listing(listing(code,labels));guard=b'\xa5'*64
 for row in case['rows']:
  values=row['values'];a,out,counter=(values[key]for key in('a','out','counter'));before=guard+bytes.fromhex(case['input_hex'])+guard
  for engine in('ptx','normal','poisoned'):
   memory=Memory();memory.write_bytes(a-64,before)
   if a!=out:memory.write_bytes(out-64,guard+bytes.fromhex(case['initial_hex'])+guard)
   memory.write_bytes(counter-64,guard+bytes.fromhex(case['initial_counter_hex'])+guard)
   if engine=='ptx':Interp(module,case['name'],PLaunch(block=(128,1,1),params=values),memory).run()
   else:Machine(program,Launch(block=(128,1,1),params=bytes.fromhex(row['packed_hex']),poison=0xdeadbeef if engine=='poisoned'else None),memory).run()
   for address,field in((a,'input_after_hex'),(out,'expected_hex'),(counter,'counter_hex')):
    wanted=guard+bytes.fromhex(row[field])+guard;assert bytes(memory.read(address-64+i,1)for i in range(len(wanted)))==wanted,(engine,field)

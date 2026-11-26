"""Stage one-byte logical Boolean storage separately from production."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4];HERE=Path(__file__).resolve().parent
INPUTS={'types':'openassembler/pyfront/types.py','buffers':'openassembler/pyfront/buffers.py','pyfront':'openassembler/pyfront/compile.py','verify':'openassembler/verify.py','runner':'openassembler/runner.py'}
def once(s,old,new):
    assert s.count(old)==1,(old,s.count(old))
    return s.replace(old,new)
if __name__=='__main__':
    for suffix,path in INPUTS.items():
        frozen=HERE/('baseline-'+suffix+'.py')
        if not frozen.exists():frozen.write_bytes((ROOT/path).read_bytes())
    (HERE/'inputs.json').write_text(json.dumps(INPUTS,indent=2)+'\n')
    s=(HERE/'baseline-types.py').read_text()
    s=once(s,'    return ArrayType(elem)','    return ArrayType(Scalar("pred") if elem is bool else elem)')
    s+='''\n\ndef element_bytes(kind: str) -> int:
    """Physical element width for an admitted logical scalar register kind."""
    return {"pred": 1, "f32": 4, "s32": 4, "u32": 4, "s64": 8, "u64": 8}[kind]


def logical_dtype(kind: str) -> str:
    return "bool" if kind == "pred" else kind
'''
    (HERE/'candidate-types.py').write_text(s)
    s=(HERE/'baseline-buffers.py').read_text()
    s=once(s,'from openassembler.pyfront.types import ArrayType,Scalar','from openassembler.pyfront.types import ArrayType,Scalar,element_bytes,logical_dtype')
    s=once(s,"    if not isinstance(elem,Scalar) or elem.ptx not in ('s32','u32','f32','s64','u64'):raise ValueError('Buffer requires a supported scalar element type')", "    if elem is bool:elem=Scalar('pred')\n    if not isinstance(elem,Scalar) or elem.ptx not in ('pred','s32','u32','f32','s64','u64'):raise ValueError('Buffer requires a supported scalar element type')")
    s=once(s,"shape=tuple(shape);size=8 if elem.ptx.endswith('64') else 4","shape=tuple(shape);size=element_bytes(elem.ptx)")
    s=once(s,"elem=c._resolve_type(ann.args[0],names)","elem=c._resolve_type(ann.args[0],names,allow_boolean=True)")
    s=once(s,"'dtype':view.elem.ptx","'dtype':logical_dtype(view.elem.ptx)")
    s=once(s,"'element_bytes':8 if view.elem.ptx.endswith('64') else 4","'element_bytes':element_bytes(view.elem.ptx)")
    s=once(s,"0.0 if array.elem=='f32' else 0","False if array.elem=='pred' else 0.0 if array.elem=='f32' else 0")
    s=once(s,"size=8 if array.elem.endswith('64') else 4","size=element_bytes(array.elem)")
    s=once(s,"row['element_bytes'] not in (4,8)","row['element_bytes'] not in (1,4,8)")
    s=once(s,"row['dtype'] not in ('f32','s32','u32','s64','u64') or (8 if row['dtype'].endswith('64') else 4)!=row['element_bytes']", "row['dtype'] not in ('bool','f32','s32','u32','s64','u64') or element_bytes('pred'if row['dtype']=='bool'else row['dtype'])!=row['element_bytes']")
    s=once(s,"Buffer(Scalar(row['dtype']),row['shape']", "Buffer(Scalar('pred'if row['dtype']=='bool'else row['dtype']),row['shape']")
    s=once(s,"zero,_=emitter._lit_as(False if array.elem=='pred' else 0.0 if array.elem=='f32' else 0,array.elem,sub)","zero,zero_kind=emitter._lit_as(0.0 if array.elem=='f32' else 0,'s32'if array.elem=='pred'else array.elem,sub)\n        if array.elem=='pred':zero=emitter._boolean_value(zero,zero_kind)")
    (HERE/'candidate-buffers.py').write_text(s)
    s=(HERE/'baseline-pyfront.py').read_text()
    s=once(s,'from openassembler.pyfront.types import ArrayType, Scalar','from openassembler.pyfront.types import ArrayType, Scalar, element_bytes as _element_bytes, logical_dtype')
    s=once(s,'elem = _resolve_type(ann.args[0], names)','elem = _resolve_type(ann.args[0], names, allow_boolean=True)')
    s=once(s,'elem = _resolve_type(call.args[0], self.names)','elem = _resolve_type(call.args[0], self.names, allow_boolean=True)')
    s=once(s,'element_bytes = 8 if elem.ptx in ("s64", "u64") else 4','element_bytes = _element_bytes(elem.ptx)')
    s=once(s,'element_bytes = 8 if arr.elem in ("s64", "u64") else 4','element_bytes = _element_bytes(arr.elem)')
    s=once(s,'{3 if element_bytes == 8 else 2}','{element_bytes.bit_length() - 1}')
    s=once(s,'''        dst = self.s.new_reg(arr.elem)
        self.s.emit(f" ld.{arr.space}.{arr.elem if arr.elem in ('f32','s64','u64') else 'u32'} {dst.name}, {addr};")
        return dst''','''        dst = self.s.new_reg(arr.elem)
        if arr.elem == "pred":
            byte = self.s.new_reg("u32")
            self.s.emit(f" ld.{arr.space}.u8 {byte.name}, {addr};")
            self.s.emit(f" setp.ne.u32 {dst.name}, {byte.name}, 0;")
        else:
            self.s.emit(f" ld.{arr.space}.{arr.elem if arr.elem in ('f32','s64','u64') else 'u32'} {dst.name}, {addr};")
        return dst''')
    s=once(s,'''        self.s.emit(f" st.{arr.space}.{arr.elem if arr.elem in ('f32','s64','u64') else 'u32'} {addr}, {val.name};")''','''        if arr.elem == "pred":
            byte = self.s.new_reg("u32")
            self.s.emit(f" selp.u32 {byte.name}, 1, 0, {val.name};")
            self.s.emit(f" st.{arr.space}.u8 {addr}, {byte.name};")
        else:
            self.s.emit(f" st.{arr.space}.{arr.elem if arr.elem in ('f32','s64','u64') else 'u32'} {addr}, {val.name};")''')
    s=once(s,'if s.boolean_inputs or s.wide_integer or any(a.view is not None for a in s.arrays.values()):','if s.boolean_inputs or s.wide_integer or any(a.elem == "pred" or a.view is not None for a in s.arrays.values()):')
    s=once(s,'"dtype": s.arrays[pname].elem if pname in s.arrays else', '"dtype": logical_dtype(s.arrays[pname].elem) if pname in s.arrays else')
    s=once(s,'element_bytes = 8 if ty in ("s64", "u64") else 4','element_bytes = _element_bytes(ty)')
    s=once(s,'storage_type = "b64" if element_bytes == 8 else', 'storage_type = "b8" if element_bytes == 1 else "b64" if element_bytes == 8 else')
    (HERE/'candidate-pyfront.py').write_text(s)
    s=(HERE/'baseline-verify.py').read_text()
    s=once(s,"        if parameter['dtype'] == 'bool' and parameter['kind'] != 'scalar':\n            raise ValueError('Boolean array storage has no validated parameter contract')\n",'')
    (HERE/'candidate-verify.py').write_text(s)
    s=(HERE/'baseline-runner.py').read_text()
    s=once(s,'DTYPES = {"f32": "<f",','DTYPES = {"bool": "<B", "f32": "<f",')
    s=once(s,"*(r.randrange(cnt) for _ in range(cnt))", "*(r.randrange(2 if dt == 'bool' else cnt) for _ in range(cnt))")
    s=once(s,"                    if dt in ('f32', 'f16', 'bf16'):","                    if dt == 'bool':\n                        if type(value) is not bool:raise ValueError('Boolean initialization values must be true or false')\n                        values.append(int(value))\n                    elif dt in ('f32', 'f16', 'bf16'):")
    s=once(s,"            is_boolean = logical_parameters.get(name,{}).get('dtype')=='bool'", "            logical = logical_parameters.get(name,{})\n            is_boolean = logical.get('kind')=='scalar' and logical.get('dtype')=='bool'\n            is_boolean_array = logical.get('kind')=='array' and logical.get('dtype')=='bool'\n            boolean_buffer = argument[0]=='buf' and spec['buffers'][argument[1]].get('dtype','f32')=='bool'\n            if is_boolean_array != boolean_buffer:\n                raise RunError(f'Boolean array {name} requires a Boolean buffer and a matching logical source contract')")
    (HERE/'candidate-runner.py').write_text(s)
    print('Staged five exact storage candidate modules; production unchanged')

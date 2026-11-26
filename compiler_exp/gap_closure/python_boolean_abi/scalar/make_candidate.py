"""Stage logical Boolean source/host ABI on the validated physical byte slots."""
from pathlib import Path

HERE=Path(__file__).resolve().parent


def once(text,old,new):
    assert text.count(old)==1,old
    return text.replace(old,new)


if __name__=='__main__':
    s=(HERE/'baseline-pyfront.py').read_text()
    s=once(s,'    smem_bytes: int = 0','    boolean_inputs: dict[str, _Reg] = field(default_factory=dict)\n    smem_bytes: int = 0')
    s=once(s,'        t = _resolve_type(arg.annotation, names)','        t = _resolve_type(arg.annotation, names, allow_boolean=True)')
    s=once(s,'''        else:
            scope.params.append((t.ptx, arg.arg))
            scope.vars[arg.arg] = scope.new_reg(t.ptx)''','''        else:
            scope.params.append(("u8" if t.ptx == "pred" else t.ptx, arg.arg))
            scope.vars[arg.arg] = scope.new_reg(t.ptx)
            if t.ptx == "pred":scope.boolean_inputs[arg.arg] = scope.new_reg("u32")''')
    s=once(s,'    if s.wide_integer or any(a.view is not None for a in s.arrays.values()):','    if s.boolean_inputs or s.wide_integer or any(a.view is not None for a in s.arrays.values()):')
    s=once(s,'"dtype": s.arrays[pname].elem if pname in s.arrays else ptx_t}',
           '"dtype": s.arrays[pname].elem if pname in s.arrays else "bool" if pname in s.boolean_inputs else ptx_t}')
    s=once(s,'''        else:
            r = s.vars[pname]
            out.append(f" ld.param.{ptx_t if ptx_t in ('f32','s64','u64') else 'u32'} {r.name}, [{pname}];")''','''        else:
            r = s.vars[pname]
            if pname in s.boolean_inputs:
                loaded = s.boolean_inputs[pname]
                out.append(f" ld.param.u8 {loaded.name}, [{pname}];")
                out.append(f" setp.ne.u32 {r.name}, {loaded.name}, 0;")
            else:
                out.append(f" ld.param.{ptx_t if ptx_t in ('f32','s64','u64') else 'u32'} {r.name}, [{pname}];")''')
    (HERE/'candidate-pyfront.py').write_text(s)
    s=(HERE/'baseline-verify.py').read_text()
    s=once(s,"_TYPED_WIDTH = {'f32': 4,", "_TYPED_WIDTH = {'bool': 1, 'f32': 4,")
    s=once(s,"        required = 'u64' if parameter['kind'] == 'array' else parameter['dtype']",'''        if parameter['dtype'] == 'bool' and parameter['kind'] != 'scalar':
            raise ValueError('Boolean array storage has no validated parameter contract')
        required = 'u64' if parameter['kind'] == 'array' else 'u8' if parameter['dtype'] == 'bool' else parameter['dtype']''')
    s=once(s,'def parameter_contract_from_ptx(source: str, kernel: Kernel) -> dict | None:',
           'def parameter_contract_from_ptx(source: str, kernel: Kernel, *, allow_buffers=False) -> dict | None:')
    s=once(s,"    if any(line.strip().startswith('// openassembler-pyfront-buffer-v1') for line in source.splitlines()):",
           "    if not allow_buffers and any(line.strip().startswith('// openassembler-pyfront-buffer-v1') for line in source.splitlines()):")
    s=once(s,"    if dtype == 'f32':\n", "    if dtype == 'bool':return index % 2\n    if dtype == 'f32':\n")
    s=once(s,"            bits = n_elems if dtype != 'f32' and name.lower() in dimension_names else _typed_bits(rng, dtype, seed)",
           "            bits = n_elems if dtype not in ('f32','bool') and name.lower() in dimension_names else _typed_bits(rng, dtype, seed)")
    (HERE/'candidate-verify.py').write_text(s)
    s=(HERE/'baseline-build.py').read_text()
    s=once(s,'    buffer_contract: dict | None = None','    logical_parameter_contract: dict | None = None\n    buffer_contract: dict | None = None')
    s=once(s,'        if self.buffer_contract is not None:', '        if self.logical_parameter_contract is not None:r["logical_parameter_contract"] = self.logical_parameter_contract\n        if self.buffer_contract is not None:')
    s=once(s,'        k.directives = {key: list(value) for key, value in module.kernels[kname].directives.items()}', '''        from openassembler.verify import parameter_contract_from_ptx
        try:
            parameter_contract = parameter_contract_from_ptx(ptx,module.kernels[kname],allow_buffers=True)
            if parameter_contract and any(row['dtype']=='bool' for row in parameter_contract['parameters']):
                k.logical_parameter_contract = parameter_contract
        except (ValueError,KeyError,TypeError) as error:
            k.error = f'invalid logical parameter contract: {error}'
            so.kernels.append(k)
            continue
        k.directives = {key: list(value) for key, value in module.kernels[kname].directives.items()}''')
    (HERE/'candidate-build.py').write_text(s)
    s=(HERE/'baseline-ownership.py').read_text()
    s=once(s,"        texts[kernel['name']]['params']=deepcopy(kernel['params'])",'''        texts[kernel['name']]['params']=deepcopy(kernel['params'])
        if kernel.get('logical_parameter_contract') is not None:
            from openassembler.verify import _parameter_contract
            from openassembler.ptx.parse import Kernel
            _parameter_contract(Kernel(kernel['name'],kernel['params'],[],{}),kernel['logical_parameter_contract'])
            texts[kernel['name']]['logical_parameter_contract']=deepcopy(kernel['logical_parameter_contract'])''')
    s=once(s,"        texts[kernel['name']]['params']=kernel['params']",'''        texts[kernel['name']]['params']=kernel['params']
        if kernel.get('logical_parameter_contract') is not None:
            from openassembler.verify import _parameter_contract
            from openassembler.ptx.parse import Kernel
            _parameter_contract(Kernel(kernel['name'],kernel['params'],[],{}),kernel['logical_parameter_contract'])
            texts[kernel['name']]['logical_parameter_contract']=kernel['logical_parameter_contract']''')
    (HERE/'candidate-ownership.py').write_text(s)
    s=(HERE/'baseline-runner.py').read_text()
    s=once(s,"def _checked_buffer_bindings(target, out_dir, arguments, buffers, buffer_specs=None):",'''def _logical_parameters(target,out_dir):
    contract = target.kernel.get('logical_parameter_contract')
    if contract is None:return {}
    import hashlib
    from openassembler.verify import parameter_contract_from_ptx
    from openassembler.ptx.parse import parse_module
    try:
        raw = P.artifact_path(out_dir,target.entry['ptx']).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=target.entry['native_ownership']['ptx_sha256']:
            raise ValueError('logical parameter PTX identity differs from native build')
        current = parameter_contract_from_ptx(raw.decode(),parse_module(raw.decode()).kernels[target.label],allow_buffers=True)
        if current != contract:raise ValueError('logical parameter manifest differs from validated PTX contract')
    except (OSError,ValueError,KeyError,TypeError) as error:
        raise RunError(f'logical parameter launch contract unavailable or invalid: {error}') from None
    return {row['name']:row for row in contract['parameters']}


def _checked_buffer_bindings(target, out_dir, arguments, buffers, buffer_specs=None):''')
    s=once(s,'    """The verify.py-style harness for one kernel: see this module\'s docstring."""', '''    if t.kernel.get('logical_parameter_contract') is not None:
        plan.cases.append(Case(t.label,str(t.source),-1,None,[],skip='logical Boolean parameters require --spec with actual Boolean values'))
        return
    """The verify.py-style harness for one kernel: see this module's docstring."""''')
    s=once(s,'        args = []\n        for a in st.get("args", []):','        logical_parameters = _logical_parameters(t,out_dir)\n        args = []\n        boolean_arguments = set()\n        for a in st.get("args", []):')
    s=once(s,"            elif any(kind in a for kind in ('i8','u8','i16','u16')):",'''            elif "bool" in a:
                if len(a)!=1 or type(a['bool']) is not bool:
                    raise RunError(f'spec: step {si} Boolean argument requires a true or false JSON value')
                boolean_arguments.add(len(args))
                args.append(('u8',int(a['bool'])))
            elif any(kind in a for kind in ('i8','u8','i16','u16')):''')
    s=once(s,"        for (dtype,name),argument in zip(t.kernel['params'],args):",'''        for index,((dtype,name),argument) in enumerate(zip(t.kernel['params'],args)):
            is_boolean = logical_parameters.get(name,{}).get('dtype')=='bool'
            if is_boolean != (index in boolean_arguments):
                raise RunError(f'Boolean parameter {name} requires a Boolean argument and a matching logical source contract')''')
    (HERE/'candidate-runner.py').write_text(s)
    print('Staged five coordinated logical Boolean kernel ABI modules')

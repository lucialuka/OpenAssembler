"""Stage complete scalar narrow entry ABI experiments from exact frozen modules."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def replace(text, old, new):
    assert text.count(old) == 1, old
    return text.replace(old, new)


if __name__ == '__main__':
    text = (HERE/'baseline-parse.py').read_text()
    text = replace(text, 'def _parameters(text: str, name: str, attributes=None)',
                   'def _parameters(text: str, name: str, attributes=None, *, allow_narrow=False)')
    text = replace(text,
        "    params = []\n    for part in _split_top(text):",
        "    params = []\n    scalar_types = 'b8|u8|s8|b16|u16|s16|' if allow_narrow else ''\n    for part in _split_top(text):")
    text = replace(text, "r'\\s*\\.param\\s+\\.(b32|u32|s32|f32|b64|u64|s64|f64)\\s+'",
                   "r'\\s*\\.param\\s+\\.('+scalar_types+r'b32|u32|s32|f32|b64|u64|s64|f64)\\s+'")
    text = replace(text, '_parameters(raw_params, name,attributes)', '_parameters(raw_params, name,attributes,allow_narrow=True)')
    insert = '''
def parameter_layout(parameters):
    """Natural scalar entry slots, in bytes from the start of the parameter bank."""
    result = {}
    offset = 0
    for dtype, name in parameters:
        if dtype not in _TYPE_SIZE or dtype == 'f16' or name in result:
            raise ValueError('unsupported or duplicate scalar parameter: '+str(name))
        size = _TYPE_SIZE[dtype]
        offset = (offset+size-1)//size*size
        result[name] = (offset, size)
        offset += size
    return result


def parameter_load(kernel, address, dtype):
    """Validate a scalar load within its declared entry parameter slot."""
    slots = parameter_layout(kernel.params)
    if not isinstance(address, Addr) or address.base not in slots or dtype not in _TYPE_SIZE:
        raise ValueError('parameter load requires a declared scalar parameter and supported width')
    offset, size = slots[address.base]
    width = _TYPE_SIZE[dtype]
    if address.off < 0 or address.off+width > size or (offset+address.off) % width:
        raise ValueError('parameter load must be naturally aligned and within its declared scalar slot')
    return offset+address.off, width


'''
    text = replace(text, 'def _parameters(text: str', insert+'def _parameters(text: str')
    (HERE/'candidate-parse.py').write_text(text)
    text = (HERE/'baseline-contracts.py').read_text()
    text = replace(text, "    for t in ('s32','u32','s64','u64'):\n",
                   "    for t in ('b8','u8','s8','b16','u16','s16'):\n        add('ld','param.'+t,2)\n    for t in ('s32','u32','s64','u64'):\n")
    (HERE/'candidate-contracts.py').write_text(text)
    for suffix in ('ra', 'backend'):
        text = (HERE/('baseline-'+suffix+'.py')).read_text()
        old = '''        self.param_off = {}
        off = 0
        for ty, name in k.params:
            sz = 8 if ty.endswith("64") else 4
            off = (off + sz - 1) // sz * sz
            self.param_off[name] = PARAM_BASE + off
            off += sz'''
        new = '''        from openassembler.ptx.parse import parameter_layout
        self.param_off = {name: PARAM_BASE+offset for name, (offset, size) in parameter_layout(k.params).items()}'''
        text = replace(text, old, new)
        if suffix == 'ra':
            old = '''            off = self.param_off[a[1].base] + a[1].off
            self.bind(a[0].name, ("cpair", off) if t and t.endswith("64") else C(off))'''
            new = '''            from openassembler.ptx.parse import parameter_load
            try: relative, width = parameter_load(self.k, a[1], t)
            except ValueError as error: raise NotSupported(str(error)) from None
            off = PARAM_BASE+relative
            if width < 4:
                value = self.new()
                self.emit("MOV {d0}, {s0}", [value], [C(off & ~3)])
                shift, bits = (off & 3)*8, width*8
                if t.startswith('s'):
                    left = 32-bits-shift
                    if left:
                        moved = self.new()
                        self.emit(f"SHF.L.U32 {{d0}}, {{s0}}, {left:#x}, RZ", [moved], [value])
                        value = moved
                    result = self.new()
                    self.emit(f"SHF.R.S32.HI {{d0}}, RZ, {32-bits:#x}, {{s0}}", [result], [value])
                else:
                    if shift:
                        moved = self.new()
                        self.emit(f"SHF.R.U32.HI {{d0}}, RZ, {shift:#x}, {{s0}}", [moved], [value])
                        value = moved
                    result = self.new()
                    self.emit(f"LOP3.LUT {{d0}}, {{s0}}, {(1 << bits)-1:#x}, RZ, 0xc0, !PT", [result], [value])
                self.bind(a[0].name, result)
            else:
                self.bind(a[0].name, ("cpair", off) if t and t.endswith("64") else C(off))'''
        else:
            old = '''            off = self.param_off[a[1].base] + a[1].off
            if t in ("u64", "b64", "s64"):'''
            new = '''            from openassembler.ptx.parse import parameter_load
            try: relative, width = parameter_load(self.k, a[1], t)
            except ValueError as error: raise NotSupported(str(error)) from None
            off = PARAM_BASE+relative
            if width < 4:
                rd = self.r(d)
                self.emit(f"MOV R{rd}, c[0x0][{off & ~3:#x}]", writes=[rd])
                shift, bits = (off & 3)*8, width*8
                if t.startswith('s'):
                    left = 32-bits-shift
                    if left:self.emit(f"SHF.L.U32 R{rd}, R{rd}, {left:#x}, RZ", reads=[rd], writes=[rd])
                    self.emit(f"SHF.R.S32.HI R{rd}, RZ, {32-bits:#x}, R{rd}", reads=[rd], writes=[rd])
                else:
                    if shift:self.emit(f"SHF.R.U32.HI R{rd}, RZ, {shift:#x}, R{rd}", reads=[rd], writes=[rd])
                    self.emit(f"LOP3.LUT R{rd}, R{rd}, {(1 << bits)-1:#x}, RZ, 0xc0, !PT", reads=[rd], writes=[rd])
                return
            if t in ("u64", "b64", "s64"):'''
        text = replace(text, old, new)
        (HERE/('candidate-'+suffix+'.py')).write_text(text)
    text = (HERE/'baseline-semantics.py').read_text()
    text = replace(text,
        '        it.put(ln, d, it.L.params[a.base], bits(t))\n        return',
        '''        from openassembler.ptx.parse import parameter_load
        parameter_load(it.k, a, t)
        value = (it.L.params[a.base] >> (8*a.off)) & M(bits(t))
        it.put(ln, d, sext(value, bits(t)) if signed(t) else value, max(32, bits(t)))
        return''')
    (HERE/'candidate-semantics.py').write_text(text)
    text = (HERE/'baseline-verify.py').read_text()
    text = replace(text, '_SCALAR_FMT = {', '_SCALAR_FMT = {"b8":"<B", "u8":"<B", "s8":"<b", "b16":"<H", "u16":"<H", "s16":"<h", ')
    text = replace(text, '    return rng.randrange(1, max(1, n_elems) + 1)',
        '''    if ptx_type in ('b8','u8','s8','b16','u16','s16'):
        bits = int(ptx_type[1:])
        return rng.randrange(-(1 << (bits-1)), 1 << (bits-1)) if ptx_type.startswith('s') else rng.randrange(1 << bits)
    return rng.randrange(1, max(1, n_elems) + 1)''')
    text = replace(text, '            _align(sass_param_bytes, 8 if ptx_type.endswith("64") else 4)',
                   '            _align(sass_param_bytes, struct.calcsize(_SCALAR_FMT.get(ptx_type, "<i")))')
    text = replace(text, '            interp_params[name] = (struct.unpack',
        '''            if ptx_type in ('b8','u8','s8','b16','u16','s16') and name.lower() in n_like:
                bits = int(ptx_type[1:])
                maximum = (1 << (bits-int(ptx_type.startswith('s'))))-1
                if not 0 <= v <= maximum:raise ValueError('verification dimension does not fit narrow parameter '+name)
            interp_params[name] = (struct.unpack''')
    (HERE/'candidate-verify.py').write_text(text)
    text = (HERE/'baseline-runner.py').read_text()
    text = replace(text, '            args.append(("u32", v & 0xFFFFFFFF))',
        '''            width = int(ty[1:])
            maximum = (1 << (width-int(ty.startswith('s'))))-1
            if width < 32 and not 0 <= v <= maximum:
                plan.cases.append(Case(k['name'],t.entry['source'],-1,-1,[],skip=f'auto-harness value for {pname} does not fit .{ty}; use --spec'))
                return
            args.append(("u"+str(width), v & ((1 << width)-1)))''')
    text = replace(text, '            elif "i32" in a or "u32" in a:',
        '''            elif any(kind in a for kind in ('i8','u8','i16','u16')):
                kind = next(kind for kind in ('i8','u8','i16','u16') if kind in a)
                width = int(kind[1:]);v = a[kind]
                low, high = (-(1 << (width-1)), 1 << (width-1)) if kind.startswith('i') else (0, 1 << width)
                if len(a) != 1 or type(v) is not int or not low <= v < high:
                    raise RunError(f"spec: step {si} invalid {kind} argument")
                args.append(("u"+str(width), v & ((1 << width)-1)))
            elif "i32" in a or "u32" in a:''')
    text = replace(text, "        _checked_buffer_bindings(t,out_dir,args,buffers,spec['buffers'])",
        '''        for (dtype,name),argument in zip(t.kernel['params'],args):
            if dtype in ('b8','u8','s8','b16','u16','s16') and argument[0] != 'u'+dtype[1:]:
                raise RunError(f'narrow scalar parameter {name} requires an exact {dtype[1:]}-bit scalar argument')
            if argument[0] in ('u8','u16') and dtype not in ('b'+argument[0][1:],'s'+argument[0][1:],argument[0]):
                raise RunError(f'narrow scalar argument for {name} does not match PTX parameter width')
        _checked_buffer_bindings(t,out_dir,args,buffers,spec['buffers'])''')
    (HERE/'candidate-runner.py').write_text(text)
    text = (HERE/'baseline-driver.py').read_text()
    text = replace(text, '                elif a[0] == "u32":',
        '''                elif a[0] in ('u8','u16'):
                    bits = int(a[0][1:])
                    if type(a[1]) is not int or not 0 <= a[1] < (1 << bits):
                        raise ValueError('invalid narrow scalar launch argument')
                    cargs.append((C.c_uint8 if bits == 8 else C.c_uint16)(a[1]))
                elif a[0] == "u32":''')
    (HERE/'candidate-driver.py').write_text(text)
    print('Staged eight coordinated scalar narrow-parameter modules')

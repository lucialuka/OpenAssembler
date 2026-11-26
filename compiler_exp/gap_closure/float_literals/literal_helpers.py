"""Candidate literal grammar and context conversion, injected into the parser."""
_DECIMAL_FLOAT = re.compile(r'[+-]?(?:(?:[0-9]+\.[0-9]*|\.[0-9]+)(?:[eE][+-]?[0-9]+)?|[0-9]+[eE][+-]?[0-9]+)')


def _decimal_float(text):
    value=float(text)
    # CUDA 12.8 ptxas reports constant overflow for nonzero binary64
    # subnormals/underflow and infinities. Zero coefficients keep their sign.
    nonzero=bool(re.search(r'[1-9]',re.split('[eE]',text)[0]))
    if not math.isfinite(value)or(nonzero and abs(value)<float.fromhex('0x1p-1022')):
        raise SyntaxError(f'decimal floating literal outside supported binary64 range: {text}')
    return ImmF(struct.unpack('<Q',struct.pack('<d',value))[0],64)


def _literal_f32(bits):
    sign=(bits>>63)<<31;exponent=(bits>>52)&2047;fraction=bits&((1<<52)-1)
    if exponent==2047:
        # The oracle preserves sign/high payload and quiets converted NaNs.
        return sign|0x7f800000|((fraction>>29)|0x400000 if fraction else 0)
    value=struct.unpack('<d',struct.pack('<Q',bits))[0]
    try:return struct.unpack('<I',struct.pack('<f',value))[0]
    except OverflowError:return sign|0x7f800000


def _literal_operands(op,mods,args,line):
    types=[mod for mod in mods if re.fullmatch(r'[busf](?:8|16|32|64)',mod)]
    dtype=types[-1]if types else None
    def convert(value,index,context):
        if isinstance(value,Vec):
            # Pack/unpack .b64 contains two .b32 source components.
            component='b32'if op=='mov'and context=='b64'else context
            return Vec(tuple(convert(part,index,component)for part in value.elems))
        if isinstance(value,ImmI)and op=='mov'and context in('f32','f64')and index==1:
            raise SyntaxError(f'PTX line {line}: {op}.{context} requires a floating literal, not an integer literal')
        if not isinstance(value,ImmF):return value
        if (index==0 or op=='selp'and index==3 or op in('shl','shr')and index==2):
            raise SyntaxError(f'PTX line {line}: floating literal is invalid at operand {index} of {op}')
        if context=='f32':return ImmF(_literal_f32(value.bits),32)if value.width==64 else value
        if context=='f64':
            # A raw 0f literal retains its exact 32 bits for mov.f64: the
            # observed wider move zero-extends them rather than converting.
            return ImmF(value.bits,64)if op in('mov','st')and value.width==32 else value
        if context in('b32','b64')and value.width==int(context[1:]):return value
        raise SyntaxError(f'PTX line {line}: floating literal width {value.width} is invalid for {op}.{context or "untyped"}')
    return[convert(value,index,dtype)for index,value in enumerate(args)]

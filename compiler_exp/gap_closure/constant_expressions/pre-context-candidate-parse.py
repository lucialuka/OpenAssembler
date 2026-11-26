"""Minimal PTX parser: one module -> kernels of flat instruction lists.

Covers the syntax our probes and typical ptxas input use: directives, .reg /
.shared / .local / .const / .global declarations (incl. `%r<N>` ranges),
labels, guards, vector operands `{a, b}`, addresses `[reg+off]`, `a|b`
predicate pairs, and immediates (hex, decimal, 0f/0d float bits).
"""

from __future__ import annotations

import re
import math
import struct
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Var:
    name: str          # %r1, smem, a param name


@dataclass(frozen=True)
class ImmI:
    value: int


@dataclass(frozen=True)
class ImmF:
    bits: int          # raw IEEE bits
    width: int         # 32 or 64


@dataclass(frozen=True)
class Addr:
    base: str          # register or symbol name
    off: int = 0


@dataclass(frozen=True)
class Vec:
    elems: tuple       # tuple of Var | ImmI | None ("_" sink)


@dataclass(frozen=True)
class PairDst:
    a: Var             # setp %p1|%p2
    b: Var


Operand = Var | ImmI | ImmF | Addr | Vec | PairDst


@dataclass
class Instr:
    op: str                         # "add"
    mods: tuple[str, ...]           # ("s32",) or ("sync", "bfly", "b32")
    args: list[Operand]
    guard: tuple[str, bool] | None  # (pred, negated)
    line: int = 0

    def has(self, *ms: str) -> bool:
        return all(m in self.mods for m in ms)


@dataclass
class VarDecl:
    space: str        # shared, local, const, global
    name: str
    size: int
    align: int


@dataclass
class Kernel:
    name: str
    params: list[tuple[str, str]]          # (type, name)
    body: list[Instr]
    labels: dict[str, int]                 # label -> index into body
    decls: list[VarDecl] = field(default_factory=list)
    branchtargets: dict[str, list[str]] = field(default_factory=dict)
    directives: dict[str, tuple[int, ...]] = field(default_factory=dict)
    parameter_attributes: dict[str, dict] = field(default_factory=dict)


@dataclass
class Module:
    kernels: dict[str, Kernel]
    decls: list[VarDecl]


_TYPE_SIZE = {"b8": 1, "u8": 1, "s8": 1, "b16": 2, "u16": 2, "s16": 2, "f16": 2,
              "b32": 4, "u32": 4, "s32": 4, "f32": 4, "b64": 8, "u64": 8, "s64": 8, "f64": 8}
_VAR_DECL = re.compile(r"\.(shared|local|const|global)\s+(?:\.align\s+(\d+)\s+)?\.(\w+)\s+(\w+)(?:\[(\d+)\])?")
_ENTRY = re.compile(r"(?<![\w$.])\.entry\s+([\w$]+)\s*\(")


def _strip_comments(src: str) -> str:
    # Strings can contain comment markers (e.g. .file paths). Preserve offsets
    # and newlines for diagnostics and for the entry scanner's source slices.
    return re.sub(r'"(?:\\.|[^"\\])*"|/\*.*?\*/|//[^\n]*',
                  lambda m: m[0] if m[0].startswith('"') else re.sub(r'[^\n]', ' ', m[0]),
                  src, flags=re.S)


def _mask_strings(src: str) -> str:
    return re.sub(r'"(?:\\.|[^"\\])*"', lambda m: ' ' * len(m[0]), src)


def entry_definitions(src: str):
    """Yield complete entry definitions, skipping quoted text and prototypes.

    Return (name, params, directives, body, start, end). Header delimiters are
    balanced independently of instruction parsing; an unrecognized header never
    makes a kernel silently disappear. Offsets refer to comment-stripped text.
    """
    src = _strip_comments(src)
    masked = _mask_strings(src)
    pos = 0
    while match := _ENTRY.search(masked, pos):
        depth, end = 1, match.end()
        while end < len(masked) and depth:
            depth += (masked[end] == '(') - (masked[end] == ')')
            end += 1
        if depth:
            raise SyntaxError(f"entry {match[1]}: unbalanced parameter list")
        params = src[match.end():end - 1]
        delimiter = re.search(r'[;{]', masked[end:])
        if delimiter is None:
            raise SyntaxError(f"entry {match[1]}: missing body or declaration terminator")
        begin = end + delimiter.start()
        if masked[begin] == ';':
            pos = begin + 1
            continue
        finish = _match_brace(masked, begin)
        yield match[1], params, src[end:begin], src[begin + 1:finish], match.start(), finish + 1
        pos = finish + 1


def entry_names(src: str) -> list[str]:
    return [definition[0] for definition in entry_definitions(src)]


def _parameters(text: str, name: str, attributes=None) -> list[tuple[str, str]]:
    params = []
    for part in _split_top(text):
        match = re.fullmatch(r'\s*\.param\s+\.(b32|u32|s32|f32|b64|u64|s64|f64)\s+'
                             r'(?:(\.ptr)(?:\s*\.(global))?(?:\s*\.align\s+([0-9]+))?\s+)?([\w$]+)\s*', part)
        if match is None:
            raise SyntaxError(f"entry {name}: unsupported parameter ABI {part!r}; scalar 32/64-bit parameters required")
        dtype, pointer, space, alignment, symbol = match.groups()
        if symbol in {p[1] for p in params}:
            raise SyntaxError(f"entry {name}: duplicate parameter {symbol}")
        if pointer:
            if dtype not in ('u64','b64'):
                raise SyntaxError(f'entry {name}: pointer parameter ABI requires u64/b64 under address_size 64')
            align=int(alignment) if alignment is not None else 4
            if align<1 or align & (align-1):raise SyntaxError(f'entry {name}: pointer alignment must be a positive power of two')
            if attributes is not None:
                attributes[symbol]={'pointer':True,'space':space or 'generic','alignment':align}
        params.append((dtype, symbol))
    return params


def _directives(text: str, name: str) -> dict[str, tuple[int, ...]]:
    directives = {}
    while text.strip():
        match = re.match(r'\s*\.(maxntid|reqntid|maxnreg|minnctapersm|maxnctapersm)\s+([0-9]+(?:\s*,\s*[0-9]+)*)', text)
        if match is None:
            raise SyntaxError(f"entry {name}: unsupported entry directive {text.strip()!r}")
        key, values = match[1], tuple(int(x) for x in match[2].split(','))
        if key in directives or any(x <= 0 for x in values):
            raise SyntaxError(f"entry {name}: duplicate or invalid .{key}")
        if key in ('maxntid', 'reqntid'):
            if len(values) > 3:
                raise SyntaxError(f"entry {name}: .{key} requires one through three dimensions")
            values += (1,) * (3 - len(values))
        elif len(values) != 1:
            raise SyntaxError(f"entry {name}: .{key} requires one integer")
        directives[key] = values
        text = text[match.end():]
    if 'maxntid' in directives and 'reqntid' in directives:
        raise SyntaxError(f"entry {name}: .maxntid and .reqntid cannot be combined")
    return directives


def _match_brace(src: str, start: int) -> int:
    depth = 0
    for i in range(start, len(src)):
        if src[i] == "{":
            depth += 1
        elif src[i] == "}":
            depth -= 1
            if depth == 0:
                return i
    raise SyntaxError("unbalanced braces")


def _split_top(s: str, sep: str = ",") -> list[str]:
    out, depth, cur = [], 0, []
    for ch in s:
        if ch in "{[(":
            depth += 1
        elif ch in "}])":
            depth -= 1
        if ch == sep and depth == 0:
            out.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    if "".join(cur).strip():
        out.append("".join(cur).strip())
    return out


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


_CONSTANT_TOKEN = re.compile(r'\s*(0[dD][0-9a-fA-F]{16}|0[fF][0-9a-fA-F]{8}|(?:[0-9]+\.[0-9]*|\.[0-9]+)(?:[eE][+-]?[0-9]+)?|[0-9]+[eE][+-]?[0-9]+|0[xX][0-9a-fA-F]+U?|0[bB][01]+U?|[0-9]+U?|\.[su]64|<<|>>|<=|>=|==|!=|&&|\|\||[()+*/%<>&|^!~?:-])')
_CONSTANT_PRECEDENCE={'||':1,'&&':2,'|':3,'^':4,'&':5,'==':6,'!=':6,'<':7,'<=':7,'>':7,'>=':7,'<<':8,'>>':8,'+':9,'-':9,'*':10,'/':10,'%':10}


def _constant_expression(text):
    """Pratt parser over s64/u64/f64 values; no host language evaluation."""
    if re.fullmatch(r'0[fF][0-9a-fA-F]{8}',text):return ImmF(int(text[2:],16),32)
    if re.search(r'%[0-9A-Za-z_$]',text):raise SyntaxError('PTX remainder must be separated from a following register-like token')
    tokens=[];position=0;mask=(1<<64)-1
    while position<len(text):
        match=_CONSTANT_TOKEN.match(text,position)
        if match is None:
            if text[position:].strip():raise SyntaxError(f'invalid PTX constant expression: {text}')
            break
        tokens.append(match[1]);position=match.end()
    if not tokens:raise SyntaxError('empty PTX constant expression')
    if len(tokens)>4096:raise SyntaxError('PTX constant expression exceeds 4096 tokens')
    cursor=0;depth=0
    def peek():return tokens[cursor]if cursor<len(tokens)else None
    def take():
        nonlocal cursor
        token=peek();cursor+=1;return token
    def integer(value):
        kind,bits=value
        return bits-(1<<64)if kind=='s'and bits>>63 else bits
    def float_value(value):return struct.unpack('<d',struct.pack('<Q',value[1]))[0]
    def floating(number):return'f',0x7ff8000000000000 if math.isnan(number)else struct.unpack('<Q',struct.pack('<d',number))[0]
    def require_int(value):
        if value[0]=='f':raise SyntaxError('PTX constant expression requires an integer operand')
    def binary(op,left,right):
        if left[0]=='f'or right[0]=='f':
            if left[0]!='f'or right[0]!='f':raise SyntaxError('PTX constant expression mixes integer and floating operands')
            a,b=float_value(left),float_value(right)
            if op=='/'and b==0:raise SyntaxError('division by zero in PTX constant expression')
            if op in('+','-','*','/'):
                if math.isnan(a):return'f',left[1]|(1<<51)
                if math.isnan(b):return'f',right[1]|(1<<51)
            if op=='+':return floating(a+b)
            if op=='-':return floating(a-b)
            if op=='*':return floating(a*b)
            if op=='/':
                if b==0:raise SyntaxError('division by zero in PTX constant expression')
                return floating(a/b)
            if op in('<','<=','>','>=','==','!='):
                return's',int({'<':a<b,'<=':a<=b,'>':a>b,'>=':a>=b,'==':a==b,'!=':a!=b}[op])
            raise SyntaxError(f'floating PTX constant expression does not support {op}')
        kind='u'if'u'in(left[0],right[0])else's'
        a,b=(left[1],right[1])if kind=='u'else(integer(left),integer(right))
        if op=='+':value=a+b
        elif op=='-':value=a-b
        elif op=='*':value=a*b
        elif op=='/':
            if b==0:raise SyntaxError('division by zero in PTX constant expression')
            value=abs(a)//abs(b);value=-value if(a<0)!=(b<0)else value
        elif op=='%':
            if right[1]==0:raise SyntaxError('remainder by zero in PTX constant expression')
            return's',left[1]%right[1]
        elif op in('<<','>>'):
            a=integer(left);count=right[1]&63
            value=a<<count if op=='<<'else a>>count
            return left[0],value&mask
        elif op in('&','|','^'):
            return kind,{'&':left[1]&right[1],'|':left[1]|right[1],'^':left[1]^right[1]}[op]
        elif op in('<','<=','>','>=','==','!='):
            return's',int({'<':a<b,'<=':a<=b,'>':a>b,'>=':a>=b,'==':a==b,'!=':a!=b}[op])
        elif op in('&&','||'):return's',int(bool(a)and bool(b)if op=='&&'else bool(a)or bool(b))
        else:raise SyntaxError(f'unknown PTX constant operator {op}')
        return kind,value&mask
    def primary():
        nonlocal depth
        depth+=1
        if depth>128:raise SyntaxError('PTX constant expression exceeds 128 nested operands')
        token=take()
        if token in('+','-','!','~'):
            value=primary()
            if token=='+':result=value
            elif token=='-':result=('f',value[1]^(1<<63))if value[0]=='f'else(value[0],(-value[1])&mask)
            elif token=='!':require_int(value);result='s',int(value[1]==0)
            else:require_int(value);result='u',value[1]^mask
        elif token=='(':
            if peek()in('.s64','.u64'):
                cast=take()
                if take()!=')':raise SyntaxError('malformed PTX constant cast')
                value=primary();require_int(value);result=cast[1],value[1]
            else:
                result=expression(0)
                if take()!=')':raise SyntaxError('unbalanced PTX constant expression')
        elif token is None:raise SyntaxError('missing PTX constant operand')
        elif re.fullmatch(r'0[dD][0-9a-fA-F]{16}',token):result='f',int(token[2:],16)
        elif re.fullmatch(r'0[fF][0-9a-fA-F]{8}',token):raise SyntaxError('raw32 floating literals cannot enter PTX constant expressions')
        elif _DECIMAL_FLOAT.fullmatch(token):result='f',_decimal_float(token).bits
        elif re.fullmatch(r'(?:0[xX][0-9a-fA-F]+|0[bB][01]+|[0-9]+)U?',token):
            spelling=token.rstrip('U')
            base=16 if spelling.lower().startswith('0x')else 2 if spelling.lower().startswith('0b')else 8 if len(spelling)>1 and spelling.startswith('0')else 10
            try:bits=int(spelling,base)
            except ValueError:raise SyntaxError(f'invalid PTX integer literal: {token}')from None
            result=('u'if token.endswith('U')or bits>>63 else's'),bits&mask
        else:raise SyntaxError(f'invalid PTX constant operand: {token}')
        depth-=1;return result
    def expression(minimum):
        left=primary()
        while peek()in _CONSTANT_PRECEDENCE and _CONSTANT_PRECEDENCE[peek()]>=minimum:
            op=take();right=expression(_CONSTANT_PRECEDENCE[op]+1);left=binary(op,left,right)
        if minimum==0 and peek()=='?':
            take();require_int(left);yes=expression(0)
            if take()!=':':raise SyntaxError('missing colon in PTX constant conditional')
            no=expression(0)
            if(yes[0]=='f')!=(no[0]=='f'):raise SyntaxError('PTX constant conditional has incompatible types')
            if yes[0]=='f':raise SyntaxError('floating branches of PTX constant conditionals are unsupported by the validated toolkit')
            if yes[0]!=no[0]:raise SyntaxError('mixed signed/unsigned PTX conditional branches require reconciled type semantics')
            chosen=yes if left[1]else no
            left=chosen if chosen[0]=='f'else('u'if'u'in(yes[0],no[0])else's',chosen[1])
        return left
    result=expression(0)
    if cursor!=len(tokens):raise SyntaxError(f'trailing PTX constant expression token: {peek()}')
    return ImmF(result[1],64)if result[0]=='f'else ImmI(integer(result))


def parse_operand(t: str) -> Operand:
    t = t.strip().replace("+-", "-")
    if t.startswith("{"):
        return Vec(tuple(None if e == "_" else parse_operand(e) for e in _split_top(t[1:-1])))
    if t.startswith("["):
        inner = t[1:-1].replace(" ", "")
        m = re.fullmatch(r"([%\w$.]+)(?:([+-])(0x[0-9a-fA-F]+|\d+))?", inner)
        if not m:
            raise SyntaxError(f"address {t}")
        off = int(m[3], 0) if m[3] else 0
        return Addr(m[1], -off if m[2] == "-" else off)
    if re.match(r'(?:[0-9(]|\.[0-9]|[~!+-]\s*(?:[0-9(]|\.[0-9]|[~!+-]))',t):
        return _constant_expression(t)
    if "|" in t:
        a, b = t.split("|")
        return PairDst(Var(a.strip()), Var(b.strip()))
    if re.fullmatch(r"0[fF][0-9a-fA-F]{8}", t):
        return ImmF(int(t[2:], 16), 32)
    if re.fullmatch(r"0[dD][0-9a-fA-F]{16}", t):
        return ImmF(int(t[2:], 16), 64)
    if re.fullmatch(r"-?(0[xX][0-9a-fA-F]+|\d+)U?", t):
        return ImmI(int(t.rstrip("U"), 0))
    if _DECIMAL_FLOAT.fullmatch(t):
        return _decimal_float(t)
    return Var(t)


def parse_instr(stmt: str, line: int) -> Instr:
    guard = None
    if stmt.startswith("@"):
        g, stmt = stmt.split(None, 1)
        neg = g.startswith("@!")
        guard = (g[2:] if neg else g[1:], neg)
    head, _, rest = stmt.partition(" ")
    if "\t" in head:
        head, _, r2 = head.partition("\t")
        rest = r2 + " " + rest
    op, *mods = head.split(".")
    args = [parse_operand(a) for a in _split_top(rest)] if rest.strip() else []
    args = _literal_operands(op,mods,args,line)
    return Instr(op, tuple(mods), args, guard, line)


def _decl(m: re.Match) -> VarDecl:
    space, align, ty, name, count = m.groups()
    size = _TYPE_SIZE[ty] * (int(count) if count else 1)
    return VarDecl(space, name, size, int(align) if align else _TYPE_SIZE[ty])


def parse_module(src: str) -> Module:
    src = _strip_comments(src)
    kernels: dict[str, Kernel] = {}
    module_decls: list[VarDecl] = []
    pos = 0
    for name, raw_params, header, body, start, end in entry_definitions(src):
        for dm in _VAR_DECL.finditer(src[pos:start]):
            module_decls.append(_decl(dm))
        if name in kernels:
            raise SyntaxError(f"duplicate entry definition {name}")
        body_start = end - len(body) - 1
        attributes={}
        kernel = _parse_body(name, _parameters(raw_params, name,attributes), body,
                             first_line=src.count('\n', 0, body_start) + 1)
        kernel.parameter_attributes=attributes
        kernel.directives = _directives(header, name)
        kernels[name] = kernel
        pos = end
    return Module(kernels, module_decls)


def _parse_body(name: str, params, body: str, *, first_line: int = 1) -> Kernel:
    k = Kernel(name, params, [], {})
    offset = 0
    for raw in body.split(";"):
        raw_start = offset
        offset += len(raw) + 1
        stmt = " ".join(raw.split())
        # Peel labels ("L1: add ...", possibly several).
        while (lm := re.match(r"^([A-Za-z_$][\w$]*):\s*(.*)$", stmt)):
            label, stmt = lm[1], lm[2]
            if stmt.startswith(".branchtargets"):
                k.branchtargets[label] = [t.strip() for t in stmt.split(None, 1)[1].split(",")]
                stmt = ""
                break
            k.labels[label] = len(k.body)
        if not stmt or stmt in ("{", "}"):
            continue
        directive = stmt.split(None, 1)[0]
        if directive in (".reg", ".pragma", ".loc", ".file") or (stmt.startswith("{") and stmt.endswith("}")):
            continue
        if dm := _VAR_DECL.match(stmt):
            k.decls.append(_decl(dm))
            continue
        if stmt.startswith("."):
            continue
        # Labels/directives may precede an instruction in the same segment.
        # Find its original spelling before whitespace normalization.
        head = stmt.split()[0]
        relative = raw.find(head)
        line = first_line + body.count('\n', 0, raw_start + max(0, relative))
        k.body.append(parse_instr(stmt, line))
    return k

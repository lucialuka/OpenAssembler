"""Candidate typed PTX constant parser, emitted into the production parser."""
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
            chosen=yes if left[1]else no
            left=chosen if chosen[0]=='f'else('u'if'u'in(yes[0],no[0])else's',chosen[1])
        return left
    result=expression(0)
    if cursor!=len(tokens):raise SyntaxError(f'trailing PTX constant expression token: {peek()}')
    return ImmF(result[1],64)if result[0]=='f'else ImmI(integer(result))

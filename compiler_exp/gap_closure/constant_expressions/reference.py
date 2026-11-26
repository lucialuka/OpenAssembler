"""Generated typed trees and rational reference independent of the parser."""
from fractions import Fraction
import random
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'float_literals'))
from rounding_probe import rounded_bits,rational

MASK=(1<<64)-1


def signed(bits):return bits if bits<1<<63 else bits-(1<<64)


def interpret(node):
    op,*args=node
    if op=='literal':return args[0],args[1]
    if op in('cast_s','cast_u'):
        kind,bits=interpret(args[0]);assert kind!='f';return op[-1],bits
    if op in('neg','not','complement','positive'):
        kind,bits=interpret(args[0])
        if op=='positive':return kind,bits
        if op=='neg':return kind,bits^(1<<63)if kind=='f'else(-bits)% (1<<64)
        if op=='not':return's',int(bits==0)
        return'u',MASK-bits
    if op=='select':
        cond=interpret(args[0]);yes,no=interpret(args[1]),interpret(args[2]);assert cond[0]!='f'
        choice=yes if cond[1]else no
        return('u'if'u'in(yes[0],no[0])else's'),choice[1]
    left,right=interpret(args[0]),interpret(args[1]);kind='u'if'u'in(left[0],right[0])else's'
    if left[0]=='f':
        a,b=rational(left[1],64),rational(right[1],64)
        if op in('<','<=','>','>=','==','!='):
            result={'<':a<b,'<=':a<=b,'>':a>b,'>=':a>=b,'==':a==b,'!=':a!=b}[op]
            return's',int(result)
        value={'+':lambda:a+b,'-':lambda:a-b,'*':lambda:a*b,'/':lambda:a/b}[op]()
        return'f',rounded_bits(value,64)
    a,b=(left[1],right[1])if kind=='u'else(signed(left[1]),signed(right[1]))
    if op in('<','<=','>','>=','==','!='):
        return's',int({'<':a<b,'<=':a<=b,'>':a>b,'>=':a>=b,'==':a==b,'!=':a!=b}[op])
    if op in('&&','||'):return's',int((left[1]!=0 and right[1]!=0)if op=='&&'else(left[1]!=0 or right[1]!=0))
    if op in('<<','>>'):
        value=left[1]if left[0]=='u'else signed(left[1]);factor=1<<(right[1]%64)
        result=value*factor if op=='<<'else value//factor
        return left[0],result%(1<<64)
    if op=='%':return's',left[1]%right[1]
    if op=='/':
        quotient=Fraction(a,b);result=int(quotient)
    elif op=='+':result=a+b
    elif op=='-':result=a-b
    elif op=='*':result=a*b
    else:result={'&':left[1]&right[1],'|':left[1]|right[1],'^':left[1]^right[1]}[op]
    return kind,result%(1<<64)


def render(node):
    op,*args=node
    if op=='literal':
        kind,bits=args
        if kind=='f':return f'0d{bits:016x}'
        if kind=='u':return f'0x{bits:x}U'
        return str(signed(bits))
    if op=='select':return'('+render(args[0])+' ? '+render(args[1])+' : '+render(args[2])+')'
    if op in('cast_s','cast_u'):return'((.'+op[-1]+'64) '+render(args[0])+')'
    if op in('neg','positive','complement','not'):return'('+{'neg':'-','positive':'+','complement':'~','not':'!'}[op]+render(args[0])+')'
    return'('+render(args[0])+' '+op+' '+render(args[1])+')'


def generated(seed=6029,count=4000):
    rng=random.Random(seed)
    def integer(depth):
        if depth==0 or rng.randrange(4)==0:
            bits=rng.getrandbits(64);return('literal','u'if bits>>63 or rng.randrange(3)==0 else's',bits)
        choice=rng.randrange(5)
        if choice==0:return(rng.choice(('neg','positive','not','complement','cast_s','cast_u')),integer(depth-1))
        if choice==1:
            cast=rng.choice(('cast_s','cast_u'))
            return('select',integer(depth-1),(cast,integer(depth-1)),(cast,integer(depth-1)))
        op=rng.choice(('+','-','*','/','%','<<','>>','&','|','^','&&','||','<','>','<=','>=','==','!='))
        left,right=integer(depth-1),integer(depth-1)
        if op in('/','%')and interpret(right)[1]==0:right=('literal','s',1)
        node=op,left,right
        return(rng.choice(('cast_s','cast_u')),node)if op=='%'else node
    def floating(depth):
        if depth==0 or rng.randrange(4)==0:
            value=Fraction(rng.randrange(-100000,100001),1<<rng.randrange(0,30))
            return'literal','f',rounded_bits(value,64)
        op=rng.choice(('+','-','*','/'));left,right=floating(depth-1),floating(depth-1)
        if op=='/'and rational(interpret(right)[1],64)==0:right=('literal','f',0x3ff0000000000000)
        return op,left,right
    for index in range(count):
        node=integer(3)if index%2==0 else floating(3)
        yield dict(name=f'generated_{index}',expression=render(node),tree=node,kind=interpret(node)[0],bits=interpret(node)[1])

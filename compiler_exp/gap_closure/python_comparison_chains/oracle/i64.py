from openassembler.pyfront import *
@kernel
def k(a:Array(i64),out:Array(i32)):
    i=tid_x()
    x=a[i]
    value=0
    if i64(-2)<x<i64(3):
        value+=1
    if i64(-2)!=x!=i64(3):
        value+=2
    if not (i64(-2)<=x<=i64(3)):
        value+=4
    step=0
    while 0<=step<3 and i64(-2)<=x<=i64(3):
        step+=1
        value+=8
    out[i]=value

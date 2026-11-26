from openassembler.pyfront import *
@kernel
def k(a:Array(f32),out:Array(i32)):
    i=tid_x()
    x=a[i]
    value=0
    if -2.0<x<3.0:
        value+=1
    if -2.0!=x!=3.0:
        value+=2
    if not (-2.0<=x<=3.0):
        value+=4
    step=0
    while 0<=step<3 and -2.0<=x<=3.0:
        step+=1
        value+=8
    out[i]=value

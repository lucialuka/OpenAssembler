from openassembler.pyfront import *
@kernel
def k(a:Array(u32),out:Array(i32)):
    i=tid_x()
    x=a[i]
    value=0
    if u32(0)<x<u32(3):
        value+=1
    if u32(0)!=x!=u32(3):
        value+=2
    if not (u32(0)<=x<=u32(3)):
        value+=4
    step=0
    while 0<=step<3 and u32(0)<=x<=u32(3):
        step+=1
        value+=8
    out[i]=value

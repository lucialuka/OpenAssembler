"""Original unsigned CUDA source and independent whole-product hardware reference."""
import struct
from pathlib import Path
from fixtures import cases,values,expected as ptx_expected,SENTINEL,CHOICE

def expanded_cases():
 yield from cases()
 yield dict(name='cuda_multiply',mode='cuda',entry='multiply',cuda_source=Path(__file__).with_name('cuda.cu').read_text())

def expected(data,case,n,lanes):
 if case['mode']!='cuda':return ptx_expected(data,case,n,lanes)
 output=[SENTINEL]*lanes
 for i in range(n):output[i]=(data[i]*data[i+lanes])&((1<<64)-1)
 return struct.pack('<'+str(lanes)+'Q',*output)

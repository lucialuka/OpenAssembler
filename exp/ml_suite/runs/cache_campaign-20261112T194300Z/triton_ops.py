"""Direct Triton 3.3 operator controls, explicit IEEE FP32 or FP16-RNE operands."""
import math
import triton
import triton.language as tl


@triton.jit
def matrix(A,B,Bias,O,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,
           AR:tl.constexpr,AK:tl.constexpr,BC:tl.constexpr,BK:tl.constexpr,
           SA:tl.constexpr,SB:tl.constexpr,SO:tl.constexpr,
           HALF:tl.constexpr,BIAS:tl.constexpr,TM:tl.constexpr,TN:tl.constexpr,TK:tl.constexpr):
    r=tl.program_id(0)*TM+tl.arange(0,TM)
    c=tl.program_id(1)*TN+tl.arange(0,TN)
    q=tl.arange(0,TK);batch=tl.program_id(2)
    acc=tl.full((TM,TN),0,tl.float32)
    for block in range(triton.cdiv(K,TK)):
        t=block*TK+q
        a=tl.load(A+batch*SA+r[:,None]*AR+t[None,:]*AK,(r[:,None]<M)&(t[None,:]<K),other=0)
        b=tl.load(B+batch*SB+c[None,:]*BC+t[:,None]*BK,(c[None,:]<N)&(t[:,None]<K),other=0)
        if HALF:
            a=a.to(tl.float16);b=b.to(tl.float16)
        acc+=tl.dot(a,b,input_precision='ieee')
    if BIAS:acc+=tl.load(Bias+c,c<N,other=0)[None,:]
    tl.store(O+batch*SO+r[:,None]*N+c[None,:],acc,(r[:,None]<M)&(c[None,:]<N))


@triton.jit
def vector(A,B,C,O,N:tl.constexpr,WIDTH:tl.constexpr,OP:tl.constexpr,SCALAR:tl.constexpr,
           HAS_B:tl.constexpr,HAS_C:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK);mask=i<N
    a=tl.load(A+i,mask,other=0)
    if HAS_B:
        if OP=='channel_mul' or OP=='channel_affine':b=tl.load(B+i%WIDTH,mask,other=0)
        else:b=tl.load(B+i,mask,other=0)
    else:b=SCALAR
    if OP=='relu':v=tl.maximum(a,0)
    elif OP=='relu_backward':v=tl.where(b>0,a,0)
    elif OP=='add':v=a+b
    elif OP=='mul' or OP=='channel_mul':v=a*b
    elif OP=='channel_affine':
        v=a*b
        if HAS_C:v+=tl.load(C+i%WIDTH,mask,other=0)
    elif OP=='copy':v=a
    elif OP=='fill':v=tl.full((BLOCK,),SCALAR,tl.float32)
    elif OP=='sgd':v=a-SCALAR*b
    elif OP=='mse_grad':
        v=(a-b)*SCALAR
        if HAS_C:v*=tl.load(C)
    elif OP=='gelu' or OP=='gelu_backward':
        x=b if OP=='gelu_backward' else a
        u=0.7978845608028654*(x+0.044715*x*x*x)
        e=tl.exp(-2*tl.abs(u))
        sigmoid=tl.where(u>=0,1/(1+e),e/(1+e))
        if OP=='gelu':v=x*sigmoid
        else:
            derivative=sigmoid+2*x*e/((1+e)*(1+e))*0.7978845608028654*(1+3*0.044715*x*x)
            v=a*derivative
    else:v=a
    tl.store(O+i,v,mask)


@triton.jit
def row(A,B,O,ROWS:tl.constexpr,W:tl.constexpr,EPS:tl.constexpr,OP:tl.constexpr,BLOCK:tl.constexpr):
    r=tl.program_id(0);i=tl.arange(0,BLOCK)
    x=tl.load(A+r*W+i,i<W,other=0)
    if OP=='softmax':
        a=tl.where(i<W,x,-float('inf'));e=tl.exp(a-tl.max(a,0));v=e/tl.sum(e,0)
    elif OP=='softmax_backward':
        y=tl.load(B+r*W+i,i<W,other=0);v=y*(x-tl.sum(x*y,0))
    elif OP=='layernorm':
        mean=tl.sum(x,0)/W;z=tl.where(i<W,x-mean,0)
        v=z*tl.rsqrt(tl.sum(z*z,0)/W+EPS)
    else:
        original=tl.load(B+r*W+i,i<W,other=0)
        mean=tl.sum(original,0)/W;z=tl.where(i<W,original-mean,0)
        inverse=tl.rsqrt(tl.sum(z*z,0)/W+EPS);y=z*inverse
        v=inverse*(x-tl.sum(x,0)/W-y*tl.sum(x*y,0)/W)
    tl.store(O+r*W+i,v,i<W)


@triton.jit
def channels(A,B,O,R:tl.constexpr,C:tl.constexpr,PRODUCT:tl.constexpr,
             BR:tl.constexpr,BC:tl.constexpr):
    r=tl.arange(0,BR);c=tl.program_id(0)*BC+tl.arange(0,BC)
    a=tl.load(A+r[:,None]*C+c[None,:],(r[:,None]<R)&(c[None,:]<C),other=0)
    if PRODUCT:
        b=tl.load(B+r[:,None]*C+c[None,:],(r[:,None]<R)&(c[None,:]<C),other=0)
        a*=b
    tl.store(O+c,tl.sum(a,0),c<C)


@triton.jit
def loss_partial(A,B,P,N:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    a=tl.load(A+i,i<N,other=0);b=tl.load(B+i,i<N,other=0)
    d=a-b;tl.store(P+tl.program_id(0),tl.sum(d*d,0))


@triton.jit
def loss_final(P,O,PARTS:tl.constexpr,N:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.arange(0,BLOCK);v=tl.load(P+i,i<PARTS,other=0)
    tl.store(O,tl.sum(v,0)/N)


@triton.jit
def permutation(A,O,N:tl.constexpr,D0:tl.constexpr,D1:tl.constexpr,D2:tl.constexpr,D3:tl.constexpr,
                S0:tl.constexpr,S1:tl.constexpr,S2:tl.constexpr,S3:tl.constexpr,
                A0:tl.constexpr,A1:tl.constexpr,A2:tl.constexpr,A3:tl.constexpr,RANK:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK);source=tl.full((BLOCK,),0,tl.int32)
    if RANK==2:
        c0=i//D1;c1=i%D1;source=c0*S0+c1*S1
    elif RANK==3:
        c0=i//(D1*D2);c1=(i//D2)%D1;c2=i%D2
        if A0==0:source+=c0*S0
        elif A0==1:source+=c0*S1
        else:source+=c0*S2
        if A1==0:source+=c1*S0
        elif A1==1:source+=c1*S1
        else:source+=c1*S2
        if A2==0:source+=c2*S0
        elif A2==1:source+=c2*S1
        else:source+=c2*S2
    else:
        c0=i//(D1*D2*D3);c1=(i//(D2*D3))%D1;c2=(i//D3)%D2;c3=i%D3
        if A0==0:source+=c0*S0
        elif A0==1:source+=c0*S1
        elif A0==2:source+=c0*S2
        else:source+=c0*S3
        if A1==0:source+=c1*S0
        elif A1==1:source+=c1*S1
        elif A1==2:source+=c1*S2
        else:source+=c1*S3
        if A2==0:source+=c2*S0
        elif A2==1:source+=c2*S1
        elif A2==2:source+=c2*S2
        else:source+=c2*S3
        if A3==0:source+=c3*S0
        elif A3==1:source+=c3*S1
        elif A3==2:source+=c3*S2
        else:source+=c3*S3
    tl.store(O+i,tl.load(A+source,i<N,other=0),i<N)


def execute(op,xs,out,attrs,math_mode,scratch,lr=.01):
    count=out.numel();shape=tuple(out.shape)
    if op in ('linear','linear_dx','linear_dw','bmm','bmm_dx','bmm_dw'):
        a,b=xs[:2];bias=xs[2] if len(xs)==3 else out
        if op=='linear':m,k=a.shape;n=b.shape[0];ar,ak,bc,bk=k,1,k,1
        elif op=='linear_dx':
            # dX = dY[M,N] @ W[N,K].  Output width is W.K, while the
            # reduction axis is dY.N.  Treat W as [K,N] through strides.
            m=a.shape[0];n=b.shape[1];k=a.shape[1];ar,ak,bc,bk=k,1,1,n
        elif op=='linear_dw':
            m,n=a.shape[1],b.shape[1];k=a.shape[0]
            # A=dY[k,m] viewed as A^T[m,k], B=X[k,n].
            ar,ak,bc,bk=1,m,1,n
        elif op=='bmm':
            batch,m,k=a.shape;n=b.shape[-1];ar,ak,bc,bk=k,1,1,n
        elif op=='bmm_dx':
            # dX = dY[B,M,N] @ W[B,K,N]^T -> [B,M,K].
            batch,m,k=a.shape;n=b.shape[-2]
            # Batched weights are stored [K,N], so output column c and
            # reduction index t address W[c,t].
            ar,ak,bc,bk=k,1,b.shape[-1],1
        else:
            # dB = X.T @ dY, with X supplying logical A.
            a,b=b,a
            batch=a.shape[0];m=a.shape[-1];k=a.shape[-2];n=b.shape[-1]
            ar,ak,bc,bk=1,m,1,n
        batch=a.shape[0] if a.ndim==3 else 1
        sa=a.shape[-2]*a.shape[-1] if batch>1 else 0
        sb=b.shape[-2]*b.shape[-1] if batch>1 else 0
        matrix[(triton.cdiv(m,32),triton.cdiv(n,32),batch)](a,b,bias,out,m,n,k,ar,ak,bc,bk,sa,sb,m*n,
            math_mode=='f16_rne_f32_accum' and op.startswith('linear'),len(xs)==3,32,32,32,num_warps=4)
    elif op in ('softmax','softmax_backward','layernorm','layernorm_backward'):
        w=shape[-1];row[(count//w,)](xs[0],xs[1] if len(xs)>1 else out,out,count//w,w,attrs.get('epsilon',1e-5),op,
                                      triton.next_power_of_2(w),num_warps=4,enable_fp_fusion=False)
    elif op in ('bias_grad','channel_sum','channel_grad'):
        c=shape[0];r=xs[0].numel()//c
        channels[(triton.cdiv(c,32),)](xs[0],xs[1] if len(xs)>1 else out,out,r,c,op=='channel_grad',
                                     triton.next_power_of_2(r),32,num_warps=4,enable_fp_fusion=False)
    elif op=='mse':
        n=xs[0].numel();parts=triton.cdiv(n,1024);temp=scratch[:parts]
        loss_partial[(parts,)](xs[0],xs[1],temp,n,1024,num_warps=4,enable_fp_fusion=False)
        loss_final[(1,)](temp,out,parts,n,triton.next_power_of_2(parts),num_warps=4,enable_fp_fusion=False)
    elif op in ('permute','transpose2d'):
        axes=attrs['axes'] if op=='permute' else [1,0]
        strides=tuple(math.prod(xs[0].shape[i+1:]) for i in range(xs[0].ndim))
        dims=tuple(shape)+(1,)*(4-len(shape));stride4=tuple(strides)+(1,)*(4-len(strides));axis4=tuple(axes)+(0,)*(4-len(axes))
        permutation[(triton.cdiv(count,256),)](xs[0],out,count,*dims,*stride4,*axis4,len(shape),256,num_warps=4)
    else:
        scalar=lr if op=='sgd' else (2/count if op=='mse_grad' else attrs.get('scalar',attrs.get('value',1.0)))
        vector[(triton.cdiv(count,256),)](xs[0] if xs else out,xs[1] if len(xs)>1 else out,xs[2] if len(xs)>2 else out,
            out,count,shape[-1] if shape else 1,op,scalar,len(xs)>1,len(xs)>2,256,num_warps=4,enable_fp_fusion=False)

"""Build research overlay; production is untouched until the gates pass."""
from pathlib import Path
P=Path(__file__).parent/'candidate/openassembler/modelfront'
def edit(name,old,new):
 p=P/name;s=p.read_text()
 if old not in s:raise ValueError((name,old[:100]))
 p.write_text(s.replace(old,new,1))
edit('kernels.py','def shape_of(spec):','def shape_of(spec, *, dtypes=("f32", "float32")):')
edit('kernels.py','spec.get("dtype", "f32") not in ("f32", "float32")','spec.get("dtype", "f32") not in dtypes')
edit('kernels.py','    from .kernels_concat import CONCAT_OPS, emit_concat_kernel','    from .kernels_semantics import SEMANTIC_OPS, emit_semantic_kernel\n    if op in SEMANTIC_OPS:\n        return emit_semantic_kernel(op, inputs, output, attrs)\n    from .kernels_concat import CONCAT_OPS, emit_concat_kernel')
edit('kernels.py','if shape_of(specs[name]) != shape_of(spec):','if specs[name].get("dtype", "f32") != spec.get("dtype", "f32") or shape_of(specs[name], dtypes=("f32","i64","bool")) != shape_of(spec, dtypes=("f32","i64","bool")):')
edit('kernels.py','    for spec in specs.values():\n        shape_of(spec)','    for spec in specs.values():\n        shape_of(spec, dtypes=("f32","i64","bool"))')
edit('kernels.py','    return specs, steps, kernels','''    from .optimizer import optimizer_options
    options = optimizer_options(plan.get("optimizer", {}))
    if options["name"] != "sgd":
        for name in backward.get("grad_parameters", {}):
            for op in ("adam_m", "adam_v", "adam_update"):
                kernel = emit_kernel(op, [specs[name]]*(4 if op == "adam_update" else 3), specs[name], options)
                kernels[kernel.name] = kernel
    if plan.get("loss", "mse") == "cross_entropy":
        if len(plan["outputs"]) != 1:
            raise ValueError("cross_entropy training requires one logits output")
        output = specs[plan["outputs"][0]]
        dims = shape_of(output)
        if len(dims) != 2:
            raise ValueError("cross_entropy expects rank-two [rows, classes] logits")
        labels = {"shape": [dims[0]], "dtype": "i64"}
        for op, args, result in (("ce_rows", [output, labels], {"shape": [dims[0]]}),
                                 ("ce_grad", [output, labels], output)):
            kernel = emit_kernel(op, args, result)
            kernels[kernel.name] = kernel
        kernel = emit_kernel("mean", [{"shape": [dims[0]]}], {"shape": []})
        kernels[kernel.name] = kernel
    elif plan.get("loss", "mse") != "mse":
        raise ValueError("unsupported native loss policy")
    return specs, steps, kernels''')
edit('graph.py','or tensor.dtype != torch.float32 or tensor.layout != torch.strided','or tensor.dtype not in ((torch.float32, torch.int64, torch.bool) if label == "input" else (torch.float32,)) or tensor.layout != torch.strided')
edit('graph.py','        functional.layer_norm: "layernorm",','        functional.layer_norm: "layernorm",\n        functional.embedding: "embedding", functional.scaled_dot_product_attention: "sdpa",')
edit('graph.py','torch.nn.GELU, torch.nn.LayerNorm)','torch.nn.GELU, torch.nn.LayerNorm, torch.nn.Embedding)')
edit('graph.py','meta.dtype != torch.float32','meta.dtype not in (torch.float32, torch.int64, torch.bool)')
edit('graph.py','    def buffer(name, dims):\n        plan["buffers"][name] = {"shape": list(dims), "dtype": "f32"}','    def buffer(name, dims, dtype="f32"):\n        plan["buffers"][name] = {"shape": list(dims), "dtype": dtype}')
edit('graph.py','name = buffer(node.name, shape(node))','name = buffer(node.name, shape(node), {torch.float32:"f32", torch.int64:"i64", torch.bool:"bool"}[node.meta["tensor_meta"].dtype])')
edit('graph.py','"linear" if isinstance(module, torch.nn.Linear) else','"embedding" if isinstance(module, torch.nn.Embedding) else\n              "linear" if isinstance(module, torch.nn.Linear) else')
edit('graph.py','        if op == "concat":','''        if op == "embedding":
            x = tensor(node.args[0] if node.args else node.kwargs.get("input"), node)
            if module is not None:
                w = parameter(node.target + ".weight", node)
                padding, max_norm, sparse, frequency = module.padding_idx, module.max_norm, module.sparse, module.scale_grad_by_freq
            else:
                get = lambda i, k, default=None: node.args[i] if len(node.args)>i else node.kwargs.get(k, default)
                w = tensor(get(1, "weight"), node)
                padding, max_norm = get(2,"padding_idx"), get(3,"max_norm")
                frequency, sparse = get(5,"scale_grad_by_freq",False), get(6,"sparse",False)
            if max_norm is not None or sparse or frequency or padding is not None:
                fail(node, "embedding requires dense gradients, no padding_idx, max_norm or frequency scaling")
            emit("embedding", [x,w], name)
            domain = {"name": x, "lower": 0, "upper": plan["buffers"][w]["shape"][0]}
            if domain not in plan.setdefault("index_domains", []): plan["index_domains"].append(domain)
        elif op == "sdpa":
            get = lambda i, k, default=None: node.args[i] if len(node.args)>i else node.kwargs.get(k,default)
            q,k,v = [tensor(get(i,key),node) for i,key in enumerate(("query","key","value"))]
            mask,dropout,causal = get(3,"attn_mask"),get(4,"dropout_p",0.0),get(5,"is_causal",False)
            if dropout != 0 or type(causal) is not bool or node.kwargs.get("enable_gqa",False):
                fail(node,"SDPA requires dropout_p=0 and no GQA")
            if mask is not None and causal: fail(node,"SDPA mask and is_causal cannot both be set")
            qs,ks,vs = [plan["buffers"][x]["shape"] for x in (q,k,v)]
            if (not 3<=len(qs)<=4 or len(ks)!=len(qs) or len(vs)!=len(qs)
                    or qs[:-2]!=ks[:-2] or ks[:-2]!=vs[:-2] or qs[-1]!=ks[-1] or ks[-2]!=vs[-2]):
                fail(node,"SDPA requires rank3/4 matching batch axes and compatible head dimensions")
            batch = math.prod(qs[:-2]); length,width,head=qs[-2],ks[-2],qs[-1]
            scale=scalar(node.kwargs.get("scale",head**-0.5),node)
            qv=view([q],fresh("query",[batch,length,head]))
            kv=view([k],fresh("key",[batch,width,head]))
            vv=view([v],fresh("value",[batch,width,vs[-1]]))
            kt=emit("permute",[kv],fresh("kt",[batch,head,width]),{"axes":[0,2,1]})
            scores=emit("bmm",[qv,kt],fresh("scores",[batch,length,width]))
            scaled=emit("mul",[scores],fresh("scaled",[batch,length,width]),{"scalar":scale})
            # Keep original batch rank for mask broadcasting; views never reinterpret dtype.
            sdims=[*qs[:-2],length,width]
            scored=view([scaled],fresh("attention_scores",sdims))
            args=[scored] if mask is None else [scored,tensor(mask,node)]
            probs=emit("masked_softmax",args,fresh("probabilities",sdims),{"causal":causal})
            pv=view([probs],fresh("probabilities3",[batch,length,width]))
            ov=emit("bmm",[pv,vv],fresh("attention_output",[batch,length,vs[-1]]))
            view([ov],name)
        elif op == "concat":''')
edit('graph.py','''                if any(plan["buffers"][x]["shape"] != dims for x in tensors):
                    fail(node, "tensor broadcasting is unsupported; add/mul require identical shapes")
                emit(op, tensors, name)''','''                broadcast = any(plan["buffers"][x]["shape"] != dims for x in tensors)
                emit("broadcast_"+op if broadcast else op, tensors, name)''')
edit('graph.py','        elif op == "softmax":\n            accumulate','''        elif op == "embedding":
            accumulate(args[1], emit("embedding_grad", [args[0],dy], shapes[1]))
        elif op in ("broadcast_add", "broadcast_mul"):
            full = plan["buffers"][out]["shape"]
            for index,arg in enumerate(args):
                g = dy if op == "broadcast_add" else emit("broadcast_mul", [dy,args[1-index]], full)
                accumulate(arg, emit("sum_to", [g], shapes[index]) if shapes[index] != full else g)
        elif op in ("softmax", "masked_softmax"):
            accumulate''')
# The replacement above retains the common softmax_backward body.
P.joinpath('optimizer.py').write_text('''"""Explicit optimizer policy. No precision, epsilon or decay substitutions."""
import math

def optimizer_options(options=None):
    options = dict(options or {})
    if set(options)-{"name","beta1","beta2","epsilon","weight_decay"}:
        raise ValueError("unsupported optimizer options")
    name=options.get("name","sgd")
    if name not in ("sgd","adam","adamw"): raise ValueError("optimizer must be sgd, adam or adamw")
    if name=="sgd":
        if set(options)-{"name"}: raise ValueError("plain SGD accepts no Adam options")
        return {"name":name}
    result={"name":name,"beta1":0.9,"beta2":0.999,"epsilon":1e-8,"weight_decay":0.0 if name=="adam" else 0.01}
    result.update(options)
    for key,value in result.items():
        if key=="name":continue
        if type(value) not in (int,float) or not math.isfinite(value):raise ValueError("optimizer scalars must be finite")
    if not 0<=result["beta1"]<1 or not 0<=result["beta2"]<1 or result["epsilon"]<=0 or result["weight_decay"]<0:
        raise ValueError("invalid Adam betas, epsilon or weight decay")
    return result
''')
print('research overlay staged')

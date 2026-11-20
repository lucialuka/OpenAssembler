"""Explicit optimizer policy. No precision, epsilon or decay substitutions."""
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

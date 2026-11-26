"""Experimental paired FP32-to-FP16 conversion for separate model packs."""
from dataclasses import replace
from hashlib import sha256
from unittest.mock import patch

from openassembler.backend import ra
from openassembler.modelfront import kernels_tensor
from openassembler.modelfront.kernels import canonical
from openassembler.ptx.parse import Var


def pack(shape, transpose, input_name, output_name):
    baseline = original_pack(shape, transpose, input_name, output_name)
    rows, cols = shape
    out_rows, out_cols = (cols, rows) if transpose else (rows, cols)
    signature = {**baseline.signature, "implementation": "native_f16x2"}
    name = "oa_pack_" + sha256(canonical(signature)).hexdigest()[:20]
    pair_cols = out_cols // 2
    if transpose:
        index = f"""div.u32 %row,%pair,{pair_cols}; rem.u32 %col,%pair,{pair_cols};
        mad.lo.u32 %i,%col,{2 * cols},%row;"""
        stride = 4 * cols
    else:
        index = "mul.lo.u32 %i,%pair,2;"
        stride = 4
    # Include the second address explicitly when it does not fit the native
    # global-memory displacement used by the scalar backend.
    ptx = f""".version 8.7
.target sm_80
.address_size 64
.visible .entry {name}(.param .u64 input,.param .u64 output) {{
 .reg .b32 %thread,%cta,%pair,%row,%col,%i,%lo,%hi,%out,%nanlo,%nanhi;
 .reg .b64 %src,%dst,%off,%addr,%second,%packed;
 .reg .f32 %a,%b;
 .reg .pred %done,%pLo,%pHi;
 ld.param.u64 %src,[input]; ld.param.u64 %dst,[output];
 mov.u32 %thread,%tid.x; mov.u32 %cta,%ctaid.x;
 mad.lo.u32 %pair,%cta,64,%thread;
 setp.ge.u32 %done,%pair,{out_rows * pair_cols}; @%done bra done;
 {index}
 mul.wide.u32 %off,%i,4; add.u64 %addr,%src,%off;
 add.u64 %second,%addr,{stride};
 ld.global.f32 %a,[%addr]; ld.global.f32 %b,[%second];
 cvt.rn.f16x2.f32 %out,%b,%a;
 setp.nan.f32 %pLo,%a,%a; setp.nan.f32 %pHi,%b,%b;
 and.b32 %lo,%out,65535; and.b32 %hi,%out,4294901760;
 mov.b32 %nanlo,%a; mov.b32 %nanhi,%b;
 shr.u32 %nanlo,%nanlo,16; and.b32 %nanlo,%nanlo,32768;
 and.b32 %nanhi,%nanhi,2147483648;
 or.b32 %nanlo,%nanlo,32256; or.b32 %nanhi,%nanhi,2113929216;
 selp.b32 %lo,%nanlo,%lo,%pLo; selp.b32 %hi,%nanhi,%hi,%pHi;
 or.b32 %out,%lo,%hi;
 mul.wide.u32 %off,%pair,4; add.u64 %packed,%dst,%off;
 st.global.b32 [%packed],%out;
done: ret;
}}
"""
    return replace(baseline, name=name, ptx=ptx, signature=signature, launch={
        "grid": [(out_rows * pair_cols + 63) // 64, 1, 1], "threads": 64, "smem": 0,
    })


def bounded_pack(shape, transpose, input_name, output_name):
    if tuple(shape) in {(128, 128), (128, 256), (256, 128)}:
        return pack(shape, transpose, input_name, output_name)
    return original_pack(shape, transpose, input_name, output_name)


original_pack = kernels_tensor._pack
original_select = ra.Selector.select


def select(self, ins):
    if (ins.op, ins.mods) != ("cvt", ("rn", "f16x2", "f32")):
        return original_select(self, ins)
    if len(ins.args) != 3 or any(not isinstance(arg, Var) for arg in ins.args):
        raise ra.NotSupported("cvt.rn.f16x2.f32 requires three registers")
    dest, high, low = ins.args
    sources = [self.reg32(high), self.reg32(low)]
    packed = self.new()
    self.emit("F2FP.PACK_AB {d0}, {s0}, {s1}", [packed], sources)
    self.bind(dest.name, packed)


def enable(stack):
    stack.enter_context(patch.object(ra.Selector, "select", select))
    stack.enter_context(patch.object(kernels_tensor, "_pack", bounded_pack))

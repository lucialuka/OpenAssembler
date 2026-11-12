"""Bounded pack-address CSE candidate; never imported by production code.

The second element of a packed pair is one column (or one source row for
transpose) after the first. Reuse its 64-bit pointer with a constant memory
displacement, keeping carry semantics and both exact conversion sequences.
"""
from dataclasses import replace
import hashlib
from openassembler.modelfront.kernels import canonical


def candidate(stage):
    if stage.op != "pack_f16_rne":
        return stage
    rows, cols = stage.signature["shape"]
    transpose = stage.signature["transpose"]
    second = "add.u32 %r5,%r4,1;\n"
    second += (f"mad.lo.u32 %r6,%r5,{cols},%r2;\n" if transpose
               else f"mad.lo.u32 %r6,%r2,{cols},%r5;\n")
    second += "mul.wide.u32 %d2,%r6,4; add.u64 %d3,%d0,%d2;\nld.global.b32 %r7,[%d3];"
    if stage.ptx.count(second) != 1:
        raise ValueError("pack source boundary changed; candidate requires a new audit")
    displacement = cols * 4 if transpose else 4
    if not 0 < displacement < 1 << 23:
        raise ValueError("pack displacement outside encoded memory offset range")
    signature = {**stage.signature, "address_policy": "pair_pointer_reuse"}
    name = "oa_pack_cse_" + hashlib.sha256(canonical(signature)).hexdigest()[:20]
    source = stage.ptx.replace(second, f"ld.global.b32 %r7,[%d3+{displacement}];")
    source = source.replace(f".entry {stage.name}(", f".entry {name}(")
    return replace(stage, name=name, ptx=source, signature=signature)


def recipe_candidate(recipe):
    return replace(recipe, name=recipe.name + "_pack_cse",
                   signature={**recipe.signature, "pack_address_policy": "pair_pointer_reuse"},
                   stages=tuple(candidate(stage) for stage in recipe.stages))

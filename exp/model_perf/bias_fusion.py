"""Candidate model-only bias epilogue preserving FP32 addition after HMMA."""
from dataclasses import replace
from hashlib import sha256
import re
from unittest.mock import patch

from openassembler.modelfront import kernels_tensor
from openassembler.modelfront.kernels import canonical


original_recipe = kernels_tensor.emit_tensor_recipe


def fuse(recipe):
    if recipe.op != 'linear' or recipe.stages[-1].op != 'bias_add':
        return recipe
    gemm = recipe.stages[-2]
    plan = gemm.gemm_plan
    if plan.tile != '64x64' or (plan.m, plan.n, plan.k) not in {
            (128, 128, 128), (128, 256, 128), (128, 128, 256)}:
        return recipe
    signature = {**recipe.signature, 'epilogue': 'fp32_bias_after_accumulation'}
    name = 'oa_tensor_' + sha256(canonical(signature)).hexdigest()[:20]
    entry = name + '_gemm_bias'
    source = gemm.ptx.replace(gemm.name, entry)
    source = source.replace('.param .u64 C) {', '.param .u64 Bias,.param .u64 C) {')
    source = source.replace('ld.param.u64 %d2,[C];', 'ld.param.u64 %d2,[C]; ld.param.u64 %bias,[Bias];')
    source = source.replace('.reg .b32 %r<600>;',
                            '.reg .b64 %bias,%bias_offset,%bias_address; .reg .b32 %bias_col; '
                            '.reg .f32 %bias0,%bias1; .reg .b32 %r<600>;')
    pattern = r'st\.global\.v2\.f32 \[%d4(?:\+(\d+))?\],\{%c(\d+),%c(\d+)\};'

    def epilogue(match):
        displacement = int(match[1] or 0)
        lo, hi = match[2], match[3]
        return (f'add.u32 %bias_col,%r51,{displacement // 4}; '
                'mul.wide.u32 %bias_offset,%bias_col,4; add.u64 %bias_address,%bias,%bias_offset; '
                'ld.global.v2.f32 {%bias0,%bias1},[%bias_address]; '
                f'add.rn.f32 %c{lo},%c{lo},%bias0; add.rn.f32 %c{hi},%c{hi},%bias1; '
                + match[0])

    source, replaced = re.subn(pattern, epilogue, source)
    assert replaced == 16, replaced
    fused = replace(gemm, name=entry, ptx=source, inputs=(*gemm.inputs, 'a2'),
                    signature={**gemm.signature, 'epilogue': signature['epilogue']})
    return replace(recipe, name=name, signature=signature, stages=(*recipe.stages[:-2], fused))


def emit_recipe(*args, **kwargs):
    return fuse(original_recipe(*args, **kwargs))


def enable(stack):
    stack.enter_context(patch.object(kernels_tensor, 'emit_tensor_recipe', emit_recipe))

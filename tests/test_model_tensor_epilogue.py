"""Bias epilogues preserve argument order and the measured admission boundary."""
import pytest

from openassembler.build import generate
from openassembler.modelfront.kernels_tensor import emit_tensor_recipe, MATH_MODE
from openassembler.ptx.parse import parse_module


@pytest.mark.parametrize('m,n,k', [(128, 128, 128), (128, 256, 128), (128, 128, 256)])
def test_bias_is_a_fp32_post_accumulation_operand(m, n, k):
    recipe = emit_tensor_recipe('linear', [{'shape': [m, k]}, {'shape': [n, k]}, {'shape': [n]}],
                                {'shape': [m, n]}, {'math_mode': MATH_MODE})
    assert len(recipe.stages) == 3
    stage = recipe.stages[-1]
    kernel = parse_module(stage.ptx).kernels[stage.name]
    assert [name for _, name in kernel.params] == ['A', 'B', 'Bias', 'C']
    assert stage.inputs[-1] == 'a2' and stage.output == 'output'
    assert recipe.signature['epilogue'] == 'fp32_bias_after_accumulation'
    assert stage.ptx.count('ld.global.v2.f32 {%bias0,%bias1}') == 16
    assert stage.ptx.count('add.rn.f32 %c') == 32
    code = generate(kernel, 'sm_80', 'opt')[0]
    assert sum(text.startswith('FADD ') for text, _ in code) == 32


def test_unmeasured_bias_shape_retains_a_separate_stage():
    recipe = emit_tensor_recipe('linear', [{'shape': [256, 256]}, {'shape': [128, 256]}, {'shape': [128]}],
                                {'shape': [256, 128]}, {'math_mode': MATH_MODE})
    assert recipe.stages[-1].op == 'bias_add'
    assert 'epilogue' not in recipe.signature

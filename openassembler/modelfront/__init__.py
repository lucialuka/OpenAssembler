"""Exact-shape Python model capture with OpenAssembler-owned GPU arithmetic."""
from .graph import ModelGraphError, append_training, capture_model, extract_weights, flatten_outputs


def compile_model(model, example_inputs, output_dir, *, opt_level="opt", math_mode="fp32",
                  storage_views=True, reuse_storage=False, sass_listings=True,tensor_packing='separate'):
    from .autograd import compile_model as compile_native
    return compile_native(model, example_inputs, output_dir, opt_level=opt_level, math_mode=math_mode,
                          storage_views=storage_views, reuse_storage=reuse_storage, sass_listings=sass_listings,tensor_packing=tensor_packing)


def load_model(output_dir):
    from .autograd import load_model as load_native
    return load_native(output_dir)


__all__ = ["ModelGraphError", "capture_model", "extract_weights", "flatten_outputs", "append_training", "compile_model", "load_model"]

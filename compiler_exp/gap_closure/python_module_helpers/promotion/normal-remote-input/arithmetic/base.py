from openassembler.pyfront import device, i64, u64, f32
raise RuntimeError('static compiler must never execute helper module code')

@device
def signed_product(x: i64, y: i64) -> i64:
    return x * y

@device
def unsigned_product(x: u64, y: u64) -> u64:
    return x * y

@device
def affine(x: f32, a: f32, b: f32) -> f32:
    return x * a + b

from openassembler.pyfront import Array, f32, i32, kernel, tid_x, ctaid_x, ntid_x


@kernel
def squared(output: Array(f32), source: Array(f32), n: i32):
    i = ctaid_x() * ntid_x() + tid_x()
    if i < n:
        output[i] = source[i] * source[i]

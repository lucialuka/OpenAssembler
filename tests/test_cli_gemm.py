"""Public tensor compilation, launch contract and exact-integer run harness."""
from __future__ import annotations

import json
from pathlib import Path
import struct

import pytest

from openassembler import oracle
from openassembler.cli import main
from openassembler.gemm import contract_from_source, source_for
from openassembler.gpu.cuda_driver import rand_tensor16
from openassembler.kernels.gemm import GemmPlan


def test_gemm_emit_and_contract_needs_exact_validated_body(tmp_path):
    assert main(["gemm", "--m", "128", "--n", "256", "--k", "128", "--dtype", "bf16",
                 "--out-dir", str(tmp_path), "--emit-only"]) == 0
    source = (tmp_path/".build/generated/gemm.ptx").read_text()
    contract = contract_from_source(source)
    assert contract["launch"] == {"grid": [1, 2, 1], "threads": 128, "smem": 65536}
    assert contract["dtype"] == "bf16"
    assert contract["pointer_mode"] == "legacy"
    assert contract_from_source(source.replace("bar.sync 0;", "", 1)) is None
    assert contract_from_source(source.split("\n", 1)[1]) is None
    assert contract_from_source('// openassembler-gemm-v1 {bad}\n') is None


def test_gemm_rejects_unsupported_shape_before_emitting_files(tmp_path, capsys):
    assert main(["gemm", "--m", "129", "--n", "128", "--k", "128",
                 "--out-dir", str(tmp_path)]) == 1
    assert "multiple of 128" in capsys.readouterr().err
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("dtype", ["f16", "bf16"])
def test_tensor16_smoke_inputs_are_reproducible_exact_small_integers(dtype):
    raw = rand_tensor16(23, 4096, dtype)
    assert raw == rand_tensor16(23, 4096, dtype)
    assert raw != rand_tensor16(29, 4096, dtype)
    assert len(raw) == 8192
    if dtype == "f16":
        values = [v[0] for v in struct.iter_unpack("<e", raw)]
    else:
        values = [struct.unpack("<f", struct.pack("<I", v[0] << 16))[0]
                  for v in struct.iter_unpack("<H", raw)]
    assert set(values) == set(range(-3, 4))


def test_gemm_run_plan_keeps_half_precision_sizes_and_launch(tmp_path):
    from openassembler.runner import RunOptions, build_plan
    g = GemmPlan(128, 256, 128, "bf16")
    source = tmp_path/"gemm.ptx"
    source.write_text(source_for(g))
    for name in ("ours.cubin", "ref.cubin"):
        (tmp_path/name).write_bytes(b"test cubin")
    from openassembler import project
    legacy_contract=contract_from_source(source.read_text())
    legacy_contract.pop("cta_order")
    legacy_contract.pop("epilogue")
    legacy_contract.pop("pointer_mode")
    project.save_manifest(tmp_path, tmp_path, "sm_80", "opt", [{
        "source": str(source), "cubin": "ours.cubin", "reference": "ref.cubin",
        "kernels": [{"name": "gemm", "params": [["u64", x] for x in ("A", "B", "C")],
                     "compiled_by": "openassembler", "gemm": legacy_contract}]}])
    plan, _ = build_plan(RunOptions(out_dir=tmp_path, n=1, reps=0))
    assert len(plan.jobs) == 2
    for job in plan.jobs:
        assert job["buffers"] == {"A": 32768, "B": 65536, "C": 131072}
        assert job["steps"][0][2:5] == ((1, 2, 1), (128, 1, 1), 65536)
        assert job["init_tensor16"]["B"][1:] == (32768, "bf16")
        assert job["outputs"] == {"C": [(0, 131072)]}


def test_fused_gemm_run_plan_allocates_bias_and_residual_without_host_random_fill(tmp_path):
    from openassembler.runner import RunOptions, build_plan
    g=GemmPlan(128,128,128,"f16",epilogue="bias_residual_relu")
    source=tmp_path/"gemm.ptx";source.write_text(source_for(g))
    for name in ("ours.cubin","ref.cubin"):(tmp_path/name).write_bytes(b"test cubin")
    from openassembler import project
    project.save_manifest(tmp_path,tmp_path,"sm_80","opt",[{
        "source":str(source),"cubin":"ours.cubin","reference":"ref.cubin",
        "kernels":[{"name":"gemm","params":[["u64",x] for x in ("A","B","C","Bias","Residual")],
                    "compiled_by":"openassembler","gemm":contract_from_source(source.read_text())}]}])
    plan,_=build_plan(RunOptions(out_dir=tmp_path,n=1,reps=0))
    assert len(plan.jobs)==2
    for job in plan.jobs:
        assert job["buffers"]=={"A":32768,"B":32768,"C":65536,"Bias":512,"Residual":65536}
        assert job["init_rand"]=={}
        assert job["steps"][0][5]==[("buf",name,0) for name in ("A","B","C","Bias","Residual")]
        assert job["outputs"]=={"C":[(0,65536)]}


def test_direct_pointer_plan_round_trips_and_legacy_source_defaults():
    direct = GemmPlan(512, 2048, 512, pointer_mode="direct")
    source = source_for(direct)
    assert contract_from_source(source)["pointer_mode"] == "direct"
    from openassembler.kernels.gemm import generate
    assert generate(direct) == generate(GemmPlan(512, 2048, 512))
    header, body = source_for(GemmPlan(128, 128, 128)).split("\n", 1)
    # Existing generated sources predate this policy option.
    from openassembler.gemm import MARKER
    old = json.loads(header[len(MARKER):])
    old.pop("pointer_mode")
    assert contract_from_source(MARKER + json.dumps(old) + "\n" + body)["pointer_mode"] == "legacy"


@pytest.mark.parametrize("options", [
    {"schedule": "combined"}, {"cta_order": "morton8"}, {"epilogue": "bias_residual"},
])
def test_direct_pointer_mode_does_not_admit_unmeasured_combinations(options):
    with pytest.raises(ValueError, match="direct pointer mode is validated"):
        GemmPlan(1024, 1024, 1024, pointer_mode="direct", **options)


def test_pointer_advance_auto_is_shape_specific_and_round_trips():
    from openassembler.kernels.gemm import generate
    wide = GemmPlan(1024, 4096, 1024, pointer_advance="auto")
    tall = GemmPlan(16384, 4096, 4096, pointer_advance="auto")
    assert generate(wide) != generate(GemmPlan(1024, 4096, 1024, pointer_advance="late"))
    assert generate(tall) == generate(GemmPlan(16384, 4096, 4096, pointer_advance="late"))


@pytest.mark.parametrize('dtype,expected', [('f16', b'\x00\x3c'), ('bf16', b'\x80\x3f')])
def test_raw_tensor_spec_uses_half_precision_buffers(dtype, expected, tmp_path):
    from openassembler import project
    from openassembler.runner import RunOptions, build_plan
    src = tmp_path / 'raw.ptx'
    src.write_text('raw tensor source')
    (tmp_path / 'ours.cubin').write_bytes(b'ours')
    (tmp_path / 'ref.cubin').write_bytes(b'reference')
    project.save_manifest(tmp_path, tmp_path, 'sm_80', 'opt', [{
        'source': str(src), 'cubin': 'ours.cubin', 'reference': 'ref.cubin',
        'kernels': [{'name': 'raw', 'params': [['u64', 'A'], ['u64', 'B']],
                     'tensor': {'architecture': 'sm_80'}}]}])
    spec = tmp_path / 'launch.json'
    spec.write_text(json.dumps({'buffers': {
        'A': {'count': 3, 'dtype': dtype, 'init': [1.0]},
        'B': {'count': 7, 'dtype': dtype, 'init': 'random'}},
        'steps': [{'kernel': 'raw', 'block': 32, 'args': [{'buf': 'A', 'offset': 2}, {'buf': 'B'}]}]}))
    plan, _ = build_plan(RunOptions(out_dir=tmp_path, spec=spec))
    for job in plan.jobs:
        assert job['buffers'] == {'A': 6, 'B': 14}
        assert job['init']['A'] == expected
        assert job['init_tensor16']['B'][1:] == (7, dtype)
        assert job['outputs']['B'] == [(0, 14)]


def test_bf16_spec_values_use_round_to_nearest_even(tmp_path):
    from openassembler.emu.fp import BF16, F32, decode, encode
    # Exactly halfway between adjacent BF16 values; both tie directions.
    for value, expected in [(1.00390625, 0x3f80), (1.01171875, 0x3f82), (-0.0, 0x8000)]:
        bits = struct.unpack('<I', struct.pack('<f', value))[0]
        assert encode(decode(bits, F32), BF16) == expected


def test_uncontracted_tensor_kernel_never_uses_scalar_launch_guesses(tmp_path):
    from openassembler.runner import Plan, Target, auto_case
    plan = Plan()
    target = Target({"source": "unannotated.ptx"},
                    {"name": "gemm", "tensor": {"architecture": "sm_80"}}, Path("unannotated.ptx"))
    auto_case(target, tmp_path, 1024, 20, plan)
    assert not plan.jobs
    assert len(plan.cases) == 1 and "--spec" in plan.cases[0].skip


def test_plan_dynamic_shared_optin_preserves_largest_launch_for_repeated_function():
    from openassembler.gpu.cuda_driver import run_plan_job
    class CUDA:
        limit = 49152
        launches = []
        def cuFuncSetAttribute(self, fn, attribute, value):
            assert attribute == 8
            self.limit = value
            return 0
        def cuLaunchKernel(self, fn, gx, gy, gz, bx, by, bz, smem, stream, params, extra):
            assert smem <= self.limit
            self.launches.append(smem)
            return 0
        def cuCtxSynchronize(self):
            return 0
    class FakeDriver:
        cu = CUDA()
        def load(self, data):return "module"
        def unload(self, module):pass
        def function(self, module, kernel):return "function"
        def func_attrs(self, function):return {}
        def ck(self, status, label):assert status == 0, label
    drv = FakeDriver()
    job = {"modules": {"m": b"cubin"}, "buffers": {}, "outputs": {}, "reps": 0,
           "steps": [("m", "k", (1, 1, 1), (128, 1, 1), size, []) for size in (98304, 65536, 0)]}
    run_plan_job(drv, job)
    assert drv.cu.launches == [98304, 65536, 0]
    assert drv.cu.limit == 98304


@pytest.mark.skipif(not all(oracle.available(t) for t in ("ptxas", "nvdisasm")), reason=oracle.NEEDS_TOOLKIT)
def test_gemm_cli_build_and_plain_recompile_use_native_tensor_backend(tmp_path):
    out = tmp_path/"first"
    assert main(["gemm", "--m", "128", "--n", "128", "--k", "128", "--out-dir", str(out)]) == 0
    manifest = json.loads((out/"manifest.json").read_text())
    entry = manifest["entries"][0]
    kernel = entry["kernels"][0]
    assert kernel["compiled_by"] == "openassembler"
    assert kernel["tensor"]["async_scoreboard"] == 0
    assert kernel["gemm"]["schedule"] == "stagepair"
    assert "HMMA.16816.F32" in (out/entry["sass"]).read_text()
    source = out/".build/generated/gemm.ptx"
    again = tmp_path/"again"
    assert main(["compile", str(source), "--out-dir", str(again)]) == 0
    second = json.loads((again/"manifest.json").read_text())["entries"][0]
    assert second["kernels"][0]["gemm"] == kernel["gemm"]
    assert (out/entry["cubin"]).read_bytes() == (again/second["cubin"]).read_bytes()


@pytest.mark.skipif(not all(oracle.available(t) for t in ("ptxas", "nvdisasm")), reason=oracle.NEEDS_TOOLKIT)
def test_gemm_cli_compiles_and_recompiles_morton_fused_contract(tmp_path):
    out=tmp_path/"morton_fused"
    assert main(["gemm","--m","1024","--n","1024","--k","128","--dtype","bf16",
                 "--cta-order","morton8","--epilogue","bias_residual_relu","--out-dir",str(out)])==0
    manifest=json.loads((out/"manifest.json").read_text())
    entry=manifest["entries"][0];kernel=entry["kernels"][0]
    assert kernel["gemm"]["cta_order"]=="morton8"
    assert kernel["gemm"]["epilogue"]=="bias_residual_relu"
    assert [name for _,name in kernel["params"]]==["A","B","C","Bias","Residual"]
    source=out/".build/generated/gemm.ptx"
    assert "ld.global.v2.f32" in source.read_text()
    again=tmp_path/"recompiled"
    assert main(["compile",str(source),"--out-dir",str(again)])==0
    second=json.loads((again/"manifest.json").read_text())["entries"][0]
    assert second["kernels"][0]["gemm"]==kernel["gemm"]
    assert (out/entry["cubin"]).read_bytes()==(again/second["cubin"]).read_bytes()


@pytest.mark.skipif(not all(oracle.available(t) for t in ("ptxas", "nvdisasm")), reason=oracle.NEEDS_TOOLKIT)
def test_direct_pointer_cli_recompile_preserves_native_policy(tmp_path):
    out = tmp_path / "direct"
    assert main(["gemm", "--m", "512", "--n", "2048", "--k", "512",
                 "--pointer-mode", "direct", "--out-dir", str(out)]) == 0
    entry = json.loads((out / "manifest.json").read_text())["entries"][0]
    kernel = entry["kernels"][0]
    assert kernel["compiled_by"] == "openassembler"
    assert kernel["gemm"]["pointer_mode"] == kernel["tensor"]["pointer_mode"] == "direct"
    source = out / ".build/generated/gemm.ptx"
    again = tmp_path / "recompiled"
    assert main(["compile", str(source), "--out-dir", str(again)]) == 0
    second = json.loads((again / "manifest.json").read_text())["entries"][0]
    assert second["kernels"][0]["tensor"] == kernel["tensor"]
    assert second["kernels"][0]["gemm"] == kernel["gemm"]
    assert (out / entry["cubin"]).read_bytes() == (again / second["cubin"]).read_bytes()

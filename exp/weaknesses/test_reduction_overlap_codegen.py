from exp.weaknesses.reduction_overlap_codegen import OverlapPlan, all_plans, generate


def test_overlap_plan_covers_held_out_reduction_shapes():
    plans = all_plans()
    assert len(plans) == 12
    assert {p.width for p in plans} == {512, 513, 1024}
    assert {p.warps for p in plans} == {4, 8}


def test_rmsnorm_overlap_interleaves_two_local_fma_accumulators():
    source, metadata = generate(OverlapPlan("rmsnorm_backward", 513, 4))
    assert metadata["candidate"] == "streamed_load_fma"
    assert "fma.rn.f32 %f64,%f1,%f1,%f64;" in source
    assert "fma.rn.f32 %f96,%f1,%f33,%f96;" in source
    assert "shfl.sync.bfly.b32" in source
    assert "st.global.f32" in source
    assert "mul.f32 %f65,%f1,%f1;" not in source
    assert "mul.f32 %f97,%f1,%f33;" not in source


def test_softmax_overlap_keeps_tail_guard_and_reduction_tree():
    source, metadata = generate(OverlapPlan("softmax_backward", 513, 8))
    assert metadata["local_accumulators"] == ["%f64"]
    assert "@%p1 ld.global.f32" in source
    assert "fma.rn.f32 %f64,%f1,%f33,%f64;" in source
    for delta in (16, 8, 4, 2, 1):
        assert f",{delta},31,0xffffffff;" in source


def test_candidate_is_exp_only_and_does_not_change_production_imports():
    source, _ = generate(OverlapPlan("rmsnorm_backward", 512, 4))
    assert "openassembler.kernels" not in source
    assert ".entry op_rmsnorm_backward_w512_c4_overlap" in source


from exp.shape_coverage.profile_worker import authoritative_metrics, parse_ncu_csv


def test_wide_raw_csv_preserves_units_and_thousands_separators_and_missing_metrics():
    metrics = ["smsp__inst_executed.sum", "dram__bytes.sum", "missing_metric"]
    content = '''==PROF== Connected to process 17
"ID","Process ID","Kernel Name","smsp__inst_executed.sum","dram__bytes.sum"
"","","","inst","byte"
"0","17","gemm","4,370,048","1,024"
OA_ROLE_PROFILE_WORKER_DONE
'''
    result = parse_ncu_csv(content, metrics)
    assert len(result["kernels"]) == 1
    kernel = result["kernels"][0]
    assert kernel["kernel"] == "gemm"
    assert kernel["metrics"][metrics[0]] == {
        "value": 4370048, "unit": "inst", "raw": "4,370,048"}
    assert kernel["metrics"][metrics[1]]["value"] == 1024
    assert kernel["metrics"][metrics[1]]["unit"] == "byte"
    assert kernel["missing_metrics"] == ["missing_metric"]
    assert result["missing_metrics"] == ["missing_metric"]


def test_long_csv_keeps_kernels_separate_and_nonnumeric_values_are_missing_not_zero():
    metrics = ["long_scoreboard", "tensor_active"]
    content = '''"ID","Process ID","Kernel Name","Metric Name","Metric Unit","Metric Value"
"0","17","gemm_main","long_scoreboard","%","77.25"
"0","17","gemm_main","tensor_active","%","N/A"
"1","17","gemm_epilogue","long_scoreboard","%","1.5"
"1","17","gemm_epilogue","tensor_active","%","0"
'''
    result = parse_ncu_csv(content, metrics)
    assert len(result["kernels"]) == 2
    main, epilogue = result["kernels"]
    assert main["metrics"]["long_scoreboard"]["value"] == 77.25
    assert main["metrics"]["long_scoreboard"]["unit"] == "%"
    assert main["metrics"]["tensor_active"]["value"] is None
    assert main["missing_metrics"] == ["tensor_active"]
    assert epilogue["metrics"]["tensor_active"]["value"] == 0
    assert not epilogue["missing_metrics"]
    # A metric seen for one kernel does not repair its absence from another.
    assert result["missing_metrics"] == []
    assert not parse_ncu_csv("==ERROR== ERR_NVGPUCTRPERM", metrics)["kernels"]
    requested, source = authoritative_metrics()
    assert "smsp__inst_executed.sum" in requested
    assert "launch__registers_per_thread" in requested
    assert source["path"] == "exp/a100_counters/run.py"

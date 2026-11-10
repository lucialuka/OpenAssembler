from __future__ import annotations

import pytest

from exp.dispatch_tuning.rowwise_modal import select_cases


CASES = [("softmax", i, i + 1) for i in range(10)]


def test_case_subset_keeps_frozen_matrix_order_and_bounds():
    assert select_cases(CASES, start=2, limit=3) == CASES[2:5]
    assert select_cases(CASES, indices="8,2,5") == [CASES[8], CASES[2], CASES[5]]
    assert select_cases(CASES) == CASES
    assert select_cases(CASES, start=len(CASES)) == []


@pytest.mark.parametrize("kwargs", [
    {"start": -1}, {"start": 11}, {"limit": -1},
    {"indices": "1,1"}, {"indices": "-1"}, {"indices": "10"},
    {"indices": "x"}, {"start": 1, "indices": "2"}, {"limit": 1, "indices": "2"},
])
def test_case_selector_rejects_ambiguous_or_invalid_inputs(kwargs):
    with pytest.raises(ValueError):
        select_cases(CASES, **kwargs)

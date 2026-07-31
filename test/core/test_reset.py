import numpy as np

from npdsp import *


def test_pipeline_reset_calls_all_blocks() -> None:
    a = ResetCounter()
    b = ResetCounter()

    pipeline = Pipeline(a, b)

    pipeline.reset()

    assert a.reset_count == 1
    assert b.reset_count == 1


def test_profile_reset_before_after() -> None:
    block = ResetCounter()

    pipeline = Pipeline(block)

    pipeline.profile(
        np.array([1]),
        reset=True,
    )

    assert block.reset_count == 2
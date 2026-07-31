import numpy as np

from npdsp import *


def test_identity_pipeline(
    pipeline: Pipeline,
) -> None:
    x = np.array([1, 2, 3])

    result = pipeline(x)

    assert np.array_equal(
        result,
        x,
    )


def test_pipeline_order(
    math_pipeline: Pipeline,
) -> None:
    result = math_pipeline(np.array([1, 2]))

    assert np.array_equal(
        result,
        np.array([9, 12]),
    )


def test_pipeline_accepts_list(
    math_pipeline: Pipeline,
) -> None:
    result = math_pipeline([1, 2])

    assert np.array_equal(
        result,
        np.array([9, 12]),
    )


def test_pipeline_is_callable(
    math_pipeline: Pipeline,
) -> None:
    result = math_pipeline([0])

    assert result[0] == 6


def test_pipeline_and_block_profile_same_output(
    math_pipeline: Pipeline,
) -> None:
    x = np.array([1])

    assert np.array_equal(
        math_pipeline(x),
        (math_pipeline.blocks[1](math_pipeline.blocks[0](x))),
    )

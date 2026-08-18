import numpy as np

from npdsp import *


def test_pipeline_can_profile(pipeline: Pipeline) -> None:
    x = np.arange(5)
    p = pipeline.profile(x)

    assert isinstance(p, ProfileResults)


def test_profile_consists_of_multiple_profileresult(pipeline: Pipeline) -> None:
    x = np.arange(5)
    p = pipeline.profile(x)

    are_profileresult = [isinstance(r, ProfileResult) for r in p]  # pyright: ignore[reportUnnecessaryIsInstance]

    assert all(are_profileresult)
    assert len(p) == len(pipeline)


def test_can_convert_result_to_str(pipeline: Pipeline) -> None:
    x = np.arange(5)
    p = pipeline.profile(x)

    assert isinstance(str(p), str)
    assert isinstance(repr(p), str)
    assert isinstance(str(p[0]), str)
    assert isinstance(repr(p[0]), str)

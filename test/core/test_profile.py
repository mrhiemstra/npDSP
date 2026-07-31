import numpy as np

from npdsp import *


def test_profile_returns_profile_results(
    math_pipeline: Pipeline,
) -> None:
    results = math_pipeline.profile(np.array([1]))

    assert isinstance(results, ProfileResults)


def test_profile_length(
    math_pipeline: Pipeline,
) -> None:
    results = math_pipeline.profile(np.array([1]))

    assert len(results) == len(math_pipeline)


def test_profile_names(
    math_pipeline: Pipeline,
) -> None:
    results = math_pipeline.profile(np.array([1]))

    assert results[0].name == "Add"
    assert results[1].name == "Multiply"


def test_profile_elapsed_positive(
    math_pipeline: Pipeline,
) -> None:
    results = math_pipeline.profile(np.array([1]))

    assert all(r.mean_time >= 0 for r in results)
    assert all(r.min_time >= 0 for r in results)
    assert all(r.max_time >= 0 for r in results)


def test_profile_total_time(
    math_pipeline: Pipeline,
) -> None:
    results = math_pipeline.profile(np.array([1]))

    assert results.tottime == sum(r.mean_time for r in results)


def test_profile_string(
    math_pipeline: Pipeline,
) -> None:
    text = str(math_pipeline.profile(np.array([1])))

    assert "Add" in text
    assert "Multiply" in text

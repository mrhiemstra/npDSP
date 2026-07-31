import pytest

from npdsp import *


def test_len(
    pipeline: Pipeline,
) -> None:
    assert len(pipeline) == 3


def test_iteration(
    pipeline: Pipeline,
    tap1: Tap,
    tap2: Tap,
    tap3: Tap,
) -> None:
    assert list(pipeline) == [
        tap1,
        tap2,
        tap3,
    ]


def test_first(
    pipeline: Pipeline,
    tap1: Tap,
) -> None:
    assert pipeline.first is tap1


def test_last(
    pipeline: Pipeline,
    tap3: Tap,
) -> None:
    assert pipeline.last is tap3


def test_index_lookup(
    pipeline: Pipeline,
    tap2: Tap,
) -> None:
    assert pipeline[1] is tap2


def test_negative_index(
    pipeline: Pipeline,
    tap3: Tap,
) -> None:
    assert pipeline[-1] is tap3


def test_name_lookup(
    pipeline: Pipeline,
    tap2: Tap,
) -> None:
    assert pipeline["tap2"] is tap2


def test_slice(
    pipeline: Pipeline,
) -> None:
    result = pipeline[1:3]

    assert isinstance(
        result,
        Pipeline,
    )
    assert len(result) == 2


def test_slice_by_name(
    pipeline: Pipeline,
) -> None:
    result = pipeline["tap2":]

    assert len(result) == 2


def test_slice_until_name(
    pipeline: Pipeline,
) -> None:
    result = pipeline[:"tap2"]

    assert len(result) == 1


def test_slice_include_stop(
    pipeline: Pipeline,
) -> None:
    result = pipeline[:"tap2", ...]

    assert len(result) == 2


def test_slice_step_not_supported(
    pipeline: Pipeline,
) -> None:
    with pytest.raises(NotImplementedError):
        pipeline[::2]


def test_slice_to_end_inclusive(
    pipeline: Pipeline,
) -> None:
    result = pipeline["tap2":, ...]

    assert len(result) == 2


def test_slice_all(
    pipeline: Pipeline,
) -> None:
    result = pipeline[:]

    assert len(result) == len(pipeline)


def test_contains_block(
    pipeline: Pipeline,
    tap1: Tap,
) -> None:
    assert tap1 in pipeline


def test_contains_name(
    pipeline: Pipeline,
) -> None:
    assert "tap2" in pipeline


def test_missing_name_not_contains(
    pipeline: Pipeline,
) -> None:
    assert "missing" not in pipeline

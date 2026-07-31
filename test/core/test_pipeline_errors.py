import pytest

from npdsp import *


def test_missing_name_lookup(
    pipeline: Pipeline,
) -> None:
    with pytest.raises(KeyError):
        pipeline["missing"]


def test_insert_missing_name(
    pipeline: Pipeline,
) -> None:
    with pytest.raises(KeyError):
        pipeline.insert(
            "missing",
            Tap("x"),
        )


def test_remove_missing_name(
    pipeline: Pipeline,
) -> None:
    with pytest.raises(KeyError):
        pipeline.remove("missing")


def test_replace_missing_name(
    pipeline: Pipeline,
) -> None:
    with pytest.raises(KeyError):
        pipeline.replace(
            "missing",
            Tap("x"),
        )


def test_out_of_range(
    pipeline: Pipeline,
) -> None:
    with pytest.raises(IndexError):
        pipeline[100]


def test_invalid_tuple_slice(
    pipeline: Pipeline,
) -> None:
    with pytest.raises(TypeError):
        pipeline[  # type: ignore
            ("tap1", "tap2")
        ]


def test_empty_first() -> None:
    with pytest.raises(IndexError):
        _ = Pipeline().first


def test_empty_last() -> None:
    with pytest.raises(IndexError):
        _ = Pipeline().last

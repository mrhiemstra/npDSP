import pytest

from npdsp import *


def test_empty_pipeline() -> None:
    pipeline = Pipeline()

    assert len(pipeline) == 0


def test_pipeline_constructor(
    tap1: Tap,
    tap2: Tap,
) -> None:
    pipeline = Pipeline(
        tap1,
        tap2,
    )

    assert pipeline.blocks == [
        tap1,
        tap2,
    ]


def test_pipeline_flattens_nested_pipeline(
    tap1: Tap,
    tap2: Tap,
) -> None:
    nested = Pipeline(
        Pipeline(tap1),
        tap2,
    )

    assert nested.blocks == [
        tap1,
        tap2,
    ]


def test_rshift_block(
    tap1: Tap,
    tap2: Tap,
) -> None:
    pipeline = tap1 >> tap2

    assert pipeline.blocks == [
        tap1,
        tap2,
    ]


def test_rshift_pipeline(
    tap1: Tap,
    tap2: Tap,
    tap3: Tap,
) -> None:
    pipeline = (tap1 >> tap2) >> tap3

    assert pipeline.blocks == [
        tap1,
        tap2,
        tap3,
    ]


def test_irshift_block(
    tap1: Tap,
    tap2: Tap,
) -> None:
    pipeline = Pipeline()

    pipeline >>= tap1
    pipeline >>= tap2

    assert pipeline.blocks == [
        tap1,
        tap2,
    ]


def test_irshift_pipeline(
    tap1: Tap,
    tap2: Tap,
) -> None:
    first = Pipeline(tap1)
    second = Pipeline(tap2)

    first >>= second

    assert first.blocks == [
        tap1,
        tap2,
    ]


def test_duplicate_names_raise() -> None:
    with pytest.raises(ValueError, match="Duplicate block name: 'same'"):
        Pipeline(
            Tap("same"),
            Tap("same"),
        )


def test_block_length(
    tap1: Tap,
) -> None:
    assert len(tap1) == 1


def test_pipeline_length(
    pipeline: Pipeline,
) -> None:
    assert len(pipeline) == 3

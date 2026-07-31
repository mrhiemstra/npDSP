import pytest

from npdsp import *


@pytest.fixture
def tap1() -> Tap:
    return Tap("tap1")


@pytest.fixture
def tap2() -> Tap:
    return Tap("tap2")


@pytest.fixture
def tap3() -> Tap:
    return Tap("tap3")


@pytest.fixture
def pipeline(
    tap1: Tap,
    tap2: Tap,
    tap3: Tap,
) -> Pipeline:
    return Pipeline(
        tap1,
        tap2,
        tap3,
    )


@pytest.fixture
def math_pipeline() -> Pipeline:
    return Add(2) >> Multiply(3)

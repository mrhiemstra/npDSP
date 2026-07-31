from npdsp import *


def test_find(
    pipeline: Pipeline,
) -> None:
    block = pipeline.find(Tap)

    assert isinstance(block, Tap)


def test_find_returns_none() -> None:
    class Dummy(Tap):
        pass

    pipeline = Pipeline(Tap("a"))

    assert pipeline.find(Dummy) is None


def test_find_all(
    pipeline: Pipeline,
) -> None:
    blocks = pipeline.find_all(Tap)

    assert len(blocks) == 3


def test_find_all_empty() -> None:
    class Dummy(Tap):
        pass

    pipeline = Pipeline(Tap("a"))

    assert pipeline.find_all(Dummy) == []

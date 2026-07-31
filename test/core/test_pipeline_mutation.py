from npdsp import *


def test_insert_index(
    pipeline: Pipeline,
) -> None:
    block = Tap("new")

    pipeline.insert(
        1,
        block,
    )

    assert pipeline[1] is block


def test_insert_name(
    pipeline: Pipeline,
) -> None:
    block = Tap("new")

    pipeline.insert(
        "tap2",
        block,
    )

    assert pipeline["new"] is block
    assert pipeline["tap2"] is pipeline[2]


def test_remove_index(
    pipeline: Pipeline,
) -> None:
    pipeline.remove(1)

    assert len(pipeline) == 2


def test_remove_name(
    pipeline: Pipeline,
) -> None:
    pipeline.remove("tap2")

    assert "tap2" not in pipeline


def test_remove_reindexes(
    pipeline: Pipeline,
) -> None:
    pipeline.remove("tap1")

    assert pipeline["tap2"] is pipeline[0]


def test_replace_index(
    pipeline: Pipeline,
) -> None:
    block = Tap("new")

    pipeline.replace(
        1,
        block,
    )

    assert pipeline[1] is block


def test_replace_name(
    pipeline: Pipeline,
) -> None:
    block = Tap("new")

    pipeline.replace(
        "tap2",
        block,
    )

    assert "new" in pipeline
    assert "tap2" not in pipeline


def test_setitem(
    pipeline: Pipeline,
) -> None:
    block = Tap("new")

    pipeline["tap2"] = block

    assert pipeline["new"] is block


def test_delitem(
    pipeline: Pipeline,
) -> None:
    del pipeline["tap2"]

    assert "tap2" not in pipeline


def test_replace_updates_names(
    pipeline: Pipeline,
) -> None:
    replacement = Tap("replacement")

    pipeline.replace(
        "tap2",
        replacement,
    )

    assert "replacement" in pipeline
    assert "tap2" not in pipeline


def test_insert_updates_names(
    pipeline: Pipeline,
) -> None:
    block = Tap("inserted")

    pipeline.insert(
        "tap2",
        block,
    )

    assert "inserted" in pipeline


def test_remove_updates_names(
    pipeline: Pipeline,
) -> None:
    pipeline.remove("tap2")

    assert "tap2" not in pipeline

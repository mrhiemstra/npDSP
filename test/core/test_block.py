import pytest

from npdsp import *


def test_block_raises_not_implemented_error() -> None:
    block = Block()
    with pytest.raises(
        NotImplementedError,
        match="The block base class does not have any associated process function",
    ):
        block([0])


def test_block_implements_latency_samples() -> None:
    block = Block()
    block.reset()
    assert block.latency_samples == 0

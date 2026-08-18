import numpy as np
import pytest

import npdsp


def test_sliding_buffer_raises_value_error() -> None:
    with pytest.raises(ValueError, match="size must be non-negative"):
        npdsp.core.SlidingBuffer(-1)


def test_sliding_buffer_initialized() -> None:
    sb = npdsp.core.SlidingBuffer(2)
    assert not sb.initialized

    sb.prepare((1, 1), dtype=np.dtype(np.int32))

    assert sb.initialized


def test_sliding_buffer_shape_value_error() -> None:
    sb = npdsp.core.SlidingBuffer(2)
    sb.prepare((1, 1), dtype=np.dtype(np.int32))
    with pytest.raises(
        ValueError,
        match=r"Leading dimensions cannot change between calls: expected \(1,\), got \(1, 1\)",
    ):
        sb.prepare((1, 1, 1), dtype=np.dtype(np.int32))

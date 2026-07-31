from __future__ import annotations

import numpy as np
import pytest

from npdsp import Delay


@pytest.fixture
def delay() -> Delay:
    return Delay(2)


def test_delay_single_channel(delay: Delay) -> None:
    x = np.array([1, 2, 3])

    y = delay(x)

    np.testing.assert_array_equal(
        y,
        np.array([0, 0, 1]),
    )


def test_delay_single_channel_across_calls(delay: Delay) -> None:
    y1 = delay(np.array([1, 2, 3]))
    y2 = delay(np.array([4, 5, 6]))

    np.testing.assert_array_equal(
        y1,
        np.array([0, 0, 1]),
    )

    np.testing.assert_array_equal(
        y2,
        np.array([2, 3, 4]),
    )


def test_delay_multichannel(delay: Delay) -> None:
    x = np.array([
        [1, 10],
        [2, 20],
        [3, 30],
    ])

    y = delay(x)

    np.testing.assert_array_equal(
        y,
        np.array([
            [0, 0],
            [0, 0],
            [1, 10],
        ]),
    )


def test_delay_multichannel_across_calls(delay: Delay) -> None:
    y1 = delay(np.array([
        [1, 10],
        [2, 20],
        [3, 30],
    ]))

    y2 = delay(np.array([
        [4, 40],
        [5, 50],
        [6, 60],
    ]))

    np.testing.assert_array_equal(
        y1,
        np.array([
            [0, 0],
            [0, 0],
            [1, 10],
        ]),
    )

    np.testing.assert_array_equal(
        y2,
        np.array([
            [2, 20],
            [3, 30],
            [4, 40],
        ]),
    )


def test_delay_zero_samples() -> None:
    delay = Delay(0)

    x = np.array([1, 2, 3])

    y = delay(x)

    np.testing.assert_array_equal(y, x)


def test_delay_preserves_dtype() -> None:
    delay = Delay(2)

    x = np.array([1, 2, 3], dtype=np.float32)

    y = delay(x)

    assert y.dtype == np.float32


def test_delay_reset() -> None:
    delay = Delay(2)

    delay(np.array([1, 2, 3]))

    delay.reset()

    y = delay(np.array([4, 5, 6]))

    np.testing.assert_array_equal(
        y,
        np.array([0, 0, 4]),
    )


def test_delay_negative_samples() -> None:
    with pytest.raises(ValueError):
        Delay(-1)


def test_delay_preserves_shape() -> None:
    delay = Delay(2)

    x = np.array([
        [1, 2, 3],
        [4, 5, 6],
    ])

    y = delay(x)

    assert y.shape == x.shape
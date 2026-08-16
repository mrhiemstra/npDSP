from __future__ import annotations

import numpy as np
import pytest

from npdsp import Downsample, Upsample


@pytest.fixture
def downsample() -> Downsample:
    return Downsample(4)


@pytest.fixture
def upsample() -> Upsample:
    return Upsample(4)


def test_downsample_single_conversion(downsample: Downsample) -> None:
    x = np.array([1, 2, 3, 4])

    y = downsample(x)

    np.testing.assert_array_equal(
        y,
        np.array([1]),
    )


def test_downsample_single_conversion_2(downsample: Downsample) -> None:
    x = np.array([1, 2, 3, 4, 5, 6, 7, 8])

    y = downsample(x)

    np.testing.assert_array_equal(
        y,
        np.array([1, 5]),
    )


def test_downsample_multi_conversion(downsample: Downsample) -> None:
    x = np.array([[1, 2, 3, 4], [1, 2, 3, 4]])

    y = downsample(x)

    np.testing.assert_array_equal(
        y,
        np.array([[1], [1]]),
    )


def test_downsample_multi_conversion2(downsample: Downsample) -> None:
    x = np.array([[1, 2, 3, 4, 5, 6, 7, 8], [9, 10, 11, 12, 13, 14, 15, 16]])

    y = downsample(x)

    np.testing.assert_array_equal(
        y,
        np.array([[1, 5], [9, 13]]),
    )


def test_downsample_stream(downsample: Downsample) -> None:
    x = np.array([1, 2, 3])

    y = downsample(x)

    np.testing.assert_array_equal(
        y,
        np.array([1]),
    )

    x = np.array([4, 5])

    y = downsample(x)

    np.testing.assert_array_equal(
        y,
        np.array([5]),
    )

    assert downsample._offset == 3  # pyright: ignore[reportPrivateUsage] # noqa: SLF001


def test_downsample_reset(downsample: Downsample) -> None:
    assert downsample._offset == 0  # pyright: ignore[reportPrivateUsage] # noqa: SLF001

    x = np.array([1, 2, 3])
    _ = downsample(x)

    assert downsample._offset == 1  # pyright: ignore[reportPrivateUsage] # noqa: SLF001

    downsample.reset()

    assert downsample._offset == 0  # pyright: ignore[reportPrivateUsage] # noqa: SLF001


def test_upsample_single_conversion(upsample: Upsample) -> None:
    x = np.array([1])

    y = upsample(x)

    np.testing.assert_array_equal(
        y,
        np.array([1, 0, 0, 0]),
    )


def test_upsample_single_conversion_2(upsample: Upsample) -> None:
    x = np.array([1, 2, 3, 4])

    y = upsample(x)

    np.testing.assert_array_equal(
        y,
        np.array([1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0]),
    )


def test_upsample_multi_conversion(upsample: Upsample) -> None:
    x = np.array([[1, 2], [3, 4]])

    y = upsample(x)

    np.testing.assert_array_equal(
        y,
        np.array([[1, 0, 0, 0, 2, 0, 0, 0], [3, 0, 0, 0, 4, 0, 0, 0]]),
    )


def test_upsample_stream(upsample: Upsample) -> None:
    x = np.array([1, 2, 3])

    y = upsample(x)

    np.testing.assert_array_equal(
        y,
        np.array([1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0]),
    )

    x = np.array([4, 5])

    y = upsample(x)

    np.testing.assert_array_equal(
        y,
        np.array([4, 0, 0, 0, 5, 0, 0, 0]),
    )

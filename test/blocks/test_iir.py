import numpy as np
import pytest

from npdsp import *

# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_iir_constructor() -> None:
    iir = IIR(
        b=[1, 2, 3],
        a=[1, 4, 5],
    )

    assert np.array_equal(
        iir.b,
        np.array([1, 2, 3]),
    )
    assert np.array_equal(
        iir.a,
        np.array([1, 4, 5]),
    )


def test_iir_name() -> None:
    iir = IIR(
        b=[1, 2],
        a=[1, 3],
        name="iir1",
    )

    assert iir.name == "iir1"


def test_iir_coefficients_are_numpy_arrays() -> None:
    iir = IIR(
        b=[1, 2],
        a=[1, 3],
    )

    assert isinstance(iir.b, np.ndarray)
    assert isinstance(iir.a, np.ndarray)


def test_iir_accepts_numpy_coefficients() -> None:
    b = np.array([1, 2, 3])
    a = np.array([1, 4, 5])

    iir = IIR(b, a)

    assert np.array_equal(iir.b, b)
    assert np.array_equal(iir.a, a)


def test_iir_accepts_per_channel_coefficients() -> None:
    iir = IIR(
        b=[
            [1, 2],
            [3, 4],
        ],
        a=[
            [1, 5],
            [1, 6],
        ],
    )

    assert iir.b.shape == (2, 2)
    assert iir.a.shape == (2, 2)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_iir_empty_b_raises() -> None:
    with pytest.raises(ValueError, match="b cannot be empty"):
        IIR([], [1])


def test_iir_empty_a_raises() -> None:
    with pytest.raises(ValueError, match="a cannot be empty"):
        IIR([1], [])


def test_iir_b_must_be_one_or_two_dimensional() -> None:
    with pytest.raises(ValueError, match="b must be one- or two-dimensional"):
        IIR(
            [
                [[1, 2]],
            ],
            [1],
        )


def test_iir_a_must_be_one_or_two_dimensional() -> None:
    with pytest.raises(ValueError, match="a must be one- or two-dimensional"):
        IIR(
            [1],
            [
                [[1, 2]],
            ],
        )


def test_iir_a0_cannot_be_zero() -> None:
    with pytest.raises(ValueError, match=r"a\[0\] must be non-zero"):
        IIR(
            [1, 2],
            [0, 1],
        )


def test_iir_per_channel_b_and_a_broadcast() -> None:
    b = np.array([[0.2, 0.4, 0.2]])  # 1 row
    a = np.array([[1.0, -0.5, 0.1], [1.0, -0.2, 0.1]])  # 2 channels
    iir = IIR(b, a)
    x = np.random.Generator(np.random.PCG64()).normal(size=(2, 100))

    y = iir(x)
    assert y.shape == (2, 100)


def test_iir_per_channel_mismatch_raises() -> None:
    b = np.array([[0.2, 0.4, 0.2], [0.1, 0.1, 0.1]])  # 2 rows
    a = np.array(
        [[1.0, -0.5, 0.1], [1.0, -0.2, 0.1], [1.0, -0.3, 0.1]]
    )  # 3 channels
    with pytest.raises(
        ValueError,
        match=r"b has 2 channels but a has 3 channels; must match or one must have exactly 1 row for broadcasting",
    ):
        IIR(b, a)


# ---------------------------------------------------------------------------
# Basic behavior
# ---------------------------------------------------------------------------


def test_iir_identity() -> None:
    iir = IIR(
        b=[1],
        a=[1],
    )

    x = np.array([1, 2, 3])

    result = iir(x)

    assert np.array_equal(
        result,
        x,
    )


def test_iir_accepts_list_input() -> None:
    iir = IIR(
        b=[1],
        a=[1],
    )

    result = iir([1, 2, 3])

    assert np.array_equal(
        result,
        np.array([1, 2, 3]),
    )


def test_iir_feed_forward_matches_fir() -> None:
    iir = IIR(
        b=[1, 2, 3],
        a=[1],
    )

    x = np.array([1, 2, 3])

    result = iir(x)

    expected = np.array(
        [
            1,
            4,
            10,
        ]
    )

    assert np.array_equal(
        result,
        expected,
    )


def test_iir_first_order_feedback() -> None:
    iir = IIR(
        b=[1],
        a=[1, -1],
    )

    x = np.array([1, 1, 1])

    result = iir(x)

    expected = np.array(
        [
            1,
            2,
            3,
        ]
    )

    assert np.array_equal(
        result,
        expected,
    )


def test_iir_feedback_with_scaled_a0() -> None:
    iir = IIR(
        b=[2],
        a=[2, -1],
    )

    x = np.array([1, 1, 1])

    result = iir(x)

    expected = np.array(
        [
            1,
            1.5,
            1.75,
        ]
    )

    assert np.allclose(
        result,
        expected,
    )


# ---------------------------------------------------------------------------
# Complex signals and coefficients
# ---------------------------------------------------------------------------


def test_iir_complex_input() -> None:
    iir = IIR(
        b=[1],
        a=[1],
    )

    x = np.array(
        [
            1 + 1j,
            2 + 2j,
            3 + 3j,
        ]
    )

    result = iir(x)

    assert np.array_equal(
        result,
        x,
    )


def test_iir_complex_coefficients() -> None:
    iir = IIR(
        b=[1 + 1j],
        a=[1],
    )

    x = np.array(
        [
            1,
            2,
            3,
        ]
    )

    result = iir(x)

    expected = np.array(
        [
            1 + 1j,
            2 + 2j,
            3 + 3j,
        ]
    )

    assert np.array_equal(
        result,
        expected,
    )


def test_iir_complex_feedback() -> None:
    iir = IIR(
        b=[1],
        a=[1, -1j],
    )

    x = np.array(
        [
            1,
            1,
            1,
        ]
    )

    result = iir(x)

    expected = np.array(
        [
            1,
            1 + 1j,
            1j,
        ]
    )

    assert np.allclose(
        result,
        expected,
    )


# ---------------------------------------------------------------------------
# Multi-channel input
# ---------------------------------------------------------------------------


def test_iir_multichannel() -> None:
    iir = IIR(
        b=[1],
        a=[1],
    )

    x = np.array(
        [
            [1, 2, 3],
            [4, 5, 6],
        ]
    )

    result = iir(x)

    assert np.array_equal(
        result,
        x,
    )


def test_iir_shared_coefficients_apply_to_each_channel() -> None:
    iir = IIR(
        b=[1, 2],
        a=[1],
    )

    x = np.array(
        [
            [1, 2, 3],
            [4, 5, 6],
        ]
    )

    result = iir(x)

    expected = np.array(
        [
            [1, 4, 7],
            [4, 13, 16],
        ]
    )

    assert np.array_equal(
        result,
        expected,
    )


def test_iir_per_channel_coefficients() -> None:
    iir = IIR(
        b=[
            [1, 2],
            [2, 3],
        ],
        a=[
            [1],
            [1],
        ],
    )

    x = np.array(
        [
            [1, 2, 3],
            [4, 5, 6],
        ]
    )

    result = iir(x)

    expected = np.array(
        [
            [1, 4, 7],
            [8, 22, 27],
        ]
    )

    assert np.array_equal(
        result,
        expected,
    )


def test_iir_wrong_number_of_per_channel_coefficients_raises_b() -> None:
    with pytest.raises(
        ValueError,
        match=r"b is 2D \(per-channel\) but a is 1D \(shared\); both must be 1D or both must be 2D with matching channel counts",
    ):
        IIR(
            b=[
                [1, 2],
                [3, 4],
            ],
            a=[1],
        )


def test_iir_wrong_number_of_per_channel_coefficients_raises_a() -> None:
    with pytest.raises(
        ValueError,
        match=r"a is 2D \(per-channel\) but b is 1D \(shared\); both must be 1D or both must be 2D with matching channel counts",
    ):
        IIR(
            b=[1],
            a=[
                [1, 2],
                [3, 4],
            ],
        )


# ---------------------------------------------------------------------------
# Stateful behavior
# ---------------------------------------------------------------------------


def test_iir_is_stateful() -> None:
    iir = IIR(
        b=[1],
        a=[1, -1],
    )

    assert iir.stateful


def test_iir_state_carries_between_calls() -> None:
    iir = IIR(
        b=[1],
        a=[1, -1],
    )

    first = iir([1, 1])
    second = iir([1, 1])

    assert np.array_equal(
        first,
        np.array([1, 2]),
    )

    assert np.array_equal(
        second,
        np.array([3, 4]),
    )


def test_iir_reset_clears_state() -> None:
    iir = IIR(
        b=[1],
        a=[1, -1],
    )

    first = iir([1, 1])

    iir.reset()

    second = iir([1, 1])

    assert np.array_equal(
        first,
        second,
    )


def test_iir_reset_allows_new_channel_shape() -> None:
    iir = IIR(
        b=[1],
        a=[1, -1],
    )

    iir(
        [
            [1, 2],
            [3, 4],
        ]
    )

    iir.reset()

    result = iir(
        [
            [1, 2],
            [3, 4],
            [5, 6],
        ]
    )

    expected = np.array(
        [
            [1, 3],
            [3, 7],
            [5, 11],
        ]
    )

    assert np.array_equal(
        result,
        expected,
    )


def test_single_row_coefficients_broadcast_across_channels():
    iir = IIR(
        b=np.array([[1.0, 0.0]]),
        a=np.array([[1.0, 0.0]]),
    )

    x = np.array(
        [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
        ]
    )

    y = iir.process(x)

    np.testing.assert_allclose(y, x)


# ---------------------------------------------------------------------------
# Variable sample length
# ---------------------------------------------------------------------------


def test_iir_sample_length_can_change_between_calls() -> None:
    iir = IIR(
        b=[1],
        a=[1],
    )

    first = iir([1, 2, 3])
    second = iir([4, 5])

    assert np.array_equal(
        first,
        np.array([1, 2, 3]),
    )

    assert np.array_equal(
        second,
        np.array([4, 5]),
    )


def test_iir_multichannel_sample_length_can_change() -> None:
    iir = IIR(
        b=[1],
        a=[1],
    )

    first = iir(
        [
            [1, 2, 3],
            [4, 5, 6],
        ]
    )

    second = iir(
        [
            [7, 8],
            [9, 10],
        ]
    )

    assert np.array_equal(
        first,
        np.array(
            [
                [1, 2, 3],
                [4, 5, 6],
            ]
        ),
    )

    assert np.array_equal(
        second,
        np.array(
            [
                [7, 8],
                [9, 10],
            ]
        ),
    )


def test_iir_multichannel_leading_shape_cannot_change() -> None:
    iir = IIR(
        b=[1],
        a=[1],
    )

    iir(
        [
            [1, 2, 3],
            [4, 5, 6],
        ]
    )

    with pytest.raises(ValueError, match="Input channel count changed"):
        iir(
            [
                [1, 2, 3],
            ]
        )


# ---------------------------------------------------------------------------
# Higher-dimensional channel shapes
# ---------------------------------------------------------------------------


def test_iir_supports_multiple_leading_dimensions() -> None:
    iir = IIR(
        b=[1],
        a=[1],
    )

    x = np.arange(24).reshape(2, 3, 4)

    result = iir(x)

    assert np.array_equal(
        result,
        x,
    )


def test_iir_higher_dimensional_leading_shape_cannot_change() -> None:
    iir = IIR(
        b=[1],
        a=[1],
    )

    iir(np.zeros((2, 3, 4)))

    with pytest.raises(ValueError, match="Input channel count changed"):
        iir(np.zeros((2, 4, 4)))


# ---------------------------------------------------------------------------
# Pipeline integration
# ---------------------------------------------------------------------------


def test_iir_can_follow_pipeline() -> None:
    iir = IIR(
        b=[1],
        a=[1],
    )

    pipeline = Add(1) >> iir

    result = pipeline([1, 2, 3])

    expected = np.array(
        [
            2,
            3,
            4,
        ]
    )

    assert np.array_equal(
        result,
        expected,
    )


def test_iir_can_precede_pipeline() -> None:
    iir = IIR(
        b=[1],
        a=[1],
    )

    pipeline = iir >> Add(1)

    result = pipeline([1, 2, 3])

    expected = np.array(
        [
            2,
            3,
            4,
        ]
    )

    assert np.array_equal(
        result,
        expected,
    )


def test_iir_can_be_composed_with_pipeline() -> None:
    iir = IIR(
        b=[1],
        a=[1],
    )

    pipeline = Add(1) >> iir >> Multiply(2)

    result = pipeline([1, 2, 3])

    expected = np.array(
        [
            4,
            6,
            8,
        ]
    )

    assert np.array_equal(
        result,
        expected,
    )


# ---------------------------------------------------------------------------
# Block API
# ---------------------------------------------------------------------------


def test_iir_len() -> None:
    iir = IIR(
        b=[1],
        a=[1],
    )

    assert len(iir) == 1


def test_iir_repr_contains_coefficients() -> None:
    iir = IIR(
        b=[1, 2],
        a=[1, 3],
        name="iir1",
    )

    result = repr(iir)

    assert "IIR" in result
    assert "iir1" in result


def test_iir_profile() -> None:
    iir = IIR(
        b=[1],
        a=[1],
    )

    results = iir.profile(
        np.array([1, 2, 3]),
    )

    assert len(results) == 1


def test_iir_profile_reset() -> None:
    iir = IIR(
        b=[1],
        a=[1, -1],
    )

    iir([1, 2, 3])

    iir.profile(
        np.array([1, 2, 3]),
        reset=True,
    )

    result = iir([1])

    assert np.array_equal(
        result,
        np.array([1]),
    )

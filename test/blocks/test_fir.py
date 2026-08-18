import numpy as np
import pytest

from npdsp import *

# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_fir_constructor(
    fir1: FIR,
) -> None:
    assert isinstance(fir1, FIR)
    assert np.array_equal(
        fir1.coefs,
        np.array([1, 2, 3]),
    )


def test_fir_coefficients_are_numpy_array(
    fir1: FIR,
) -> None:
    assert isinstance(
        fir1.coefs,
        np.ndarray,
    )


def test_fir_name(
    fir1: FIR,
) -> None:
    assert fir1.name == "fir1"


def test_fir_repr(
    fir1: FIR,
) -> None:
    assert "FIR" in repr(fir1)
    assert "fir1" in repr(fir1)


def test_fir_length(
    fir1: FIR,
) -> None:
    assert len(fir1) == 1


def test_fir_is_block(
    fir1: FIR,
) -> None:
    assert isinstance(
        fir1,
        Block,
    )


def test_fir_is_stateful(
    fir1: FIR,
) -> None:
    assert fir1.stateful


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------


def test_fir_scalar_coefficients_raise() -> None:
    with pytest.raises(
        ValueError, match="FIR coefficients must be at least one-dimensional"
    ):
        FIR(1)


def test_fir_empty_coefficients_raise() -> None:
    with pytest.raises(ValueError, match="FIR coefficients cannot be empty"):
        FIR([])


# ---------------------------------------------------------------------------
# Basic filtering
# ---------------------------------------------------------------------------


def test_fir_identity() -> None:
    fir = FIR([1])

    x = np.array([1, 2, 3, 4])

    result = fir(x)

    assert np.array_equal(
        result,
        x,
    )


def test_fir_single_tap() -> None:
    fir = FIR([2])

    result = fir([1, 2, 3])

    assert np.array_equal(
        result,
        [2, 4, 6],
    )


def test_fir_basic(
    fir1: FIR,
) -> None:
    result = fir1([1, 2, 3])

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


def test_fir_impulse_response(
    fir1: FIR,
) -> None:
    result = fir1([1, 0, 0, 0, 0])

    expected = np.array(
        [
            1,
            2,
            3,
            0,
            0,
        ]
    )

    assert np.array_equal(
        result,
        expected,
    )


def test_fir_zero_input(
    fir1: FIR,
) -> None:
    result = fir1([0, 0, 0, 0])

    assert np.array_equal(
        result,
        np.zeros(4),
    )


def test_fir_negative_coefficients() -> None:
    fir = FIR([1, -1])

    result = fir([1, 2, 4, 7])

    expected = np.array(
        [
            1,
            1,
            2,
            3,
        ]
    )

    assert np.array_equal(
        result,
        expected,
    )


# ---------------------------------------------------------------------------
# Input handling
# ---------------------------------------------------------------------------


def test_fir_accepts_list(
    fir1: FIR,
) -> None:
    result = fir1([1, 2, 3])

    assert isinstance(
        result,
        np.ndarray,
    )


def test_fir_accepts_numpy_array(
    fir1: FIR,
) -> None:
    x = np.array([1, 2, 3])

    result = fir1(x)

    assert isinstance(
        result,
        np.ndarray,
    )


def test_fir_preserves_shape(
    fir1: FIR,
) -> None:
    x = np.zeros((2, 3, 10))

    result = fir1(x)

    assert result.shape == x.shape


@pytest.mark.parametrize(
    "length",
    [1, 2, 5, 100],
)
def test_fir_preserves_sample_length(
    fir1: FIR,
    length: int,
) -> None:
    result = fir1(np.zeros(length))

    assert result.shape == (length,)


def test_fir_scalar_input_raises(
    fir1: FIR,
) -> None:
    with pytest.raises(
        ValueError, match="FIR input must be at least one-dimensional"
    ):
        fir1(1)


# ---------------------------------------------------------------------------
# Multi-dimensional signals
# ---------------------------------------------------------------------------


def test_fir_multiple_channels() -> None:
    fir = FIR([1, 2, 3])

    x = np.array(
        [
            [1, 2, 3],
            [4, 5, 6],
        ]
    )

    result = fir(x)

    expected = np.array(
        [
            [1, 4, 10],
            [4, 13, 28],
        ]
    )

    assert np.array_equal(
        result,
        expected,
    )


def test_fir_same_filter_applied_independently_to_channels() -> None:
    fir = FIR([1, 2])

    x = np.array(
        [
            [1, 2, 3],
            [10, 20, 30],
        ]
    )

    result = fir(x)

    expected = np.array(
        [
            [1, 4, 7],
            [10, 40, 70],
        ]
    )

    assert np.array_equal(
        result,
        expected,
    )


def test_fir_three_dimensional_input() -> None:
    fir = FIR([1, 2])

    x = np.array(
        [
            [
                [1, 2, 3],
                [4, 5, 6],
            ],
            [
                [10, 20, 30],
                [40, 50, 60],
            ],
        ]
    )

    result = fir(x)

    expected = np.array(
        [
            [
                [1, 4, 7],
                [4, 13, 16],
            ],
            [
                [10, 40, 70],
                [40, 130, 160],
            ],
        ]
    )

    assert np.array_equal(
        result,
        expected,
    )


# ---------------------------------------------------------------------------
# Per-channel coefficients
# ---------------------------------------------------------------------------


def test_fir_per_channel_coefficients() -> None:
    fir = FIR(
        [
            [1, 2],
            [10, 20],
        ]
    )

    x = np.array(
        [
            [1, 2, 3],
            [1, 2, 3],
        ]
    )

    result = fir(x)

    expected = np.array(
        [
            [1, 4, 7],
            [10, 40, 70],
        ]
    )

    assert np.array_equal(
        result,
        expected,
    )


def test_fir_per_channel_coefficients_with_three_dimensions() -> None:
    fir = FIR(
        [
            [
                [1, 2],
                [3, 4],
            ],
            [
                [10, 20],
                [30, 40],
            ],
        ]
    )

    x = np.ones((2, 2, 3))

    result = fir(x)

    expected = np.array(
        [
            [
                [1, 3, 3],
                [3, 7, 7],
            ],
            [
                [10, 30, 30],
                [30, 70, 70],
            ],
        ]
    )

    assert np.array_equal(
        result,
        expected,
    )


def test_fir_per_channel_coefficients_must_match_input_channels() -> None:
    fir = FIR(
        [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ]
    )

    with pytest.raises(
        ValueError,
        match=r"Coefficient channel count \(3\) does not match input \(2\)",
    ):
        fir(
            [
                [1, 2, 3],
                [4, 5, 6],
            ]
        )


def test_fir_per_channel_coefficients_must_match_input_channels_ndim3() -> None:
    fir = FIR(
        [
            [
                [1, 2, 3],
            ]
        ]
    )

    with pytest.raises(
        ValueError,
        match=r"Coefficient channel count \(1\) does not match input \(2\)",
    ):
        fir([[[1, 2, 3], [1, 2, 3]]])


def test_fir_0d_coefficients_not_supported() -> None:

    with pytest.raises(
        ValueError,
        match=r"FIR coefficients cannot be empty",
    ):
        FIR([])


# ---------------------------------------------------------------------------
# Complex data
# ---------------------------------------------------------------------------


def test_fir_complex() -> None:
    fir = FIR([1, 2])

    result = fir(
        np.array(
            [
                1 + 1j,
                2 + 2j,
                3 + 3j,
            ]
        )
    )

    expected = np.array(
        [
            1 + 1j,
            4 + 4j,
            7 + 7j,
        ]
    )

    assert np.array_equal(
        result,
        expected,
    )


def test_fir_complex_coefficients() -> None:
    fir = FIR(
        [
            1 + 1j,
            2 - 1j,
        ]
    )

    result = fir(
        np.array(
            [
                1,
                2,
                3,
            ]
        )
    )

    expected = np.array(
        [
            1 + 1j,
            4 + 1j,
            7 + 1j,
        ]
    )

    assert np.allclose(
        result,
        expected,
    )


def test_fir_complex_result_is_complex() -> None:
    fir = FIR([1, 2])

    result = fir(
        np.array(
            [
                1 + 1j,
                2 + 2j,
            ]
        )
    )

    assert np.iscomplexobj(result)


# ---------------------------------------------------------------------------
# Stateful processing
# ---------------------------------------------------------------------------


def test_fir_state_is_preserved_between_calls() -> None:
    fir = FIR([1, 2, 3])

    first = fir([1, 2])
    second = fir([3, 4])

    assert np.array_equal(
        first,
        [1, 4],
    )

    assert np.array_equal(
        second,
        [10, 16],
    )


def test_fir_chunked_processing_equals_continuous_processing() -> None:
    x = np.array(
        [
            1,
            2,
            3,
            4,
            5,
            6,
        ]
    )

    continuous = FIR([1, 2, 3])
    expected = continuous(x)

    chunked = FIR([1, 2, 3])

    result = np.concatenate(
        [
            chunked(x[:2]),
            chunked(x[2:4]),
            chunked(x[4:]),
        ]
    )

    assert np.allclose(
        result,
        expected,
    )


def test_fir_chunked_processing_with_channels() -> None:
    x = np.array(
        [
            [1, 2, 3, 4, 5],
            [10, 20, 30, 40, 50],
        ]
    )

    continuous = FIR([1, 2, 3])
    expected = continuous(x)

    chunked = FIR([1, 2, 3])

    result = np.concatenate(
        [
            chunked(x[:, :2]),
            chunked(x[:, 2:4]),
            chunked(x[:, 4:]),
        ],
        axis=-1,
    )

    assert np.allclose(
        result,
        expected,
    )


def test_fir_sample_dimension_can_change_between_calls() -> None:
    fir = FIR([1, 2, 3])

    first = fir(
        [
            [1, 2, 3],
            [4, 5, 6],
        ]
    )

    second = fir(
        [
            [1, 2],
            [4, 5],
        ]
    )

    assert first.shape == (2, 3)
    assert second.shape == (2, 2)


def test_fir_leading_shape_cannot_change() -> None:
    fir = FIR([1, 2, 3])

    fir(
        [
            [1, 2, 3],
            [4, 5, 6],
        ]
    )

    with pytest.raises(ValueError, match="Input channel count changed"):
        fir(
            [
                [1, 2, 3],
            ]
        )


def test_fir_higher_dimension_leading_shape_cannot_change() -> None:
    fir = FIR([1, 2, 3])

    fir(np.zeros((2, 4, 10)))

    with pytest.raises(ValueError, match="Input channel count changed"):
        fir(np.zeros((2, 3, 10)))


def test_fir_leading_shape_established_by_first_call() -> None:
    fir = FIR([1, 2, 3])

    fir(np.zeros((2, 3)))

    with pytest.raises(ValueError, match="Input channel count changed"):
        fir(np.zeros((3, 3)))


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------


def test_fir_reset_clears_state() -> None:
    fir = FIR([1, 2, 3])

    fir([1, 2, 3])
    fir.reset()

    result = fir([4])

    assert np.array_equal(
        result,
        [4],
    )


def test_fir_reset_allows_new_leading_shape() -> None:
    fir = FIR([1, 2, 3])

    fir(np.zeros((2, 10)))

    fir.reset()

    result = fir(np.zeros((1, 10)))

    assert result.shape == (1, 10)


def test_fir_reset_restores_initial_state() -> None:
    fir = FIR([1, 2, 3])

    fir([1, 2, 3])
    fir.reset()

    result = fir([1, 2])

    expected = np.array(
        [
            1,
            4,
        ]
    )

    assert np.array_equal(
        result,
        expected,
    )


# ---------------------------------------------------------------------------
# Dtype
# ---------------------------------------------------------------------------


def test_fir_integer_input_produces_numeric_output() -> None:
    fir = FIR([1, 2, 3])

    result = fir(
        np.array(
            [1, 2, 3],
            dtype=int,
        )
    )

    assert np.issubdtype(
        result.dtype,
        np.number,
    )


def test_fir_float_input() -> None:
    fir = FIR([0.5, 0.25])

    result = fir(
        np.array(
            [1.0, 2.0, 3.0],
            dtype=float,
        )
    )

    expected = np.array(
        [
            0.5,
            1.25,
            2.0,
        ]
    )

    assert np.allclose(
        result,
        expected,
    )


# ---------------------------------------------------------------------------
# Pipeline integration
# ---------------------------------------------------------------------------


def test_fir_can_be_used_in_pipeline() -> None:
    fir = FIR([1, 2])

    pipeline = fir >> Add(1)

    result = pipeline([1, 2, 3])

    expected = np.array(
        [
            2,
            5,
            8,
        ]
    )

    assert np.array_equal(
        result,
        expected,
    )


def test_fir_can_follow_pipeline() -> None:
    fir = FIR([1, 2])

    pipeline = Add(1) >> fir

    result = pipeline([1, 2, 3])

    expected = np.array(
        [
            2,
            7,
            10,
        ]
    )

    assert np.array_equal(
        result,
        expected,
    )


def test_fir_pipeline_preserves_fir_state() -> None:
    fir = FIR([1, 2])

    pipeline = fir >> Add(1)

    first = pipeline([1, 2])
    second = pipeline([3, 4])

    assert np.array_equal(
        first,
        [2, 5],
    )

    assert np.array_equal(
        second,
        [8, 11],
    )


# ---------------------------------------------------------------------------
# Find / naming integration
# ---------------------------------------------------------------------------


def test_fir_can_be_found_in_pipeline() -> None:
    fir = FIR(
        [1, 2, 3],
        name="fir",
    )

    pipeline = Add(1) >> fir >> Multiply(2)

    assert pipeline.find(FIR) is fir


def test_fir_can_be_found_in_pipeline_by_name() -> None:
    fir = FIR(
        [1, 2, 3],
        name="fir",
    )

    pipeline = Add(1) >> fir

    assert pipeline["fir"] is fir


# ---------------------------------------------------------------------------
# Profiling
# ---------------------------------------------------------------------------


def test_fir_profile() -> None:
    fir = FIR([1, 2, 3])

    results = fir.profile(
        np.array([1, 2, 3]),
    )

    assert len(results) == 1


def test_fir_profile_has_name(
    fir1: FIR,
) -> None:
    results = fir1.profile(
        np.array([1, 2, 3]),
    )

    assert results[0].name == "fir1"


def test_fir_profile_reset() -> None:
    fir = FIR([1, 2, 3])

    fir([1, 2, 3])

    fir.profile(
        np.array([1, 2, 3]),
        reset=True,
    )

    result = fir([1])

    assert np.array_equal(
        result,
        [1],
    )


def test_fir_latency_property() -> None:
    fir = FIR([1, 2, 3])
    assert fir.latency_samples == 1


def test_fir_is_type1() -> None:
    fir = FIR([1, 2, 1])
    assert fir.type == 1


def test_fir_is_type2() -> None:
    fir = FIR([1, 2, 2, 1])
    assert fir.type == 2


def test_fir_is_type3() -> None:
    fir = FIR([1, 2, 0, -2, -1])
    assert fir.type == 3


def test_fir_is_type4() -> None:
    fir = FIR([1, 2, -2, -1])
    assert fir.type == 4


def test_fir_is_not_any_type() -> None:
    fir = FIR([1, 2, 3, -2, -1])
    assert fir.type is None


def test_fir_2ch_1coef() -> None:
    fir = FIR([[1, 1]])
    y = fir.process(np.array([[1, 1]]))

    assert np.array_equal(
        np.array([[1, 2]]),
        y,
    )

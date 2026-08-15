import numpy as np

from npdsp import *
from npdsp.blocks.filters import design, impulse_response, window

# ---------------------------------------------------------------------------
# Windows
# ---------------------------------------------------------------------------


def test_window_returns_zeros() -> None:
    result = window.Window(5)

    assert np.array_equal(result, np.zeros(5))


def test_rectangular_window_returns_ones() -> None:
    result = window.rectangular(4)

    assert np.array_equal(result, np.ones(4))


def test_triangular_window_matches_expected_values() -> None:
    n = np.arange(5) - (5 - 1) / 2
    expected = 1 - np.abs(n) / ((5 - 1) / 2)

    result = window.triangular(5)

    assert np.allclose(result, expected)


def test_hanning_window_matches_expected_values() -> None:
    n = np.arange(5) - (5 - 1) / 2
    expected = 0.5 + 0.5 * np.cos(n * np.pi / 5)

    result = window.hanning(5)

    assert np.allclose(result, expected)


def test_hamming_window_matches_expected_values() -> None:
    n = np.arange(5) - (5 - 1) / 2
    expected = 0.54 + 0.46 * np.cos(n * np.pi / 5)

    result = window.hamming(5)

    assert np.allclose(result, expected)


def test_blackman_window_matches_expected_values() -> None:
    n = np.arange(5) - (5 - 1) / 2
    expected = 0.42 + 0.5 * np.cos(n * np.pi / 5) + 0.08 * np.cos(2 * n * np.pi / 5)

    result = window.blackman(5)

    assert np.allclose(result, expected)


# ---------------------------------------------------------------------------
# Impulse response helpers
# ---------------------------------------------------------------------------


def test_ideal_allpass_matches_sinc_formula() -> None:
    n = np.arange(5) - (5 - 1) / 2
    expected = np.sinc(n)

    result = impulse_response.ideal_allpass(5)

    assert np.allclose(result, expected)


def test_lowpass_matches_sinc_formula() -> None:
    fc_norm = 0.1
    N = 5
    n = np.arange(N) - (N - 1) / 2
    expected = 2 * fc_norm * np.sinc(2 * fc_norm * n)

    result = impulse_response.lowpass(fc_norm, N)

    assert np.allclose(result, expected)


def test_highpass_matches_allpass_minus_lowpass() -> None:
    fc_norm = 0.1
    N = 5

    result = impulse_response.highpass(fc_norm, N)
    expected = impulse_response.ideal_allpass(N) - impulse_response.lowpass(fc_norm, N)

    assert np.allclose(result, expected)


def test_bandpass_matches_lowpass_difference() -> None:
    f_low_norm = 0.1
    f_high_norm = 0.2
    N = 5

    result = impulse_response.bandpass(f_low_norm, f_high_norm, N)
    expected = impulse_response.lowpass(f_high_norm, N) - impulse_response.lowpass(
        f_low_norm, N
    )

    assert np.allclose(result, expected)


def test_bandstop_matches_allpass_minus_bandpass() -> None:
    f_low_norm = 0.1
    f_high_norm = 0.2
    N = 5

    result = impulse_response.bandstop(f_low_norm, f_high_norm, N)
    expected = impulse_response.ideal_allpass(N) - impulse_response.bandpass(
        f_low_norm, f_high_norm, N
    )

    assert np.allclose(result, expected)


# ---------------------------------------------------------------------------
# Filter design blocks
# ---------------------------------------------------------------------------


def test_lowpass_design_uses_window_and_normalizes_coefficients() -> None:
    block = design.Lowpass(
        fc=0.1,
        ft_or_n=0.1,
        f_norm=True,
        window=window.rectangular,
        name="lowpass",
    )

    assert isinstance(block, Pipeline)
    assert isinstance(block.fir, FIR)
    assert block.name == "lowpass"
    assert block.fc == 0.1
    assert block.ft == 0.1
    assert block.num_coefs == 41
    assert len(block.coefs) == 41
    assert np.isclose(np.sum(block.coefs), 1.0)
    assert np.array_equal(block.fir.coefs, block.coefs)


def test_lowpass_design_can_use_fixed_coefficient_length() -> None:
    block = design.Lowpass(
        fc=0.1,
        ft_or_n=9,
        f_norm=True,
        window=window.rectangular,
        use_fixed_coef_len=True,
        name="fixed_lowpass",
    )

    assert block.ft is None
    assert block.num_coefs == 9
    assert len(block.coefs) == 9


def test_lowpass_design_can_allow_even_num_coefs() -> None:
    block = design.Lowpass(
        fc=0.1,
        ft_or_n=0.25,
        f_norm=True,
        window=window.rectangular,
        allow_even_n=True,
    )

    assert block.num_coefs == 16


def test_highpass_design_normalizes_response() -> None:
    block = design.Highpass(
        fc=0.1,
        ft_or_n=0.2,
        f_norm=True,
        window=window.rectangular,
        name="highpass",
    )

    assert block.fc == 0.1
    assert block.ft == 0.2
    assert block.num_coefs == 21
    assert len(block.coefs) == 21
    assert np.isclose(np.abs(np.fft.rfft(block.coefs)[-1]), 1.0)


def test_bandpass_design_uses_requested_cutoffs() -> None:
    block = design.Bandpass(
        fc1=0.1,
        fc2=0.2,
        ft_or_n=0.1,
        f_norm=True,
        window=window.rectangular,
        name="bandpass",
    )

    assert block.fc1 == 0.1
    assert block.fc2 == 0.2
    assert block.ft == 0.1
    assert block.num_coefs == 41
    assert len(block.coefs) == 41
    assert np.array_equal(block.fir.coefs, block.coefs)


def test_bandstop_design_normalizes_sum() -> None:
    block = design.Bandstop(
        fc1=0.1,
        fc2=0.2,
        ft_or_n=0.1,
        f_norm=True,
        window=window.rectangular,
        name="bandstop",
    )

    assert block.fc1 == 0.1
    assert block.fc2 == 0.2
    assert block.ft == 0.1
    assert block.num_coefs == 41
    assert len(block.coefs) == 41
    assert np.isclose(np.sum(block.coefs), 1.0)

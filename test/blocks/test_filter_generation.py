from collections.abc import Callable

import numpy as np
import pytest

from npdsp import *
from npdsp.blocks.filters import design, impulse_response, window


def test_lowpass_generation_matches_windowed_sinc_and_normalizes() -> None:
    # Use a rectangular window so windowing is identity (ones)
    fc = 0.1
    ft = 0.1

    block = design.Lowpass(
        fc=fc,
        ft_or_n=ft,
        f_norm=True,
        window=window.Rectangular,
        name="lp_test",
    )

    # Manual computation of expected coefficients: window * ideal lowpass, then normalize
    N = block.num_coefs
    w = window.Rectangular(N)
    ir = impulse_response.lowpass(block.fc, N)
    expected = w * ir
    expected = expected / np.sum(expected)

    assert isinstance(block, Pipeline)
    assert np.allclose(block.coefs, expected)
    assert np.isclose(np.sum(block.coefs), 1.0)
    # FIR block inside pipeline should carry the same coefficients
    assert np.array_equal(block.fir.coefs, block.coefs)


def test_lowpass_num_coefs_calculation_and_oddness() -> None:
    # ft=0.1 -> ceil(4/0.1)=40 -> should be incremented to 41 (odd) by default
    block = design.Lowpass(
        fc=0.1,
        ft_or_n=0.1,
        f_norm=True,
        window=window.Rectangular,
    )

    assert block.ft == 0.1
    assert block.num_coefs == 41
    assert len(block.coefs) == 41


def test_highpass_normalizes_response_magnitude() -> None:
    # Verify highpass normalization uses the FFT magnitude as implemented
    fc = 0.1
    block = design.Highpass(
        fc=fc,
        ft_or_n=0.2,
        f_norm=True,
        window=window.Rectangular,
        name="hp_test",
    )

    N = block.num_coefs
    c = block.coefs

    # Determine the index the implementation uses for magnitude normalization
    mag_idx = -1 if N % 2 == 1 else int(N * (0.5 + block.fc) / 2)

    # The chosen FFT bin magnitude should be normalized to 1.0
    fft_vals = np.fft.rfft(c)
    assert np.isclose(np.abs(fft_vals[mag_idx]), 1.0, atol=1e-8)


def test_bandpass_normalizes_at_center_frequency() -> None:
    fc1 = 0.1
    fc2 = 0.2
    block = design.Bandpass(
        fc1=fc1,
        fc2=fc2,
        ft_or_n=0.1,
        f_norm=True,
        window=window.Rectangular,
        name="bp_test",
    )

    N = block.num_coefs
    c = block.coefs

    # Index used in implementation for normalization is int(N * (fc1 + fc2) / 2)
    idx = int(N * (fc1 + fc2) / 2)
    fft_vals = np.fft.rfft(c)

    assert np.isclose(np.abs(fft_vals[idx]), 1.0, atol=1e-8)


def test_bandstop_generation_matches_windowed_ir_and_normalizes_sum() -> None:
    fc1 = 0.1
    fc2 = 0.2
    block = design.Bandstop(
        fc1=fc1,
        fc2=fc2,
        ft_or_n=0.1,
        f_norm=True,
        window=window.Rectangular,
        name="bs_test",
    )

    N = block.num_coefs
    w = window.Rectangular(N)
    ir = impulse_response.bandstop(block.fc1, block.fc2, N)
    expected = w * ir
    expected = expected / np.sum(expected)

    assert np.allclose(block.coefs, expected)
    assert np.isclose(np.sum(block.coefs), 1.0)


# ---------------------------------------------------------------------------
# Frequency response checks
# ---------------------------------------------------------------------------


def _mag_at(coefs: np.ndarray, f: float, nfft: int = 4096) -> float:
    """Return magnitude of the frequency response of coefs at normalized
    frequency f (cycles/sample), with zero-padding to nfft for resolution.
    """
    H = np.fft.rfft(coefs, n=nfft)
    idx = round(f * nfft)
    idx = max(0, min(idx, len(H) - 1))
    return float(np.abs(H[idx]))


def test_lowpass_frequency_behavior() -> None:
    # Low frequencies should pass, high frequencies should be attenuated
    block = design.Lowpass(
        fc=0.1,
        ft_or_n=0.1,
        f_norm=True,
        window=window.Rectangular,
    )

    pass_mag = _mag_at(block.coefs, 0.05)
    stop_mag = _mag_at(block.coefs, 0.40)

    assert pass_mag > 0.9
    assert stop_mag < 0.1


def test_highpass_frequency_behavior() -> None:
    # High frequencies should pass, low frequencies should be attenuated
    block = design.Highpass(
        fc=0.1,
        ft_or_n=0.1,
        f_norm=True,
        window=window.Rectangular,
    )

    low_mag = _mag_at(block.coefs, 0.05)
    high_mag = _mag_at(block.coefs, 0.40)

    assert high_mag > 0.9
    assert low_mag < 0.1


def test_bandpass_frequency_behavior() -> None:
    # Frequencies inside the band should pass, outside should be attenuated
    fc1 = 0.1
    fc2 = 0.2
    center = 0.5 * (fc1 + fc2)

    block = design.Bandpass(
        fc1=fc1,
        fc2=fc2,
        ft_or_n=0.1,
        f_norm=True,
        window=window.Rectangular,
    )

    center_mag = _mag_at(block.coefs, center)
    below_mag = _mag_at(block.coefs, 0.05)
    above_mag = _mag_at(block.coefs, 0.40)

    assert center_mag > 0.9
    assert below_mag < 0.1
    assert above_mag < 0.1


def test_bandstop_frequency_behavior() -> None:
    # Frequencies inside the stop band should be attenuated; outside should pass
    fc1 = 0.1
    fc2 = 0.2
    center = 0.5 * (fc1 + fc2)

    block = design.Bandstop(
        fc1=fc1,
        fc2=fc2,
        ft_or_n=0.1,
        f_norm=True,
        window=window.Rectangular,
    )

    center_mag = _mag_at(block.coefs, center)
    dc_mag = _mag_at(block.coefs, 0.0)
    high_mag = _mag_at(block.coefs, 0.45)

    assert center_mag < 0.1
    assert dc_mag > 0.9
    assert high_mag > 0.9


def test_num_coefs_is_computed_from_ft_formula_and_is_monotonic() -> None:
    # Validate formula used in design: num_coefs = ceil(4/ft) adjusted to odd when needed
    ft1 = 0.1
    ft2 = 0.2

    def expected_num_coefs(ft: float, allow_even_n: bool = False) -> int:
        import math

        num = math.ceil(4 / ft)
        if not (num % 2) and not allow_even_n:
            num += 1
        return num

    b1 = design.Lowpass(
        fc=0.1, ft_or_n=ft1, f_norm=True, window=window.Rectangular
    )
    b2 = design.Lowpass(
        fc=0.1, ft_or_n=ft2, f_norm=True, window=window.Rectangular
    )

    assert b1.num_coefs == expected_num_coefs(ft1)
    assert b2.num_coefs == expected_num_coefs(ft2)
    assert b1.num_coefs > b2.num_coefs


def test_ft_reducing_makes_transition_sharper() -> None:
    # A smaller ft (narrower transition) should produce a steeper roll-off.
    fc = 0.1
    ft_small = 0.05
    ft_large = 0.2

    b_small = design.Lowpass(
        fc=fc, ft_or_n=ft_small, f_norm=True, window=window.Rectangular
    )
    b_large = design.Lowpass(
        fc=fc, ft_or_n=ft_large, f_norm=True, window=window.Rectangular
    )

    # Test at a frequency slightly above the cutoff where the difference should be visible
    f_test = fc + 0.05

    mag_small = _mag_at(b_small.coefs, f_test)
    mag_large = _mag_at(b_large.coefs, f_test)

    # Smaller transition width should attenuate more just above cutoff
    assert mag_small < mag_large


@pytest.mark.parametrize(
    "window_fn",
    [
        window.Rectangular,
        window.Triangular,
        window.Hanning,
        window.Hamming,
        window.Blackman,
    ],
)
def test_corner_frequency_matches_3db_within_transition_width(
    window_fn: Callable[[int], Signal],
) -> None:
    """Locate the -3dB frequency and assert it lies within half the transition
    width ft of the requested cutoff frequency for multiple window functions.
    """
    fc = 0.12
    ft = 0.1

    low = design.Lowpass(fc=fc, ft_or_n=ft, f_norm=True, window=window_fn)
    high = design.Highpass(fc=fc, ft_or_n=ft, f_norm=True, window=window_fn)

    # Dense FFT for accurate interpolation
    nfft = 16384

    # Lowpass: locate frequency where |H(f)| falls to DC/sqrt(2)
    H_low = np.abs(np.fft.rfft(low.coefs, n=nfft))
    freqs = np.arange(len(H_low)) / nfft
    target_low = H_low[0] / np.sqrt(2)

    # Find first crossing after DC where magnitude <= target
    cross_idx = None
    for i in range(1, len(H_low)):
        if H_low[i] <= target_low:
            cross_idx = i
            break

    assert cross_idx is not None, (
        f"No -3dB crossing found for lowpass with {window_fn}"
    )

    # Linear interpolation between bins to estimate crossing frequency
    y1, y2 = H_low[cross_idx - 1], H_low[cross_idx]
    f1, f2 = freqs[cross_idx - 1], freqs[cross_idx]
    f3db_low = (
        f1 if y2 == y1 else f1 + (target_low - y1) * (f2 - f1) / (y2 - y1)
    )

    assert abs(f3db_low - fc) <= ft / 2

    # Highpass: use the high-frequency magnitude as reference (last bin)
    H_high = np.abs(np.fft.rfft(high.coefs, n=nfft))
    target_high = H_high[-1] / np.sqrt(2)

    # Find first crossing from high end downward where magnitude <= target_high
    cross_idx = None
    for i in range(len(H_high) - 2, -1, -1):
        if H_high[i] <= target_high:
            cross_idx = i
            break

    assert cross_idx is not None, (
        f"No -3dB crossing found for highpass with {window_fn}"
    )

    y1, y2 = H_high[cross_idx], H_high[cross_idx + 1]
    f1, f2 = freqs[cross_idx], freqs[cross_idx + 1]
    f3db_high = (
        f2 if y2 == y1 else f1 + (target_high - y1) * (f2 - f1) / (y2 - y1)
    )

    assert abs(f3db_high - fc) <= ft / 2

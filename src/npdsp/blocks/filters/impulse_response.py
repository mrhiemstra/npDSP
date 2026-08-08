import numpy as np

from ...core import Signal


def ideal_allpass(N: int) -> Signal:
    """Ideal all-pass impulse response (sinc centered).

    Parameters
    ----------
    N : int
        Length of the impulse response.

    Returns
    -------
    Signal
        Real-valued impulse response of length N corresponding to an
        ideal all-pass (sinc) kernel centered at (N-1)/2.
    """
    n = np.arange(N) - (N - 1) / 2
    return np.sinc(n)


def lowpass(fc_norm: float, N: int) -> Signal:
    """Ideal lowpass FIR impulse response using sinc.

    Parameters
    ----------
    fc_norm : float
        Normalized cutoff frequency in cycles/sample. Must satisfy
        0 <= fc_norm < 0.5.
    N : int
        Length of the impulse response.

    Returns
    -------
    Signal
        Impulse response (length N) of an ideal lowpass filter with
        normalized cutoff fc_norm. Uses the windowless sinc formula
        scaled by 2*fc_norm.
    """

    assert 0 <= fc_norm < 0.5, "fc_norm should be within [0, 0.5>"

    n = np.arange(N) - (N - 1) / 2
    impulse_response = 2 * fc_norm * np.sinc(2 * fc_norm * n)

    return impulse_response


def highpass(fc_norm: float, N: int) -> Signal:
    """Ideal highpass FIR impulse response.

    Constructed as the difference between an ideal all-pass (sinc)
    impulse response and the corresponding lowpass impulse response.

    Parameters
    ----------
    fc_norm : float
        Normalized cutoff frequency in cycles/sample. Must satisfy
        0 <= fc_norm < 0.5.
    N : int
        Length of the impulse response.

    Returns
    -------
    Signal
        Impulse response (length N) of an ideal highpass filter.
    """

    assert 0 <= fc_norm < 0.5, "fc_norm should be within [0, 0.5>"

    impulse_response = ideal_allpass(N) - lowpass(fc_norm, N)

    return impulse_response


def bandpass(f_low_norm: float, f_high_norm: float, N: int) -> Signal:
    """Ideal bandpass FIR impulse response.

    Constructed by subtracting two lowpass responses: one with the
    higher cutoff and one with the lower cutoff.

    Parameters
    ----------
    f_low_norm : float
        Lower normalized cutoff frequency (cycles/sample). 0 <= f_low_norm < 0.5
    f_high_norm : float
        Upper normalized cutoff frequency (cycles/sample). 0 <= f_high_norm < 0.5
    N : int
        Length of the impulse response.

    Returns
    -------
    Signal
        Impulse response (length N) of an ideal bandpass filter.
    """

    assert 0 <= f_low_norm < 0.5, "f_low_norm should be within [0, 0.5>"
    assert 0 <= f_high_norm < 0.5, "f_high_norm should be within [0, 0.5>"

    impulse_response = lowpass(f_high_norm, N) - lowpass(f_low_norm, N)

    return impulse_response


def bandstop(f_low_norm: float, f_high_norm: float, N: int) -> Signal:
    """Ideal bandstop (notch) FIR impulse response.

    Constructed by subtracting the bandpass response from an ideal
    all-pass (sinc) impulse response.

    Parameters
    ----------
    f_low_norm : float
        Lower normalized cutoff frequency (cycles/sample). 0 <= f_low_norm < 0.5
    f_high_norm : float
        Upper normalized cutoff frequency (cycles/sample). 0 <= f_high_norm < 0.5
    N : int
        Length of the impulse response.

    Returns
    -------
    Signal
        Impulse response (length N) of an ideal bandstop filter.
    """

    assert 0 <= f_low_norm < 0.5, "f_low_norm should be within [0, 0.5>"
    assert 0 <= f_high_norm < 0.5, "f_high_norm should be within [0, 0.5>"

    impulse_response = ideal_allpass(N) - bandpass(f_low_norm, f_high_norm, N)

    return impulse_response

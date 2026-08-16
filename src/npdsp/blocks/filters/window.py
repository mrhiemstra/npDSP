"""Window functions.

Each window class is callable by constructing the class with the desired
length N and returns a 1-D numpy array (Signal) containing the window
coefficients.

All functions accept:
 - N (int): number of points in the window

They return:
 - Signal: numpy array of length N with window coefficients
"""

import numpy as np

from npdsp.core import Signal


class Window:
    """Base window: returns zeros.

    Parameters
    ----------
    N : int
        Number of samples in the window.

    Returns
    -------
    Signal
        Array of zeros of length N.

    """

    def __new__(cls, N: int) -> Signal:  # noqa: N803
        """Return a window of length N, base class returns zeros."""
        return np.zeros(N)


class Rectangular(Window):
    """Rectangular (boxcar) window: all ones."""

    def __new__(cls, N: int) -> Signal:  # noqa: N803
        """Return the rectangular window of length N."""
        return np.ones(N)


class Triangular(Window):
    """Triangular (Bartlett) window.

    Produces a symmetric triangular shape with peak 1 at the center.
    """

    def __new__(cls, N: int) -> Signal:  # noqa: N803
        """Return the triangular window of length N."""
        n = np.arange(N) - (N - 1) / 2
        return 1 - np.abs(n) / ((N - 1) / 2)


class Hanning(Window):
    """Hanning (raised cosine) window.

    Formula: w[n] = 0.5 + 0.5*cos(pi*n/N) with n centered about (N-1)/2.
    """

    def __new__(cls, N: int) -> Signal:  # noqa: N803
        """Return the Hanning window of length N."""
        n = np.arange(N) - (N - 1) / 2
        return 0.5 + 0.5 * np.cos(n * np.pi / N)


class Hamming(Window):
    """Hamming window.

    Formula: w[n] = 0.54 + 0.46*cos(pi*n/N) with n centered about (N-1)/2.
    """

    def __new__(cls, N: int) -> Signal:  # noqa: N803
        """Return the Hamming window of length N."""
        n = np.arange(N) - (N - 1) / 2
        return 0.54 + 0.46 * np.cos(n * np.pi / N)


class Blackman(Window):
    """Blackman window.

    Uses the common 0.42/0.5/0.08 coefficient form.
    """

    def __new__(cls, N: int) -> Signal:  # noqa: N803
        """Return the Blackman window of length N."""
        n = np.arange(N) - (N - 1) / 2
        return (
            0.42
            + 0.5 * np.cos(n * np.pi / N)
            + 0.08 * np.cos(2 * n * np.pi / N)
        )

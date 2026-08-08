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

from ...core import Signal


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

    def __new__(cls, N: int) -> Signal:
        return np.zeros(N)


class rectangular(Window):
    """Rectangular (boxcar) window: all ones."""

    def __new__(cls, N: int) -> Signal:
        return np.ones(N)


class triangular(Window):
    """Triangular (Bartlett) window.

    Produces a symmetric triangular shape with peak 1 at the center.
    """

    def __new__(cls, N: int) -> Signal:
        n = np.arange(N) - (N - 1) / 2
        w = 1 - np.abs(n) / ((N - 1) / 2)
        return w


class hanning(Window):
    """Hanning (raised cosine) window.

    Formula: w[n] = 0.5 + 0.5*cos(pi*n/N) with n centered about (N-1)/2.
    """

    def __new__(cls, N: int) -> Signal:
        n = np.arange(N) - (N - 1) / 2
        w = 0.5 + 0.5 * np.cos(n * np.pi / N)
        return w


class hamming(Window):
    """Hamming window.

    Formula: w[n] = 0.54 + 0.46*cos(pi*n/N) with n centered about (N-1)/2.
    """

    def __new__(cls, N: int) -> Signal:
        n = np.arange(N) - (N - 1) / 2
        w = 0.54 + 0.46 * np.cos(n * np.pi / N)
        return w


class blackman(Window):
    """Blackman window.

    Uses the common 0.42/0.5/0.08 coefficient form.
    """

    def __new__(cls, N: int) -> Signal:
        n = np.arange(N) - (N - 1) / 2
        w = 0.42 + 0.5 * np.cos(n * np.pi / N) + 0.08 * np.cos(2 * n * np.pi / N)
        return w

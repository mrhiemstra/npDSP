"""Filter design blocks for FIR pipeline construction."""

from collections.abc import Callable

import numpy as np

from npdsp.blocks.fir import FIR
from npdsp.core import Pipeline, Signal

from . import impulse_response


class Lowpass(Pipeline):
    """Design a lowpass FIR filter block.

    Parameters
    ----------
    fc : float
        Cutoff frequency. If ``f_norm`` is True, this is normalized [0, 1].
        Otherwise it is interpreted as a frequency in Hz when a sample rate
        is available.
    ft_or_n : float | int
        Transition width or coefficient length, depending on
        ``use_fixed_coef_len``.
    f_norm : bool
        Whether the frequency parameters are normalized.
    window : Window
        Window function used to shape the ideal impulse response.
        Must be window from ``npdsp.blocks.filters.window``.
    use_fixed_coef_len : bool, optional
        If True, ``ft_or_n`` is the exact number of coefficients.
    name : str | None, optional
        Optional pipeline name.
    allow_even_n : bool, optional
        Allow an even number of coefficients when True.

    """

    def __init__(
        self,
        fc: float,
        ft_or_n: float,
        f_norm: bool,
        window: Callable[[int], Signal],
        use_fixed_coef_len: bool = False,
        name: str | None = None,
        allow_even_n: bool = False,
    ) -> None:
        """Initialize a lowpass FIR filter block."""
        self._fc = fc
        self._ft_or_n = ft_or_n
        self._f_norm = f_norm
        self._use_fixed_coef_len = use_fixed_coef_len
        self.window = window
        self.allow_even_n = allow_even_n

        self.fir = FIR(self.coefs, f"{name}_fir")

        super().__init__(self.fir, name=name)

    @property
    def fc(self) -> float:
        """Cutoff frequency after normalization if necessary."""
        if self._f_norm:
            return self._fc
        if self.fir.sample_rate is not None:
            return self._fc / self.fir.sample_rate
        return self._fc

    @property
    def ft(self) -> float | None:
        """Transition width after normalization, or None for fixed length."""
        if self._use_fixed_coef_len:
            return None
        if self._f_norm:
            return self._ft_or_n
        if self.fir.sample_rate is not None:
            return self._ft_or_n / self.fir.sample_rate
        return self._ft_or_n

    @property
    def num_coefs(self) -> int:
        """Number of filter coefficients used for design."""
        if self.ft is None:
            assert not isinstance(self._ft_or_n, float)
            return self._ft_or_n

        num_coefs = int(np.ceil(4 / self.ft))

        if not (num_coefs % 2) and not self.allow_even_n:
            num_coefs += 1

        return num_coefs

    @property
    def coefs(self) -> Signal:
        """Compute the FIR coefficients using the selected window and response."""
        N = self.num_coefs  # noqa: N806
        w = self.window(N)
        ir = impulse_response.lowpass(self.fc, N)

        c = w * ir
        c /= np.sum(c)

        return c


class Highpass(Pipeline):
    """Design a highpass FIR filter block.

    Parameters
    ----------
    fc : float
        Cutoff frequency. If ``f_norm`` is True, this is normalized [0, 1].
        Otherwise it is interpreted as a frequency in Hz when a sample rate
        is available.
    ft_or_n : float | int
        Transition width or coefficient length, depending on
        ``use_fixed_coef_len``.
    f_norm : bool
        Whether the frequency parameters are normalized.
    window : Window
        Window function used to shape the ideal impulse response.
        Must be window from ``npdsp.blocks.filters.window``.
    use_fixed_coef_len : bool, optional
        If True, ``ft_or_n`` is the exact number of coefficients.
    name : str | None, optional
        Optional pipeline name.
    allow_even_n : bool, optional
        Allow an even number of coefficients when True.

    """

    def __init__(
        self,
        fc: float,
        ft_or_n: float,
        f_norm: bool,
        window: Callable[[int], Signal],
        use_fixed_coef_len: bool = False,
        name: str | None = None,
        allow_even_n: bool = False,
    ) -> None:
        """Initialize a highpass FIR filter block."""
        self._fc = fc
        self._ft_or_n = ft_or_n
        self._f_norm = f_norm
        self._use_fixed_coef_len = use_fixed_coef_len
        self.window = window
        self.allow_even_n = allow_even_n

        self.fir = FIR(self.coefs, f"{name}_fir")

        super().__init__(self.fir, name=name)

    @property
    def fc(self) -> float:
        """Highpass cutoff frequency after normalization if necessary."""
        if self._f_norm:
            return self._fc
        if self.fir.sample_rate is not None:
            return self._fc / self.fir.sample_rate
        return self._fc

    @property
    def ft(self) -> float | None:
        """Transition width after normalization, or None for fixed length."""
        if self._use_fixed_coef_len:
            return None
        if self._f_norm:
            return self._ft_or_n
        if self.fir.sample_rate is not None:
            return self._ft_or_n / self.fir.sample_rate
        return self._ft_or_n

    @property
    def num_coefs(self) -> int:
        """Number of filter coefficients used for design."""
        if self.ft is None:
            assert not isinstance(self._ft_or_n, float)
            return self._ft_or_n

        num_coefs = int(np.ceil(4 / self.ft))

        if not (num_coefs % 2) and not self.allow_even_n:
            num_coefs += 1

        return num_coefs

    @property
    def coefs(self) -> Signal:
        """Compute the highpass filter coefficients."""
        N = self.num_coefs  # noqa: N806
        w = self.window(N)
        ir = impulse_response.highpass(self.fc, N)

        c = w * ir

        mag_idx = -1 if N % 2 == 1 else int(N * (0.5 + self.fc) / 2)

        c /= np.abs(np.fft.rfft(c)[mag_idx])

        return c


class Bandpass(Pipeline):
    """Design a bandpass FIR filter block.

    Parameters
    ----------
    fc1 : float
        Lower cutoff frequency. If ``f_norm`` is True, this is normalized [0, 1].
        Otherwise it is interpreted as a frequency in Hz when a sample rate
        is available.
    fc2 : float
        Upper cutoff frequency. If ``f_norm`` is True, this is normalized [0, 1].
        Otherwise it is interpreted as a frequency in Hz when a sample rate
        is available.
    ft_or_n : float | int
        Transition width or coefficient length, depending on
        ``use_fixed_coef_len``.
    f_norm : bool
        Whether the frequency parameters are normalized.
    window : Window
        Window function used to shape the ideal impulse response.
        Must be window from ``npdsp.blocks.filters.window``.
    use_fixed_coef_len : bool, optional
        If True, ``ft_or_n`` is the exact number of coefficients.
    name : str | None, optional
        Optional pipeline name.
    allow_even_n : bool, optional
        Allow an even number of coefficients when True.

    """

    def __init__(
        self,
        fc1: float,
        fc2: float,
        ft_or_n: float,
        f_norm: bool,
        window: Callable[[int], Signal],
        use_fixed_coef_len: bool = False,
        name: str | None = None,
        allow_even_n: bool = False,
    ) -> None:
        """Initialize a bandpass FIR filter block."""
        self._fc1 = fc1
        self._fc2 = fc2
        self._ft_or_n = ft_or_n
        self._f_norm = f_norm
        self._use_fixed_coef_len = use_fixed_coef_len
        self.window = window
        self.allow_even_n = allow_even_n

        self.fir = FIR(self.coefs, f"{name}_fir")

        super().__init__(self.fir, name=name)

    @property
    def fc1(self) -> float:
        """Lower band edge frequency after normalization if needed."""
        if self._f_norm:
            return self._fc1
        if self.fir.sample_rate is not None:
            return self._fc1 / self.fir.sample_rate
        return self._fc1

    @property
    def fc2(self) -> float:
        """Upper band edge frequency after normalization if needed."""
        if self._f_norm:
            return self._fc2
        if self.fir.sample_rate is not None:
            return self._fc2 / self.fir.sample_rate
        return self._fc2

    @property
    def ft(self) -> float | None:
        """Transition width after normalization, or None for fixed length."""
        if self._use_fixed_coef_len:
            return None
        if self._f_norm:
            return self._ft_or_n
        if self.fir.sample_rate is not None:
            return self._ft_or_n / self.fir.sample_rate
        return self._ft_or_n

    @property
    def num_coefs(self) -> int:
        """Number of filter coefficients used for design."""
        if self.ft is None:
            assert not isinstance(self._ft_or_n, float)
            return self._ft_or_n

        num_coefs = int(np.ceil(4 / self.ft))

        if not (num_coefs % 2) and not self.allow_even_n:
            num_coefs += 1

        return num_coefs

    @property
    def coefs(self) -> Signal:
        """Compute the bandpass filter coefficients."""
        N = self.num_coefs  # noqa: N806
        w = self.window(N)
        ir = impulse_response.bandpass(self.fc1, self.fc2, N)

        c = w * ir
        c /= np.abs(np.fft.rfft(c)[int(N * (self.fc1 + self.fc2) / 2)])

        return c


class Bandstop(Pipeline):
    """Design a bandstop FIR filter block.

    Parameters
    ----------
    fc1 : float
        Lower cutoff frequency. If ``f_norm`` is True, this is normalized [0, 1].
        Otherwise it is interpreted as a frequency in Hz when a sample rate
        is available.
    fc2 : float
        Upper cutoff frequency. If ``f_norm`` is True, this is normalized [0, 1].
        Otherwise it is interpreted as a frequency in Hz when a sample rate
        is available.
    ft_or_n : float | int
        Transition width or coefficient length, depending on
        ``use_fixed_coef_len``.
    f_norm : bool
        Whether the frequency parameters are normalized.
    window : Window
        Window function used to shape the ideal impulse response.
        Must be window from ``npdsp.blocks.filters.window``.
    use_fixed_coef_len : bool, optional
        If True, ``ft_or_n`` is the exact number of coefficients.
    name : str | None, optional
        Optional pipeline name.
    allow_even_n : bool, optional
        Allow an even number of coefficients when True.

    """

    def __init__(
        self,
        fc1: float,
        fc2: float,
        ft_or_n: float,
        f_norm: bool,
        window: Callable[[int], Signal],
        use_fixed_coef_len: bool = False,
        name: str | None = None,
        allow_even_n: bool = False,
    ) -> None:
        """Initialize a bandstop FIR filter block."""
        self._fc1 = fc1
        self._fc2 = fc2
        self._ft_or_n = ft_or_n
        self._f_norm = f_norm
        self._use_fixed_coef_len = use_fixed_coef_len
        self.window = window
        self.allow_even_n = allow_even_n

        self.fir = FIR(self.coefs, f"{name}_fir")

        super().__init__(self.fir, name=name)

    @property
    def fc1(self) -> float:
        """Lower stopband frequency after normalization if needed."""
        if self._f_norm:
            return self._fc1
        if self.fir.sample_rate is not None:
            return self._fc1 / self.fir.sample_rate
        return self._fc1

    @property
    def fc2(self) -> float:
        """Upper stopband frequency after normalization if needed."""
        if self._f_norm:
            return self._fc2
        if self.fir.sample_rate is not None:
            return self._fc2 / self.fir.sample_rate
        return self._fc2

    @property
    def ft(self) -> float | None:
        """Transition width after normalization, or None for fixed length."""
        if self._use_fixed_coef_len:
            return None
        if self._f_norm:
            return self._ft_or_n
        if self.fir.sample_rate is not None:
            return self._ft_or_n / self.fir.sample_rate
        return self._ft_or_n

    @property
    def num_coefs(self) -> int:
        """Number of filter coefficients used for design."""
        if self.ft is None:
            assert not isinstance(self._ft_or_n, float)
            return self._ft_or_n

        num_coefs = int(np.ceil(4 / self.ft))

        if not (num_coefs % 2) and not self.allow_even_n:
            num_coefs += 1

        return num_coefs

    @property
    def coefs(self) -> Signal:
        """Compute the bandstop filter coefficients."""
        N = self.num_coefs  # noqa: N806
        w = self.window(N)
        ir = impulse_response.bandstop(self.fc1, self.fc2, N)

        c = w * ir
        c /= np.sum(c)

        return c

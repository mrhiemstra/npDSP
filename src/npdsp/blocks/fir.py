from __future__ import annotations

from collections.abc import Callable

import numba
import numpy as np

from ..core import Block, Signal, SignalLike

ProcessFn = Callable[[Signal, Signal, Signal], Signal]


class FIR(Block):
    """Finite impulse response filter with Numba JIT compilation.

    Uses Numba's JIT to compile to machine code on first process() call.
    Filtering is performed along the final axis of the input signal.
    Leading dimensions represent independent signals and follow NumPy
    broadcasting rules.

    The final axis is the sample axis. Leading dimensions identify independent
    signals. Once processing has started, leading dimensions cannot change;
    the sample-axis length may vary between calls.

    Parameters
    ----------
    coefs : array-like
        FIR coefficients. The final axis represents the filter taps.
        Leading dimensions may be used to specify independent filters.
    name : str, optional
        Optional name used to identify the block.

    Notes
    -----
    **Compilation:** The first call to ``process()`` compiles the JIT-compiled
    function for the given filter order and data dtype (~50-100ms). Subsequent
    calls reuse the compiled code, and handle any chunk size transparently.

    Examples
    --------
    A single filter can be applied to multiple signals::

        fir = FIR([1, 2, 3])
        y = fir(x)

    where ``x`` may have shape ``(N,)`` or ``(..., N)``.

    Independent filters can be specified using leading dimensions::

        fir = FIR(
            [
                [1, 2, 3],
                [4, 5, 6],
            ]
        )
    """

    def __init__(
        self,
        coefs: SignalLike,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)

        self.coefs = np.asarray(coefs)

        if self.coefs.size == 0:
            raise ValueError("FIR coefficients cannot be empty")

        if self.coefs.ndim == 0:
            raise ValueError("FIR coefficients must be at least one-dimensional")

        taps = self.coefs.shape[-1]

        # Coefficients are reversed once up front: FIR convolution needs coefs
        # applied in the same order as the history window (oldest-to-newest),
        # while FIR coefficients are conventionally given newest-tap-first.
        self._coefs_rev = self.coefs[..., ::-1]

        self._channels: int | None = None
        self._input_shape: tuple[int, ...] | None = None
        self._coefs_ch: Signal | None = None
        self._state: Signal | None = None
        self._dtype: np.dtype | None = None
        self._taps = taps

        # JIT-compiled function, compiled on first process() call
        self._jitted_process: ProcessFn | None = None

    @property
    def stateful(self) -> bool:
        """Whether the filter maintains state between calls."""
        return True

    @property
    def latency_samples(self) -> float:
        return (self._taps - 1) / 2

    @property
    def type(self) -> int | None:
        """
        Determine the type of FIR filter based on symmetry properties.

        Type 1: Even order, symmetric (h[n] = h[N-1-n])
        Type 2: Odd order, symmetric (h[n] = h[N-1-n])
        Type 3: Even order, antisymmetric (h[n] = -h[N-1-n])
        Type 4: Odd order, antisymmetric (h[n] = -h[N-1-n])

        Returns
        -------
        int
            Filter type (1, 2, 3, or 4). Returns None if filter doesn't match any type.
        """
        N = len(self.coefs)
        tol = 1e-10

        # Check symmetry
        is_symmetric = np.allclose(self.coefs, self._coefs_rev, atol=tol)
        is_antisymmetric = np.allclose(self.coefs, -self._coefs_rev, atol=tol)

        if not (is_symmetric or is_antisymmetric):
            return None  # Not a linear-phase filter

        # Determine order (even or odd)
        # Even order = odd number of taps (N is odd)
        # Odd order = even number of taps (N is even)
        is_even_order = N % 2 == 1

        if is_symmetric:
            return 1 if is_even_order else 2
        else:  # antisymmetric
            return 3 if is_even_order else 4

    def reset(self) -> None:
        """Clear all retained filter state."""
        self._channels = None
        self._input_shape = None
        self._dtype = None
        self._state = None
        self._coefs_ch = None
        # Don't reset _jitted_process; recompile only if needed

    def process(self, x: Signal) -> Signal:
        """Process a signal through the FIR filter.

        On the first call, the filter compiles the JIT-compiled function for
        the given number of taps and data dtype. On subsequent calls, the
        compiled code is reused.

        Parameters
        ----------
        x : array-like
            Input signal. Can be 1D (single channel) or ND (multiple channels
            along leading dimensions, samples on final axis).

        Returns
        -------
        ndarray
            Filtered output, same shape as input.
        """
        x = np.asarray(x)

        if x.ndim == 0:
            raise ValueError("FIR input must be at least one-dimensional")

        taps = self._taps

        if taps == 1:
            # No history needed: a single-tap FIR is a pure elementwise gain.
            return x * self.coefs

        one_dimensional = x.ndim == 1

        if one_dimensional:
            x_work = x[np.newaxis, :]
        else:
            channels = int(np.prod(x.shape[:-1]))
            x_work = x.reshape(channels, x.shape[-1])

        channels = x_work.shape[0]

        if self._channels is None:
            self._channels = channels
        elif self._channels != channels:
            raise ValueError("Input channel count changed")

        coefs = self.coefs

        if coefs.ndim == 1:
            coefs_ch = np.broadcast_to(coefs, (channels, taps))
        elif coefs.ndim == 2:
            if coefs.shape[0] == 1:
                coefs_ch = np.broadcast_to(coefs, (channels, taps))
            else:
                # Multiple independent filters
                if coefs.shape[0] != channels:
                    raise ValueError(
                        f"Coefficient channel count ({coefs.shape[0]}) "
                        f"does not match input ({channels})"
                    )
                coefs_ch = coefs
        elif coefs.ndim > 2:
            # Flatten leading dimensions to match channels
            coefs_flat = coefs.reshape(-1, taps)
            if coefs_flat.shape[0] != channels:
                raise ValueError(
                    f"Coefficient channel count ({coefs_flat.shape[0]}) "
                    f"does not match input ({channels})"
                )
            coefs_ch = coefs_flat
        else:
            raise ValueError("Coefficients must be 1D or 2D")

        dtype = np.result_type(
            x.dtype,
            self.coefs.dtype,
            np.float64,
        )

        if self._state is None:
            self._state = np.zeros((channels, taps - 1), dtype=dtype)
            self._dtype = dtype

        # Convert to dtype
        x_work = x_work.astype(dtype, copy=False)
        coefs_ch = coefs_ch.astype(dtype, copy=False)

        if self._jitted_process is None:
            # First call: compile JIT for this tap count and dtype
            self._jitted_process = self._make_jitted_process(taps, dtype)

        assert self._jitted_process is not None
        assert self._state is not None

        y = self._jitted_process(x_work, coefs_ch, self._state)

        return y[0] if one_dimensional else y.reshape(x.shape)

    @staticmethod
    def _make_jitted_process(
        taps: int,
        dtype: np.dtype,
    ) -> ProcessFn:
        """Create tap-specific JIT-compiled process function."""

        if taps == 1:

            @numba.njit
            def jit_taps1(x_work: Signal, coefs: Signal, state: Signal) -> Signal:
                y = np.empty_like(x_work)
                y[:, :] = coefs[:, 0:1] * x_work
                return y

            return jit_taps1

        elif taps == 2:

            @numba.njit
            def jit_taps2(x_work: Signal, coefs: Signal, state: Signal) -> Signal:
                channels, n_samples = x_work.shape
                y = np.empty_like(x_work)
                for c in range(channels):
                    s0 = state[c, 0]
                    for n in range(n_samples):
                        xn = x_work[c, n]
                        yn = coefs[c, 0] * xn + coefs[c, 1] * s0
                        y[c, n] = yn
                        s0 = xn
                    state[c, 0] = s0
                return y

            return jit_taps2

        else:  # taps > 2

            @numba.njit
            def jit_taps_general(
                x_work: Signal, coefs: Signal, state: Signal
            ) -> Signal:
                channels, n_samples = x_work.shape
                taps_val = coefs.shape[1]
                y = np.empty_like(x_work)

                for c in range(channels):
                    for n in range(n_samples):
                        xn = x_work[c, n]
                        yn = coefs[c, 0] * xn
                        for k in range(1, taps_val):
                            yn += coefs[c, k] * state[c, taps_val - k - 1]
                        y[c, n] = yn

                        for i in range(taps_val - 2):
                            state[c, i] = state[c, i + 1]
                        state[c, taps_val - 2] = xn

                return y

            return jit_taps_general

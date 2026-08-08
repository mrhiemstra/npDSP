from __future__ import annotations

import numpy as np

from ..core import Block, Signal, SignalLike, SlidingBuffer


class FIR(Block):
    """Finite impulse response filter.

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

        # Coefficients are reversed once up front: convolution via
        # sliding windows needs coefs applied in the *same* order as the
        # window (oldest-to-newest), while FIR coefficients are conventionally
        # given newest-tap-first. Caching this avoids re-deriving it per call
        self._coefs_rev = self.coefs[..., ::-1]

        self._history = SlidingBuffer(max(taps - 1, 0), axis=-1)
        self._coefs_rev_bc: np.ndarray | None = None
        self._dtype: np.dtype | None = None

    @property
    def stateful(self) -> bool:
        """Whether the filter maintains state between calls."""
        return True

    @property
    def latency_samples(self) -> float:
        return (len(self.coefs) - 1) / 2

    @property
    def type(self) -> int | None:
        """
        Determine the type of FIR filter based on symmetry properties.

        Type 1: Even order, symmetric (h[n] = h[N-1-n])
        Type 2: Odd order, symmetric (h[n] = h[N-1-n])
        Type 3: Even order, antisymmetric (h[n] = -h[N-1-n])
        Type 4: Odd order, antisymmetric (h[n] = -h[N-1-n])

        Parameters
        ----------
        h : ndarray
            FIR filter coefficients.

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
        """Reset the filter state."""
        self._history.reset()
        self._coefs_rev_bc = None
        self._dtype = None

    def process(self, x: Signal) -> Signal:
        """Filter the input signal.

        Filtering is performed along the final axis. The filter state is
        preserved between calls so that separately processed chunks produce
        the same result as processing the concatenated signal.
        """
        if x.ndim == 0:
            raise ValueError("FIR input must be at least one-dimensional")

        taps = self.coefs.shape[-1]

        if taps == 1:
            # No history needed: a single-tap FIR is a pure elementwise gain.
            return x * self.coefs

        state_shape = np.broadcast_shapes(x.shape[:-1], self.coefs.shape[:-1])

        if self._dtype is None:
            self._dtype = np.result_type(x, self.coefs)
            self._coefs_rev_bc = np.broadcast_to(
                self._coefs_rev.astype(self._dtype, copy=False),
                (*state_shape, taps),
            ).copy()
        dtype = self._dtype

        x = np.broadcast_to(x.astype(dtype, copy=False), (*state_shape, x.shape[-1]))

        self._history.prepare((*state_shape, taps - 1), dtype)
        extended = self._history.extend(x)

        windows = np.lib.stride_tricks.sliding_window_view(extended, taps, axis=-1)
        # windows shape: (*state_shape, x.shape[-1], taps); coefs_rev_bc shape: (*state_shape, taps)
        assert self._coefs_rev_bc is not None
        y = np.einsum("...t,...nt->...n", self._coefs_rev_bc, windows)

        return y.astype(dtype, copy=False)

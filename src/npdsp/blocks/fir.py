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

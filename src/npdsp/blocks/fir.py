from __future__ import annotations

import numpy as np

from ..core import Block, Signal, SignalLike



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

        self._state: np.ndarray | None = None

    @property
    def stateful(self) -> bool:
        """Whether the filter maintains state between calls."""
        return True

    def reset(self) -> None:
        """Reset the filter state."""
        self._state = None

    def process(self, x: Signal) -> Signal:
        """Filter the input signal.

        Filtering is performed along the final axis. The filter state is
        preserved between calls so that separately processed chunks produce
        the same result as processing the concatenated signal.
        """

        if x.ndim == 0:
            raise ValueError("FIR input must be at least one-dimensional")

        taps = self.coefs.shape[-1]

        if self._state is None:
            state_shape = np.broadcast_shapes(
                x.shape[:-1],
                self.coefs.shape[:-1],
            )

            self._state = np.zeros(
                (*state_shape, max(taps - 1, 0)),
                dtype=np.result_type(x, self.coefs),
            )
        else:
            state_shape = self._state.shape[:-1]

            if x.shape[:-1] != state_shape:
                raise ValueError(
                    "FIR input leading dimensions cannot change between calls: "
                    f"expected {state_shape}, got {x.shape[:-1]}"
                )

        if taps == 1:
            y = x * self.coefs

            self._state = np.empty(
                (*state_shape, 0),
                dtype=np.result_type(x, self.coefs),
            )

            return y

        dtype = np.result_type(
            x,
            self.coefs,
            self._state,
        )

        state = self._state.astype(dtype, copy=False)
        x = x.astype(dtype, copy=False)
        coefs = self.coefs.astype(dtype, copy=False)

        state = np.broadcast_to(
            state,
            (*state_shape, taps - 1),
        )

        x = np.broadcast_to(
            x,
            (*state_shape, x.shape[-1]),
        )

        coefs = np.broadcast_to(
            coefs,
            (*state_shape, taps),
        )

        extended = np.concatenate(
            [state, x],
            axis=-1,
        )

        y = np.zeros_like(
            x,
            dtype=dtype,
        )

        for k in range(taps):
            y += coefs[..., k, np.newaxis] * extended[
                ...,
                taps - 1 - k : taps - 1 - k + x.shape[-1],
            ]

        self._state = extended[..., -(taps - 1) :].copy()

        return y


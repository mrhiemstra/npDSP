from __future__ import annotations

import numpy as np

from ..core import Block, Signal, SignalLike


class IIR(Block):
    """Infinite impulse response filter.

    Parameters
    ----------
    b : array-like
        Feed-forward coefficients. A one-dimensional array is shared across
        all channels. A two-dimensional array specifies separate coefficients
        for each channel.
    a : array-like
        Feedback coefficients. ``a[0]`` must be non-zero. A one-dimensional
        array is shared across all channels. A two-dimensional array specifies
        separate coefficients for each channel.
    name : str, optional
        Optional name used to identify the filter.

    Notes
    -----
    The filter implements the difference equation

    ``a[0] * y[n] = sum(b[k] * x[n-k]) - sum(a[k] * y[n-k])``

    Missing samples before the beginning of the input are treated as zero.

    Filter state is retained between calls, making the block stateful.
    :meth:`reset` clears all retained input and output history.

    One-dimensional coefficient arrays are shared between all input channels.
    Two-dimensional coefficient arrays specify one coefficient set per
    channel.
    """

    def __init__(
        self,
        b: SignalLike,
        a: SignalLike,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)

        self.b = np.asarray(b)
        self.a = np.asarray(a)

        if self.b.ndim not in (1, 2):
            raise ValueError("b must be one- or two-dimensional")

        if self.a.ndim not in (1, 2):
            raise ValueError("a must be one- or two-dimensional")

        if self.b.size == 0:
            raise ValueError("b cannot be empty")

        if self.a.size == 0:
            raise ValueError("a cannot be empty")

        if np.any(self.a[..., 0] == 0):
            raise ValueError("a[0] must be non-zero")

        if self.b.ndim == 2 and self.a.ndim == 2 and self.b.shape[0] != self.a.shape[0]:
            raise ValueError("b and a must have the same number of channels")

        self._input_state: np.ndarray | None = None
        self._output_state: np.ndarray | None = None
        self._input_shape: tuple[int, ...] | None = None

    @property
    def stateful(self) -> bool:
        """Whether the filter maintains state between calls."""
        return True

    def _validate_input_shape(self, x: Signal) -> None:
        """Validate and initialize the fixed input shape."""
        if x.ndim == 1:
            shape = ()
        else:
            shape = x.shape[:-1]

        if self._input_shape is None:
            self._input_shape = shape
        elif shape != self._input_shape:
            raise ValueError(
                f"Input leading shape changed from {self._input_shape} to {shape}"
            )

    def _channel_coefficients(
        self,
        coefficients: np.ndarray,
        channels: int,
    ) -> np.ndarray:
        """Return coefficients expanded to one row per channel."""
        if coefficients.ndim == 1:
            return np.broadcast_to(
                coefficients,
                (channels, coefficients.shape[0]),
            )

        if coefficients.shape[0] != channels:
            raise ValueError(
                f"Expected {channels} coefficient channels, got {coefficients.shape[0]}"
            )

        return coefficients

    def _initialize_state(
        self,
        x: Signal,
        channels: int,
    ) -> None:
        """Initialize delay state for the current input shape."""
        input_order = max(self.b.shape[-1] - 1, 0)
        output_order = max(self.a.shape[-1] - 1, 0)

        dtype = np.result_type(
            x.dtype,
            self.b.dtype,
            self.a.dtype,
            np.float64,
        )

        self._input_state = np.zeros(
            (channels, input_order),
            dtype=dtype,
        )

        self._output_state = np.zeros(
            (channels, output_order),
            dtype=dtype,
        )

    def process(self, x: Signal) -> Signal:
        """Process a signal through the IIR filter."""
        x = np.asarray(x)

        self._validate_input_shape(x)

        one_dimensional = x.ndim == 1

        if one_dimensional:
            x_work = x[np.newaxis, :]
        else:
            channels = int(np.prod(x.shape[:-1]))
            x_work = x.reshape(channels, x.shape[-1])

        channels = x_work.shape[0]

        b = self._channel_coefficients(self.b, channels)
        a = self._channel_coefficients(self.a, channels)

        dtype = np.result_type(
            x.dtype,
            self.b.dtype,
            self.a.dtype,
            np.float64,
        )

        if self._input_state is None or self._output_state is None:
            self._initialize_state(x, channels)

        assert self._input_state is not None
        assert self._output_state is not None

        if (
            self._input_state.shape[0] != channels
            or self._output_state.shape[0] != channels
        ):
            raise ValueError("Input channel count changed")

        x_work = x_work.astype(dtype, copy=False)

        y = np.empty(x_work.shape, dtype=dtype)

        for channel in range(channels):
            x_history = self._input_state[channel]
            y_history = self._output_state[channel]

            for n in range(x_work.shape[1]):
                value = b[channel, 0] * x_work[channel, n]

                for k in range(1, b.shape[1]):
                    if k <= len(x_history):
                        value += b[channel, k] * x_history[-k]

                for k in range(1, a.shape[1]):
                    if k <= len(y_history):
                        value -= a[channel, k] * y_history[-k]

                value /= a[channel, 0]

                y[channel, n] = value

                if len(x_history):
                    x_history[:-1] = x_history[1:]
                    x_history[-1] = x_work[channel, n]

                if len(y_history):
                    y_history[:-1] = y_history[1:]
                    y_history[-1] = value

        if one_dimensional:
            return y[0]

        return y.reshape(x.shape)

    def reset(self) -> None:
        """Clear all input and output filter state."""
        self._input_state = None
        self._output_state = None
        self._input_shape = None

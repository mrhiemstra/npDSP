from collections.abc import Callable

import numba
import numpy as np

from ..core import Block, Signal, SignalLike


class IIR(Block):
    """Infinite impulse response filter with Numba JIT compilation.

    Uses Numba's JIT to compile to machine code on first process() call.

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
        Optional name used to identify the block within a pipeline.

    Notes
    -----
    The filter implements the difference equation

    ``a[0] * y[n] = sum(b[k] * x[n-k]) - sum(a[k] * y[n-k])``

    Missing samples before the beginning of the input are treated as zero.

    Filter state is retained between calls, making the block stateful.
    :meth:`reset` clears all retained state.

    One-dimensional coefficient arrays are shared between all input channels.
    Two-dimensional coefficient arrays specify one coefficient set per
    channel. Both ``b`` and ``a`` follow the same broadcasting rule:
    a 2D array with a single row broadcasts across channels.

    **Compilation:** The first call to ``process()`` compiles the JIT-compiled
    function for the given filter order and data dtype (~100-300ms). Subsequent
    calls reuse the compiled code, and handle any chunk size transparently.

    Examples
    --------
    >>> iir = IIR([0.2, 0.4, 0.2], [1.0, -0.5, 0.1])
    >>> y = iir(x)  # First call compiles, subsequent calls are fast
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

        # Validate channel compatibility: both must be 1D, or if both are 2D,
        # they must have matching channel counts OR one of them has shape[0]==1 (broadcast).
        # Reject: one is 1D and the other is 2D with shape[0] > 1
        if self.b.ndim == 2 and self.a.ndim == 1:
            raise ValueError(
                "b is 2D (per-channel) but a is 1D (shared); "
                "both must be 1D or both must be 2D with matching channel counts"
            )
        if self.a.ndim == 2 and self.b.ndim == 1:
            raise ValueError(
                "a is 2D (per-channel) but b is 1D (shared); "
                "both must be 1D or both must be 2D with matching channel counts"
            )
        if (
            self.b.ndim == 2
            and self.a.ndim == 2
            and self.b.shape[0] != self.a.shape[0]
            and self.b.shape[0] != 1
            and self.a.shape[0] != 1
        ):
            raise ValueError(
                f"b has {self.b.shape[0]} channels but a has {self.a.shape[0]} channels; "
                "must match or one must have exactly 1 row for broadcasting"
            )

        order = max(self.b.shape[-1], self.a.shape[-1]) - 1
        self._order = order

        a0 = self.a[..., 0:1]
        b_norm = self.b / a0
        a_norm = self.a / a0

        pad_b = order + 1 - b_norm.shape[-1]
        pad_a = order + 1 - a_norm.shape[-1]

        if b_norm.ndim == 1:
            b_norm = np.pad(b_norm, (0, pad_b))
        else:
            b_norm = np.pad(b_norm, ((0, 0), (0, pad_b)))

        if a_norm.ndim == 1:
            a_norm = np.pad(a_norm, (0, pad_a))
        else:
            a_norm = np.pad(a_norm, ((0, 0), (0, pad_a)))

        self._b_padded = b_norm
        self._a_padded = a_norm

        self._channels: int | None = None
        self._input_shape: tuple[int, ...] | None = None
        self._b_ch: Signal | None = None
        self._a_ch: Signal | None = None
        self._input_state: Signal | None = None
        self._output_state: Signal | None = None
        self._dtype: np.dtype | None = None

        # JIT-compiled function, compiled on first process() call
        self._jitted_process: (
            Callable[[Signal, Signal, Signal, Signal, Signal], Signal] | None
        ) = None

    @property
    def stateful(self) -> bool:
        """Whether the filter maintains state between calls."""
        return True

    def reset(self) -> None:
        """Clear all retained filter state."""
        self._input_shape = None
        self._channels = None
        self._dtype = None
        self._input_state = None
        self._output_state = None
        # Reset _jitted_process ?

    def process(self, x: Signal) -> Signal:
        """Process a signal through the IIR filter.

        On the first call, the filter compiles the JIT-compiled function for
        the given order and data dtype. On subsequent calls, the compiled code
        is reused.

        Parameters
        ----------
        x : array-like
            Input signal. Can be 1D (single channel) or ND (multiple channels
            along leading dimensions).

        Returns
        -------
        ndarray
            Filtered output, same shape as input.
        """
        x = np.asarray(x)

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

        b = self._b_padded
        a = self._a_padded

        if b.ndim == 1:
            b = np.broadcast_to(b, (channels, b.shape[0]))
        elif b.shape[0] == 1:
            b = np.broadcast_to(b, (channels, b.shape[1]))

        if a.ndim == 1:
            a = np.broadcast_to(a, (channels, a.shape[0]))
        elif a.shape[0] == 1:
            a = np.broadcast_to(a, (channels, a.shape[1]))

        dtype = np.result_type(
            x.dtype,
            self.b.dtype,
            self.a.dtype,
            np.float64,
        )

        if self._input_state is None:
            self._input_state = np.zeros((channels, self._order), dtype=dtype)
            self._output_state = np.zeros((channels, self._order), dtype=dtype)
            self._dtype = dtype

        # Convert to dtype (JIT compiles for this dtype on first call)
        x_work = x_work.astype(dtype, copy=False)
        b = b.astype(dtype, copy=False)
        a = a.astype(dtype, copy=False)

        if self._jitted_process is None:
            # First call: compile JIT for this order and dtype
            self._jitted_process = self._make_jitted_process(self._order, dtype)

        assert self._jitted_process is not None
        assert self._input_state is not None
        assert self._output_state is not None

        y = self._jitted_process(x_work, b, a, self._input_state, self._output_state)

        return y[0] if one_dimensional else y.reshape(x.shape)

    @staticmethod
    def _make_jitted_process(
        order: int,
        dtype: np.dtype,
    ) -> Callable[[Signal, Signal, Signal, Signal, Signal], Signal]:
        """Create order-specific JIT-compiled process function."""

        if order == 0:

            @numba.njit
            def jit_order0(
                x_work: Signal,
                b: Signal,
                a: Signal,
                input_state: Signal,
                output_state: Signal,
            ) -> Signal:
                #  channels, n_samples = x_work.shape
                y = np.empty_like(x_work)
                y[:, :] = b[:, 0:1] * x_work
                return y

            return jit_order0

        elif order == 1:

            @numba.njit
            def jit_order1(
                x_work: Signal,
                b: Signal,
                a: Signal,
                input_state: Signal,
                output_state: Signal,
            ) -> Signal:
                channels, n_samples = x_work.shape
                y = np.empty_like(x_work)
                for c in range(channels):
                    w0 = input_state[c, 0]
                    y0 = output_state[c, 0]
                    for n in range(n_samples):
                        xn = x_work[c, n]
                        yn = b[c, 0] * xn + w0
                        y[c, n] = yn
                        w0 = b[c, 1] * xn - a[c, 1] * yn
                        y0 = yn
                    input_state[c, 0] = w0
                    output_state[c, 0] = y0
                return y

            return jit_order1

        else:  # order > 1

            @numba.njit
            def jit_general(
                x_work: Signal,
                b: Signal,
                a: Signal,
                input_state: Signal,
                output_state: Signal,
            ) -> Signal:
                channels, n_samples = x_work.shape
                order_val = input_state.shape[1]
                y = np.empty_like(x_work)
                for c in range(channels):
                    for n in range(n_samples):
                        xn = x_work[c, n]
                        yn = b[c, 0] * xn + input_state[c, 0]
                        y[c, n] = yn
                        for i in range(order_val - 1):
                            input_state[c, i] = (
                                b[c, i + 1] * xn
                                - a[c, i + 1] * yn
                                + input_state[c, i + 1]
                            )
                        input_state[c, order_val - 1] = (
                            b[c, order_val] * xn - a[c, order_val] * yn
                        )
                        for i in range(order_val - 1):
                            output_state[c, i] = output_state[c, i + 1]
                        output_state[c, order_val - 1] = yn
                return y

            return jit_general

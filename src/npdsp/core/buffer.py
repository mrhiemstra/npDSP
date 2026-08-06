from __future__ import annotations

import numpy as np

from .typing import Signal


class SlidingBuffer:
    """Shared fixed-length trailing-history buffer for streaming blocks.

    Retains the last ``size`` samples along a chosen axis so a block can be
    fed arbitrary chunks and still see correct trailing history. Centralizes
    the leading-shape/dtype validation for ``Delay`` and ``FIR``.

    Parameters
    ----------
    size : int
        Number of trailing samples to retain. Must be non-negative.
    axis : int, default=-1
        Sample axis along which history is tracked. ``Delay`` uses axis 0;
        ``FIR``/``IIR`` use the final axis.

    Notes
    -----
    This still allocates a new array per call (via ``np.concatenate``) to
    build the history+input window needed for vectorized tap/convolution
    math. A true zero-allocation ring buffer would require abandoning that
    vectorization in favor of per-sample writes (as IIR now does for its
    recursive part).
    """

    def __init__(self, size: int, axis: int = -1) -> None:
        if size < 0:
            raise ValueError("size must be non-negative")

        self.size = size
        self.axis = axis
        self._buffer: Signal | None = None
        self._leading_shape: tuple[int, ...] | None = None

    @property
    def initialized(self) -> bool:
        """Whether the buffer has been sized against an input yet."""
        return self._buffer is not None

    def _leading_shape_of(self, shape: tuple[int, ...]) -> tuple[int, ...]:
        dims = list(shape)
        del dims[self.axis]
        return tuple(dims)

    def prepare(self, shape: tuple[int, ...], dtype: np.dtype) -> None:
        """Initialize the buffer, or validate/upgrade it on later calls.

        Parameters
        ----------
        shape : tuple of int
            Full shape of the current input (including the sample axis).
        dtype : numpy.dtype
            Result dtype required for the current call.

        Raises
        ------
        ValueError
            If the leading (non-sample) dimensions differ from those used
            to initialize the buffer.
        """
        leading_shape = self._leading_shape_of(shape)

        if self._buffer is None:
            self._leading_shape = leading_shape

            buffer_shape = list(leading_shape)
            insert_pos = (
                self.axis if self.axis >= 0 else len(buffer_shape) + 1 + self.axis
            )
            buffer_shape.insert(insert_pos, self.size)

            self._buffer = np.zeros(tuple(buffer_shape), dtype=dtype)
            return

        if leading_shape != self._leading_shape:
            raise ValueError(
                "Leading dimensions cannot change between calls: "
                f"expected {self._leading_shape}, got {leading_shape}"
            )

        if self._buffer.dtype != dtype:
            # Upgrade only; e.g. int input followed by float input later.
            self._buffer = self._buffer.astype(dtype, copy=False)

    def extend(self, x: Signal) -> Signal:
        """Concatenate retained history with new samples and update state.

        Parameters
        ----------
        x : numpy.ndarray
            New samples.

        Returns
        -------
        numpy.ndarray
            ``history`` concatenated with ``x`` along ``self.axis``.
        """
        assert self._buffer is not None

        combined = np.concatenate((self._buffer, x), axis=self.axis)

        moved = np.moveaxis(combined, self.axis, -1)
        if self.size > 0:
            trimmed = moved[..., -self.size :].copy()
        else:
            trimmed = moved[..., 0:0].copy()
        self._buffer = np.moveaxis(trimmed, -1, self.axis)

        return combined

    def reset(self) -> None:
        """Clear the buffer so the next call starts from zero history."""
        self._buffer = None
        self._leading_shape = None

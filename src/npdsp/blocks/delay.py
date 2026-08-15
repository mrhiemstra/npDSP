from __future__ import annotations

from ..core import Block, Signal, SlidingBuffer


class Delay(Block):
    """Delay a signal by a fixed number of samples.

    The block maintains an internal buffer containing samples from previous
    calls. Initially, the buffer is filled with zeros. Each call returns the
    input delayed by the configured number of samples.

    Parameters
    ----------
    samples : int
        Number of samples by which to delay the input signal. Must be
        non-negative.
    name : str, optional
        Optional name used to identify the block within a pipeline.

    Raises
    ------
    ValueError
        If ``samples`` is negative.

    Notes
    -----
    The delay is applied along the final axis, which represents samples.
    Leading dimensions are preserved, allowing the block to operate on signals
    containing multiple channels.

    The block is stateful. Call :meth:`reset` to clear the internal buffer.

    Examples
    --------
    >>> delay = Delay(2)
    >>> delay([1, 2, 3])
    array([0, 0, 1])
    >>> delay([4, 5, 6])
    array([2, 3, 4])

    A delay of zero samples returns the input unchanged::

        >>> delay = Delay(0)
        >>> delay([1, 2, 3])
        array([1, 2, 3])

    Multi-channel signals work naturally::

        >>> delay = Delay(1)
        >>> x = np.array([[1, 2, 3], [4, 5, 6]])  # 2 channels, 3 samples
        >>> delay(x)
        array([[0, 1, 2],
               [0, 4, 5]])
    """

    def __init__(self, samples: int, name: str | None = None):
        """Initialize a delay block.

        Parameters
        ----------
        samples : int
            Number of samples by which to delay the input signal. Must be
            non-negative.
        name : str, optional
            Optional name used to identify the block within a pipeline.

        Raises
        ------
        ValueError
            If ``samples`` is negative.
        """
        super().__init__(name=name)

        if samples < 0:
            raise ValueError("samples must be non-negative")

        self.samples = samples
        # axis=-1: Delay's sample axis is the final axis (consistent with FIR/IIR)
        self._history = SlidingBuffer(samples, axis=-1)

    @property
    def stateful(self) -> bool:
        """Whether the block maintains state between calls."""
        return True

    def process(self, x: Signal) -> Signal:
        """Process and delay an input signal.

        Parameters
        ----------
        x : Signal
            Input signal. The final axis represents samples; any leading
            dimensions are treated as channel or batch dimensions.

        Returns
        -------
        Signal
            Input signal delayed by ``self.samples`` samples. Samples for
            which no previous input exists are filled with zeros.
        """
        if self.samples == 0:
            return x

        n = x.shape[-1]

        self._history.prepare(x.shape, x.dtype)
        combined = self._history.extend(x)

        return combined[..., :n]

    def reset(self) -> None:
        """Clear the internal delay buffer.

        After resetting, the next input processed by the block behaves as if
        no previous samples had been processed.
        """
        self._history.reset()

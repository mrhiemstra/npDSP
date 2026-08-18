"""IO blocks."""

from npdsp.core import Block, Signal


class SampleRate(Block):
    """Pass a signal through unchanged. Sets an input sample rate for a pipeline. Does not resample.

    Parameters
    ----------
    sample_rate : float
        Sample rate in, for example, Hz
    name : str
        Name used to identify the tap within a pipeline.

    """

    def __init__(self, sample_rate: float, name: str | None = None) -> None:
        """Initialize a tap block.

        Parameters
        ----------
        sample_rate : float
            Sample rate in, for example, Hz
        name : str, optional
            Optional name used to identify the tap within a pipeline.

        """
        super().__init__(name)
        self._sample_rate = sample_rate

    @property
    def sample_rate(self) -> float:
        """Return the effective sample rate of the block."""
        return self._sample_rate

    def process(self, x: Signal) -> Signal:
        """Pass the input signal through unchanged.

        Parameters
        ----------
        x : Signal
            Input signal.

        Returns
        -------
        Signal
            The unchanged input signal.

        """
        return x

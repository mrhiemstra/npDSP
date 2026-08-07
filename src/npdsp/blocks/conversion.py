import numpy as np
import numpy.typing as npt

from ..core import Block, Signal


class Convert(Block):
    """Convert the data type of a signal.

    Parameters
    ----------
    dtype : numpy.typing.DTypeLike
        NumPy data type to convert the input signal to.
    name : str, optional
        Optional name used to identify the block within a pipeline.

    Notes
    -----
    Conversion uses :meth:`numpy.ndarray.astype` with ``copy=False``.
    NumPy may still create a copy when the requested data type differs
    from the input data type.

    Examples
    --------
    >>> block = Convert(np.float32)
    >>> block([1, 2, 3]).dtype
    dtype('float32')
    """

    def __init__(self, dtype: npt.DTypeLike, name: str | None = None):
        """Initialize a conversion block.

        Parameters
        ----------
        dtype : numpy.typing.DTypeLike
            NumPy data type to convert the signal to.
        name : str, optional
            Optional name used to identify the block within a pipeline.
        """
        super().__init__(name=name)
        self.dtype = np.dtype(dtype)

    def process(self, x: Signal) -> Signal:
        """Convert the input signal to the configured data type.

        Parameters
        ----------
        x : Signal
            Input signal.

        Returns
        -------
        Signal
            Signal converted to ``self.dtype``.
        """
        return x.astype(self.dtype, copy=False)


class Upsample(Block):
    """Upsample a signal with a specified factor.

    Parameters
    ----------
    factor : int
        Factor to upsample the input signal with.
    name : str, optional
        Optional name used to identify the block within a pipeline.

    Examples
    --------
    >>> block = Upsample(2)
    >>> block([1, 2, 3, 4])
    [1,0,2,0,3,0,4,0]
    """

    def __init__(self, factor: int, name: str | None = None):
        """Initialize a upsample block.

        Parameters
        ----------
        factor : int
            Factor to upsample the input signal with.
        name : str, optional
            Optional name used to identify the block within a pipeline.
        """
        super().__init__(name=name)
        self.factor = factor

    @property
    def sample_rate_ratio(self):
        return self.factor

    def process(self, x: Signal) -> Signal:
        """Upsample the input signal with the configured factor.

        Parameters
        ----------
        x : Signal
            Input signal.

        Returns
        -------
        Signal
            Signal upsampled with ``self.factor``.
        """
        y = np.zeros((*x.shape[:-1], x.shape[-1] * self.factor))
        y[..., :: self.factor] = x
        return y


class Downsample(Block):
    """Downsample a signal with a specified factor.

    Parameters
    ----------
    factor : int
        Factor to downsample the input signal with.
    name : str, optional
        Optional name used to identify the block within a pipeline.

    Examples
    --------
    >>> block = Downsample(2)
    >>> block([1, 2, 3, 4])
    [1,3]
    """

    def __init__(self, factor: int, name: str | None = None):
        """Initialize a downsample block.

        Parameters
        ----------
        factor : int
            Factor to downsample the input signal with.
        name : str, optional
            Optional name used to identify the block within a pipeline.
        """
        super().__init__(name=name)
        self.factor = factor
        self._offset = 0

    @property
    def stateful(self) -> bool:
        return True

    @property
    def sample_rate_ratio(self):
        return self.factor

    def reset(self) -> None:
        self._offset = 0

    def process(self, x: Signal) -> Signal:
        """Downsample the input signal with the configured factor.

        Parameters
        ----------
        x : Signal
            Input signal.

        Returns
        -------
        Signal
            Signal downsampled with ``self.factor``.
        """
        y = x[..., self._offset :: self.factor]

        # Find start index for next chunk (for streaming with undetermined chunk lengths)
        consumed = len(x) - self._offset
        self._offset = (-consumed) % self.factor

        return y

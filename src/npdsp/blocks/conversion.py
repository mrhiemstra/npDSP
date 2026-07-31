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
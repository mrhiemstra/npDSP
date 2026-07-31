from collections.abc import Callable

from ..core import Block, Signal


class Lambda(Block):
    """Apply a user-provided function to a signal.

    Parameters
    ----------
    func : Callable[[Signal], Signal]
        Function that receives the input signal and returns the processed
        signal.
    name : str, optional
        Optional name used to identify the block within a pipeline.
    """

    def __init__(self, func: Callable[[Signal], Signal], name: str | None = None) -> None:
        """Initialize a lambda block.

        Parameters
        ----------
        func : Callable[[Signal], Signal]
            Function to apply to each input signal.
        name : str, optional
            Optional name used to identify the block within a pipeline.
        """
        super().__init__(name)
        self.func = func

    def process(self, x: Signal) -> Signal:
        """Apply the configured function to the input signal.

        Parameters
        ----------
        x : Signal
            Input signal.

        Returns
        -------
        Signal
            Signal returned by ``self.func``.
        """
        return self.func(x)


class ResetCounter(Block):
    """Pass signals through while counting resets.

    This block does not modify its input signal. Each time :meth:`reset` is
    called, the ``reset_count`` attribute is incremented.

    Parameters
    ----------
    name : str, optional
        Optional name used to identify the block within a pipeline.

    Notes
    -----
    This block is primarily useful for testing and verifying reset behavior.
    """

    def __init__(self, name: str | None = None) -> None:
        """Initialize a reset counter.

        Parameters
        ----------
        name : str, optional
            Optional name used to identify the block within a pipeline.
        """
        super().__init__(name)
        self.reset_count = 0

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

    def reset(self) -> None:
        """Increment the reset counter."""
        self.reset_count += 1


class Tap(Block):
    """Pass a signal through unchanged.

    A tap provides a named, transparent point in a pipeline. It can be used
    to identify or inspect a position in a pipeline without modifying the
    signal.

    Parameters
    ----------
    name : str
        Name used to identify the tap within a pipeline.
    """

    def __init__(self, name: str) -> None:
        """Initialize a tap block.

        Parameters
        ----------
        name : str
            Name used to identify the tap within a pipeline.
        """
        super().__init__(name)

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


class Copy(Block):
    """Return a copy of the input signal.

    Unlike a transparent pass-through block, this block ensures that the
    returned signal does not share the same underlying data buffer as the
    input.

    Parameters
    ----------
    name : str
        Name used to identify the block within a pipeline.
    """

    def __init__(self, name: str) -> None:
        """Initialize a copy block.

        Parameters
        ----------
        name : str
            Name used to identify the block within a pipeline.
        """
        super().__init__(name)

    def process(self, x: Signal) -> Signal:
        """Return a copy of the input signal.

        Parameters
        ----------
        x : Signal
            Input signal.

        Returns
        -------
        Signal
            Copy of the input signal.
        """
        return x.copy()
import numpy as np

from ..core import Block, Signal, SignalLike


class Add(Block):
    """Add a value to the input signal.

    Parameters
    ----------
    value : SignalLike
        Value or array to add to the input signal.
    name : str, optional
        Optional name used to identify the block within a pipeline.
    """

    def __init__(self, value: SignalLike, name: str | None = None):
        """Initialize an addition block.

        Parameters
        ----------
        value : SignalLike
            Value or array to add to the input signal.
        name : str, optional
            Optional name used to identify the block within a pipeline.
        """
        super().__init__(name=name)
        self.value = value

    def process(self, x: Signal) -> Signal:
        """Add the configured value to the input signal.

        Parameters
        ----------
        x : Signal
            Input signal.

        Returns
        -------
        Signal
            Signal with ``self.value`` added element-wise using NumPy
            broadcasting rules.
        """
        return x + self.value


class Subtract(Block):
    """Subtract a value from the input signal.

    Parameters
    ----------
    value : SignalLike
        Value or array to subtract from the input signal.
    name : str, optional
        Optional name used to identify the block within a pipeline.
    """

    def __init__(self, value: SignalLike, name: str | None = None):
        """Initialize a subtraction block.

        Parameters
        ----------
        value : SignalLike
            Value or array to subtract from the input signal.
        name : str, optional
            Optional name used to identify the block within a pipeline.
        """
        super().__init__(name=name)
        self.value = value

    def process(self, x: Signal) -> Signal:
        """Subtract the configured value from the input signal.

        Parameters
        ----------
        x : Signal
            Input signal.

        Returns
        -------
        Signal
            Signal with ``self.value`` subtracted element-wise using NumPy
            broadcasting rules.
        """
        return x - self.value


class Multiply(Block):
    """Multiply the input signal by a value.

    Parameters
    ----------
    value : SignalLike
        Value or array to multiply the input signal by.
    name : str, optional
        Optional name used to identify the block within a pipeline.
    """

    def __init__(self, value: SignalLike, name: str | None = None):
        """Initialize a multiplication block.

        Parameters
        ----------
        value : SignalLike
            Value or array to multiply the input signal by.
        name : str, optional
            Optional name used to identify the block within a pipeline.
        """
        super().__init__(name=name)
        self.value = value

    def process(self, x: Signal) -> Signal:
        """Multiply the input signal by the configured value.

        Parameters
        ----------
        x : Signal
            Input signal.

        Returns
        -------
        Signal
            Signal multiplied element-wise by ``self.value`` using NumPy
            broadcasting rules.
        """
        return x * self.value


class Divide(Block):
    """Divide the input signal by a value.

    Parameters
    ----------
    value : SignalLike
        Value or array by which to divide the input signal. It must not
        contain zero.
    name : str, optional
        Optional name used to identify the block within a pipeline.

    Raises
    ------
    ZeroDivisionError
        If ``value`` contains zero.
    """

    def __init__(self, value: SignalLike, name: str | None = None):
        """Initialize a division block.

        Parameters
        ----------
        value : SignalLike
            Value or array by which to divide the input signal.
        name : str, optional
            Optional name used to identify the block within a pipeline.

        Raises
        ------
        ZeroDivisionError
            If ``value`` contains zero.
        """
        super().__init__(name=name)
        self.value = value

        if np.any(self.value == 0):
            raise ZeroDivisionError("Divide value cannot contain zero")

    def process(self, x: Signal) -> Signal:
        """Divide the input signal by the configured value.

        Parameters
        ----------
        x : Signal
            Input signal.

        Returns
        -------
        Signal
            Signal divided element-wise by ``self.value`` using NumPy
            broadcasting rules.
        """
        return x / self.value


class Floor(Block):
    """Perform floor division on the input signal.

    Parameters
    ----------
    value : SignalLike
        Value or array by which to floor-divide the input signal. It must not
        contain zero.
    name : str, optional
        Optional name used to identify the block within a pipeline.

    Raises
    ------
    ZeroDivisionError
        If ``value`` contains zero.
    """

    def __init__(self, value: SignalLike, name: str | None = None):
        """Initialize a floor-division block.

        Parameters
        ----------
        value : SignalLike
            Value or array by which to floor-divide the input signal.
        name : str, optional
            Optional name used to identify the block within a pipeline.

        Raises
        ------
        ZeroDivisionError
            If ``value`` contains zero.
        """
        super().__init__(name=name)
        self.value = value

        if np.any(self.value == 0):
            raise ZeroDivisionError("Floor value cannot contain zero")

    def process(self, x: Signal) -> Signal:
        """Perform floor division on the input signal.

        Parameters
        ----------
        x : Signal
            Input signal.

        Returns
        -------
        Signal
            Result of floor-dividing ``x`` by ``self.value`` using NumPy
            broadcasting rules.
        """
        return x // self.value


class Modulo(Block):
    """Calculate the remainder of division by a value.

    Parameters
    ----------
    value : SignalLike
        Value or array used as the divisor. It must not contain zero.
    name : str, optional
        Optional name used to identify the block within a pipeline.

    Raises
    ------
    ZeroDivisionError
        If ``value`` contains zero.
    """

    def __init__(self, value: SignalLike, name: str | None = None):
        """Initialize a modulo block.

        Parameters
        ----------
        value : SignalLike
            Value or array used as the divisor.
        name : str, optional
            Optional name used to identify the block within a pipeline.

        Raises
        ------
        ZeroDivisionError
            If ``value`` contains zero.
        """
        super().__init__(name=name)
        self.value = value

        if np.any(self.value == 0):
            raise ZeroDivisionError("Modulus value cannot contain zero")

    def process(self, x: Signal) -> Signal:
        """Calculate the element-wise remainder of division.

        Parameters
        ----------
        x : Signal
            Input signal.

        Returns
        -------
        Signal
            Remainder after dividing ``x`` by ``self.value`` using NumPy
            broadcasting rules.
        """
        return x % self.value


class Power(Block):
    """Raise the input signal to a power.

    Parameters
    ----------
    value : SignalLike
        Exponent or array of exponents.
    name : str, optional
        Optional name used to identify the block within a pipeline.
    """

    def __init__(self, value: SignalLike, name: str | None = None):
        """Initialize a power block.

        Parameters
        ----------
        value : SignalLike
            Exponent or array of exponents.
        name : str, optional
            Optional name used to identify the block within a pipeline.
        """
        super().__init__(name=name)
        self.value = value

    def process(self, x: Signal) -> Signal:
        """Raise the input signal to the configured power.

        Parameters
        ----------
        x : Signal
            Input signal.

        Returns
        -------
        Signal
            Input signal raised element-wise to ``self.value`` using NumPy
            broadcasting rules.
        """
        return x**self.value


class Absolute(Block):
    """Calculate the absolute value of a signal.

    Parameters
    ----------
    name : str, optional
        Optional name used to identify the block within a pipeline.
    """

    def __init__(self, name: str | None = None):
        """Initialize an absolute-value block.

        Parameters
        ----------
        name : str, optional
            Optional name used to identify the block within a pipeline.
        """
        super().__init__(name=name)

    def process(self, x: Signal) -> Signal:
        """Calculate the absolute value of the input signal.

        Parameters
        ----------
        x : Signal
            Input signal.

        Returns
        -------
        Signal
            Element-wise absolute value of the input signal.
        """
        return np.abs(x)


class Negate(Block):
    """Negate the input signal.

    Parameters
    ----------
    name : str, optional
        Optional name used to identify the block within a pipeline.
    """

    def __init__(self, name: str | None = None):
        """Initialize a negation block.

        Parameters
        ----------
        name : str, optional
            Optional name used to identify the block within a pipeline.
        """
        super().__init__(name=name)

    def process(self, x: Signal) -> Signal:
        """Negate the input signal element-wise.

        Parameters
        ----------
        x : Signal
            Input signal.

        Returns
        -------
        Signal
            Negated input signal.
        """
        return np.negative(x)


class Conjugate(Block):
    """Calculate the complex conjugate of a signal.

    Parameters
    ----------
    name : str, optional
        Optional name used to identify the block within a pipeline.
    """

    def __init__(self, name: str | None = None):
        """Initialize a conjugate block.

        Parameters
        ----------
        name : str, optional
            Optional name used to identify the block within a pipeline.
        """
        super().__init__(name=name)

    def process(self, x: Signal) -> Signal:
        """Calculate the complex conjugate of the input signal.

        Parameters
        ----------
        x : Signal
            Input signal.

        Returns
        -------
        Signal
            Element-wise complex conjugate of the input signal.
        """
        return np.conj(x)


class Clip(Block):
    """Clip signal values to a specified range.

    Parameters
    ----------
    bounds : SignalLike
        Values defining the clipping range. The minimum value is used as the
        lower bound and the maximum value is used as the upper bound.
    name : str, optional
        Optional name used to identify the block within a pipeline.
    """

    def __init__(self, bounds: SignalLike, name: str | None = None):
        """Initialize a clipping block.

        Parameters
        ----------
        bounds : SignalLike
            Values defining the clipping range. The minimum value is used as
            the lower bound and the maximum value is used as the upper bound.
        name : str, optional
            Optional name used to identify the block within a pipeline.
        """
        bounds = np.asarray(bounds)
        self.lower_bound = np.min(bounds)
        self.upper_bound = np.max(bounds)
        super().__init__(name=name)

    def process(self, x: Signal) -> Signal:
        """Clip the input signal to the configured bounds.

        Parameters
        ----------
        x : Signal
            Input signal.

        Returns
        -------
        Signal
            Signal with values restricted to the configured lower and upper
            bounds.
        """
        return np.clip(x, a_min=self.lower_bound, a_max=self.upper_bound)


class Minimum(Block):
    """Calculate the minimum value of a signal.

    Parameters
    ----------
    name : str, optional
        Optional name used to identify the block within a pipeline.
    """

    def __init__(self, name: str | None = None):
        """Initialize a minimum-value block.

        Parameters
        ----------
        name : str, optional
            Optional name used to identify the block within a pipeline.
        """
        super().__init__(name=name)

    def process(self, x: Signal) -> Signal:
        """Calculate the minimum value of the input signal.

        Parameters
        ----------
        x : Signal
            Input signal.

        Returns
        -------
        Signal
            Minimum value of the input signal.
        """
        return np.min(x)


class Maximum(Block):
    """Calculate the maximum value of a signal.

    Parameters
    ----------
    name : str, optional
        Optional name used to identify the block within a pipeline.
    """

    def __init__(self, name: str | None = None):
        """Initialize a maximum-value block.

        Parameters
        ----------
        name : str, optional
        Optional name used to identify the block within a pipeline.
        """
        super().__init__(name=name)

    def process(self, x: Signal) -> Signal:
        """Calculate the maximum value of the input signal.

        Parameters
        ----------
        x : Signal
            Input signal.

        Returns
        -------
        Signal
            Maximum value of the input signal.
        """
        return np.max(x)

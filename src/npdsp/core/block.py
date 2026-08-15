from __future__ import annotations

from abc import ABC, abstractmethod
from time import perf_counter
from typing import TYPE_CHECKING

import numpy as np

from .profile import ProfileResult, ProfileResults
from .typing import Signal, SignalLike

if TYPE_CHECKING:
    from .pipeline import Pipeline


class Block(ABC):
    """Base class for all npDSP processing blocks.

    A block transforms an input signal into an output signal. Blocks can be
    composed into :class:`Pipeline` objects using the ``>>`` operator.

    Parameters
    ----------
    name : str, optional
        Optional name used to identify the block within a pipeline.

    Notes
    -----
    Subclasses must implement :meth:`process`. Blocks are stateless by
    default, but subclasses can override :attr:`stateful` and :meth:`reset`
    when they maintain state between calls.
    """

    @abstractmethod
    def process(self, x: Signal) -> Signal:
        """Process an input signal.

        Parameters
        ----------
        x : Signal
            Input signal to process.

        Returns
        -------
        Signal
            Processed output signal.

        Raises
        ------
        NotImplementedError
            Raised if a subclass does not implement this method.
        """
        raise NotImplementedError

    @property
    def stateful(self) -> bool:
        """Whether the block maintains state between calls.

        Returns
        -------
        bool
            ``False`` by default. Stateful blocks should override this
            property and return ``True``.
        """
        return False

    @property
    def sample_rate_ratio(self) -> float:
        return 1

    @property
    def latency_samples(self) -> float | None:
        return 0

    @property
    def sample_rate(self) -> float | None:
        return self._sample_rate

    def reset(self) -> None:
        """Reset the internal state of the block.

        Stateless blocks do not need to perform any action when reset.
        Stateful subclasses should override this method to restore their
        initial state.
        """

    def profile(self, x: Signal, runs: int = 1, reset: bool = False) -> ProfileResults:
        """Profile the execution time of the block.

        The block is executed one or more times and the minimum, mean, and
        maximum execution times are recorded.

        Parameters
        ----------
        x : Signal
            Input signal to process during profiling.
        runs : int, default=1
            Number of times to execute the block.
        reset : bool, default=False
            If ``True``, reset the block before and after profiling.

        Returns
        -------
        ProfileResults
            Profiling results containing the minimum, mean, and maximum
            execution times.

        Notes
        -----
        The output of each run is passed as the input to the next run.
        Therefore, for stateful blocks, successive runs may operate on
        different inputs or internal states.
        """
        results = ProfileResults()

        if reset:
            self.reset()

        run_times: list[float] = []

        for _ in range(runs):
            start = perf_counter()

            x = self(x)

            elapsed = perf_counter() - start
            run_times.append(elapsed)

        if reset:
            self.reset()

        min_time = float(np.min(run_times))
        mean_time = float(np.mean(run_times))
        max_time = float(np.max(run_times))

        results.append(
            ProfileResult(
                name=self.name or self.__class__.__name__,
                min_time=min_time,
                mean_time=mean_time,
                max_time=max_time,
                runs=runs,
            )
        )

        return results

    def __init__(self, name: str | None = None) -> None:
        """Initialize a block.

        Parameters
        ----------
        name : str, optional
            Optional name used to identify the block within a pipeline.
        """
        self.name = name
        self._sample_rate: float | None = None

    def __call__(self, x: SignalLike) -> Signal:
        """Process a signal by calling the block.

        Parameters
        ----------
        x : SignalLike
            Input signal. Array-like values are converted to a NumPy array
            before being passed to :meth:`process`.

        Returns
        -------
        Signal
            Processed output signal.
        """
        return self.process(np.asarray(x))

    def __rshift__(self, other: Block | Pipeline) -> Pipeline:
        """Compose this block with another block or pipeline.

        Parameters
        ----------
        other : Block or Pipeline
            Block or pipeline to execute after this block.

        Returns
        -------
        Pipeline
            Pipeline containing this block followed by ``other``.
        """
        from .pipeline import Pipeline

        return Pipeline(self, other)

    def __repr__(self) -> str:
        """Return a representation of the block.

        Returns
        -------
        str
            Class name followed by the block's non-``None`` instance
            attributes.
        """
        args = ", ".join(
            f"{k}={v!r}"
            for k, v in self.__dict__.items()
            if v is not None and not k.startswith("_")
        )
        return f"{self.__class__.__name__}({args})"

    __str__ = __repr__

    def __len__(self) -> int:
        """Return the number of processing stages represented by the block.

        Returns
        -------
        int
            Always returns ``1`` for a single block.
        """
        return 1

"""Pipeline of DSP blocks."""

from __future__ import annotations

from time import perf_counter
from typing import TYPE_CHECKING

import numpy as np
from typing_extensions import Self

from .block import Block
from .profile import ProfileResult, ProfileResults

if TYPE_CHECKING:
    from collections.abc import Iterator
    from types import EllipsisType

    from .typing import Signal


class Pipeline(Block):
    """A sequence of DSP blocks executed in order.

    A pipeline applies each block to the output of the preceding block.
    Pipelines can be constructed directly or composed using the ``>>``
    operator.

    Parameters
    ----------
    *blocks : Block or Pipeline
        Blocks to include in the pipeline. Nested pipelines are flattened
        when the pipeline is constructed.

    Examples
    --------
    >>> pipeline = Pipeline(Add(1), Multiply(2))
    >>> pipeline([1, 2, 3])
    array([4, 6, 8])

    Notes
    -----
    Named blocks can be accessed using their names, and blocks can be
    selected using integer indices or slices.

    Examples
    --------
    A pipeline can be composed using the right-shift operator::

        pipeline = Add(1) >> Multiply(2)

    Blocks can also be selected by name::

        pipeline["my_block"]

    A slice can be made inclusive by appending an ellipsis::

        pipeline["first":"last", ...]

    """

    def __init__(
        self, *blocks: Block | Pipeline, name: str | None = None
    ) -> None:
        """Initialize a pipeline.

        Parameters
        ----------
        *blocks : Block or Pipeline
            Blocks to include in the pipeline. Nested pipelines are flattened
            into the new pipeline.
        name : str, optional
            Optional name used to identify the pipeline in a graph.

        """
        super().__init__(name=name)
        self.blocks: list[Block] = []
        self._names: dict[str, int] = {}

        for block in blocks:
            if isinstance(block, Pipeline):
                self.blocks.extend(block.blocks)
            else:
                self.blocks.append(block)

        if self.blocks:
            self._reindex()

    @property
    def first(self) -> Block:
        """Return the first block in the pipeline.

        Returns
        -------
        Block
            First block in the pipeline.

        Raises
        ------
        IndexError
            Raised if the pipeline contains no blocks.

        """
        return self.blocks[0]

    @property
    def last(self) -> Block:
        """Return the last block in the pipeline.

        Returns
        -------
        Block
            Last block in the pipeline.

        Raises
        ------
        IndexError
            Raised if the pipeline contains no blocks.

        """
        return self.blocks[-1]

    @property
    def sample_rate(self) -> float:
        """Return the effective sample rate of the pipeline."""
        rate = 1

        for block in self.blocks:
            rate *= block.sample_rate_ratio

        return rate

    @property
    def latency_samples(self) -> float:
        """Return the total latency of the pipeline in samples."""
        latency: float = 0

        for block in self.blocks:
            latency += (
                block.latency_samples
                if block.latency_samples is not None
                else 0
            )

        return latency

    @property
    def has_frequency_dependent_latency(self) -> bool:
        """Return whether any block in the pipeline has frequency-dependent latency."""
        return any(block.latency_samples is None for block in self.blocks)

    @property
    def latency(self) -> float:
        """Return the total latency of the pipeline in seconds."""
        if self.has_frequency_dependent_latency:
            raise NotImplementedError(
                "Frequency dependent latency reporting is not yet implemented"
            )

        # assert self.first.sample_rate is not None
        # assert self.latency_samples is not None

        return self.latency_samples / self.first.sample_rate

    def _reindex(self) -> None:
        """Rebuild the mapping between block names and their indices, and the sample rate calculations.

        Unnamed blocks are ignored. Block names must be unique within the pipeline.

        Raises
        ------
        ValueError
            Raised if multiple blocks have the same name.

        """
        self._names.clear()

        sample_rate = self.first.sample_rate

        for idx, block in enumerate(self.blocks):
            if block.name is None:
                continue
            if block.name in self._names:
                raise ValueError(f"Duplicate block name: {block.name!r}")

            self._names[block.name] = idx

            if idx > 0:
                block._sample_rate = sample_rate * block.sample_rate_ratio  # pyright: ignore[reportPrivateUsage] # noqa: SLF001

    def _resolve_block_index(self, key: str | int) -> int:
        """Resolve a block name or index to an integer index.

        Parameters
        ----------
        key : str or int
            Block name or positional index.

        Returns
        -------
        int
            Resolved block index.

        Raises
        ------
        KeyError
            Raised when ``key`` is a string that does not match a named block.

        """
        if isinstance(key, str):
            try:
                return self._names[key]
            except KeyError:
                raise KeyError(f"No block named {key!r}") from None
        else:  # int
            return key

    def _resolve_slice_index(self, key: str | int | None) -> int | None:
        """Resolve a slice boundary to an integer index.

        Parameters
        ----------
        key : str, int, or None
            Block name, positional index, or ``None`` for an open-ended
            boundary.

        Returns
        -------
        int or None
            Resolved index, or ``None`` when the boundary is open-ended.

        """
        if key is None:
            return None

        return self._resolve_block_index(key)

    def process(self, x: Signal) -> Signal:
        """Process a signal through every block in sequence.

        Parameters
        ----------
        x : Signal
            Input signal.

        Returns
        -------
        Signal
            Signal produced by the final block in the pipeline.

        """
        for block in self.blocks:
            x = block(x)

        return x

    def reset(self) -> None:
        """Reset every block in the pipeline.

        Each block's :meth:`Block.reset` method is called in pipeline order.
        Stateful blocks are reported to standard output when they are reset.
        """
        for block in self.blocks:
            block.reset()
            if block.stateful:
                pass

    def profile(
        self, x: Signal, runs: int = 1, reset: bool = False
    ) -> ProfileResults:
        """Profile the execution time of each block in the pipeline.

        Each block is executed ``runs`` times and its minimum, mean, and
        maximum execution times are recorded separately.

        Parameters
        ----------
        x : Signal
            Input signal to process.
        runs : int, default=1
            Number of times to execute each block during profiling.
        reset : bool, default=False
            If ``True``, reset the pipeline before and after profiling.

        Returns
        -------
        ProfileResults
            Profiling results containing timing information for each block.

        Notes
        -----
        The output of each block becomes the input to the next block.
        Repeated profiling runs of an individual block use the same input
        produced by the preceding block.

        """
        results = ProfileResults()

        if reset:
            self.reset()

        for block in self.blocks:
            run_times: list[float] = []

            _x = x

            for _ in range(runs):
                start = perf_counter()

                _x = block(x)

                elapsed = perf_counter() - start
                run_times.append(elapsed)

            x = _x

            min_time = np.min(run_times).astype(float)
            mean_time = np.mean(run_times).astype(float)
            max_time = np.max(run_times).astype(float)

            results.append(
                ProfileResult(
                    name=block.name or block.__class__.__name__,
                    min_time=min_time,
                    mean_time=mean_time,
                    max_time=max_time,
                    runs=runs,
                )
            )

        if reset:
            self.reset()

        return results

    def state(self) -> None:
        """Display or inspect the state of the pipeline.

        This method is currently a placeholder and does not perform any
        operation.
        """

    def find(self, cls: type[Block]) -> Block | None:
        """Find the first block that is an instance of a given class.

        Parameters
        ----------
        cls : type[Block]
            Block class to search for.

        Returns
        -------
        Block or None
            First matching block, or ``None`` if no matching block exists.

        """
        for block in self.blocks:
            if isinstance(block, cls):
                return block

        return None

    def find_all(self, cls: type[Block]) -> list[Block]:
        """Find all blocks that are instances of a given class.

        Parameters
        ----------
        cls : type[Block]
            Block class to search for.

        Returns
        -------
        list of Block
            All blocks matching the requested class, in pipeline order.

        """
        return [block for block in self.blocks if isinstance(block, cls)]

    def insert(self, key: int | str, block: Block) -> None:
        """Insert a block before the block at the specified position.

        Parameters
        ----------
        key : int or str
            Index or name of the block before which ``block`` will be
            inserted.
        block : Block
            Block to insert.

        """
        self.blocks.insert(self._resolve_block_index(key), block)
        self._reindex()

    def remove(self, key: int | str) -> None:
        """Remove a block from the pipeline.

        Parameters
        ----------
        key : int or str
            Index or name of the block to remove.

        """
        del self.blocks[self._resolve_block_index(key)]
        self._reindex()

    def replace(self, key: int | str, block: Block) -> None:
        """Replace a block in the pipeline.

        Parameters
        ----------
        key : int or str
            Index or name of the block to replace.
        block : Block
            Replacement block.

        """
        self.blocks[self._resolve_block_index(key)] = block
        self._reindex()

    def __setitem__(self, key: int | str, block: Block) -> None:
        """Replace a block using item assignment.

        Parameters
        ----------
        key : int or str
            Index or name of the block to replace.
        block : Block
            Replacement block.

        """
        self.replace(key, block)

    def __delitem__(self, key: int | str) -> None:
        """Remove a block using item deletion.

        Parameters
        ----------
        key : int or str
            Index or name of the block to remove.

        """
        self.remove(key)

    def __contains__(self, item: Block | str) -> bool:
        """Check whether a block or block name exists in the pipeline.

        Parameters
        ----------
        item : Block or str
            Block instance or block name to search for.

        Returns
        -------
        bool
            ``True`` if the block or name exists in the pipeline,
            otherwise ``False``.

        """
        if isinstance(item, str):
            return item in self._names
        # Block
        return item in self.blocks

    def __rshift__(self, other: Pipeline | Block) -> Pipeline:
        """Append a block or pipeline and return a new pipeline.

        Parameters
        ----------
        other : Pipeline or Block
            Block or pipeline to append.

        Returns
        -------
        Pipeline
            New pipeline containing the blocks from both operands.

        """
        if isinstance(other, Pipeline):
            return Pipeline(*self.blocks, *other.blocks)
        return Pipeline(*self.blocks, other)

    def __irshift__(self, other: Pipeline | Block) -> Self:
        """Append a block or pipeline to this pipeline in place.

        Parameters
        ----------
        other : Pipeline or Block
            Block or pipeline to append.

        Returns
        -------
        Self
            This pipeline after the blocks have been appended.

        """
        if isinstance(other, Pipeline):
            self.blocks.extend(other.blocks)
        else:  # Block
            self.blocks.append(other)

        self._reindex()
        return self

    def __getitem__(
        self, key: str | int | slice | tuple[str | int | slice, EllipsisType]
    ) -> Pipeline | Block:
        """Retrieve a block or sub-pipeline.

        Parameters
        ----------
        key : str, int, slice, or tuple
            Selection key.

            A string selects a named block.

            An integer selects a block by positional index.

            A slice selects a sub-pipeline. Slice boundaries may be block
            names or integer indices.

            A slice can include its stop boundary by appending an ellipsis,
            for example ``pipeline["first":"last", ...]``.

        Returns
        -------
        Pipeline or Block
            A single block for string or integer indexing, or a new pipeline
            for a slice.

        Raises
        ------
        TypeError
            Raised when an ellipsis tuple does not contain exactly a slice
            key followed by ``...``.
        NotImplementedError
            Raised when a slice step is provided.

        """
        inclusive_stop = False
        if isinstance(key, tuple):
            if not (len(key) == 2 and key[1] is Ellipsis):
                raise TypeError(
                    f"Expected key to be (str | int | slice, Ellipsis)), but {key=}"
                )
            key, _ = key
            inclusive_stop = True

        if isinstance(key, str):
            return self.blocks[self._names[key]]

        if isinstance(key, int):
            return self.blocks[key]

        # Slice
        if key.step is not None:
            raise NotImplementedError("Slice steps are not implemented")

        start = self._resolve_slice_index(key.start)
        stop = self._resolve_slice_index(key.stop)

        if stop is not None and inclusive_stop:
            stop += 1

        return Pipeline(*self.blocks[start:stop])

    def __repr__(self) -> str:
        """Return a string representation of the pipeline.

        Returns
        -------
        str
            Pipeline blocks joined using the ``>>`` operator notation.

        """
        return " >> ".join(map(str, self.blocks))

    __str__ = __repr__

    def __len__(self) -> int:
        """Return the number of blocks in the pipeline.

        Returns
        -------
        int
            Number of blocks contained in the pipeline.

        """
        return len(self.blocks)

    def __iter__(self) -> Iterator[Block]:
        """Iterate over the blocks in the pipeline.

        Returns
        -------
        Iterator[Block]
            Iterator yielding blocks in pipeline order.

        """
        return iter(self.blocks)

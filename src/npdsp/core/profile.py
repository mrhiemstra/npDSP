from dataclasses import dataclass

from si_prefix import si_format  # type: ignore[reportMissingTypeStubs]


@dataclass
class ProfileResult:
    """Timing results for a single profiled block.

    Parameters
    ----------
    name : str
        Name of the profiled block.
    min_time : float
        Minimum execution time measured across all profiling runs, in
        seconds.
    mean_time : float
        Mean execution time measured across all profiling runs, in seconds.
    max_time : float
        Maximum execution time measured across all profiling runs, in
        seconds.
    """

    name: str
    min_time: float  # Seconds
    mean_time: float  # Seconds
    max_time: float  # Seconds


class ProfileResults(list[ProfileResult]):
    """Collection of profiling results for one or more blocks.

    This class extends :class:`list` and contains one
    :class:`ProfileResult` for each profiled block.

    Examples
    --------
    Profile results can be iterated over like a normal list::

        for result in results:
            print(result.name, result.mean_time)

    The total mean execution time can be accessed through :attr:`tottime`.
    """

    @property
    def tottime(self) -> float:
        """Return the total mean execution time.

        The total is calculated by summing the mean execution time of every
        profile result.

        Returns
        -------
        float
            Total mean execution time in seconds.
        """
        return sum((result.mean_time for result in self), 0.0)

    def __str__(self) -> str:
        """Return a human-readable representation of the profiling results.

        Each block is displayed on a separate line with its mean execution
        time and percentage of the total mean execution time. The final line
        contains the total execution time.

        Returns
        -------
        str
            Formatted profiling results with SI-prefixed execution times.
        """
        return (
            "\n".join(
                f"{result.name} {si_format(result.mean_time, precision=3)}s {100 * result.mean_time / self.tottime:.2f}%"
                for result in self
            )
            + f"\n{si_format(self.tottime, precision=3)}s"
        )

from dataclasses import dataclass


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
    runs : int
        Number of profiling runs performed.
    """

    name: str
    min_time: float  # Seconds
    mean_time: float  # Seconds
    max_time: float  # Seconds
    runs: int  # Number of profiling runs

    def __repr__(self) -> str:
        return f"{self.name}: {_time_to_si_string(self.mean_time)}"

    __str__ = __repr__


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

    def __repr__(self) -> str:
        """Return a human-readable representation of the profiling results.

        Each block is displayed on a separate line with its mean execution
        time and percentage of the total mean execution time. The final line
        contains the total execution time.

        Returns
        -------
        str
            Formatted profiling results with SI-prefixed execution times.
        """
        max_name_length = max((len(result.name) for result in self), default=0)
        max_min_time_length = max(
            (len(_time_to_si_string(result.mean_time)) for result in self), default=0
        )
        max_mean_time_length = max(
            (len(_time_to_si_string(result.mean_time)) for result in self), default=0
        )
        max_max_time_length = max(
            (len(_time_to_si_string(result.max_time)) for result in self), default=0
        )
        max_percent_length = max(
            max(
                (
                    len(f"{100 * result.mean_time / self.tottime:.2f}%")
                    for result in self
                ),
                default=0,
            ),
            len("percent"),
        )

        lines = [
            f"{'name':<{max_name_length}}|{'min_time':>{max_min_time_length}}|{'mean_time':>{max_mean_time_length}}|{'max_time':>{max_max_time_length}}|{'percent':>{max_percent_length}}",
            "-" * max_name_length
            + "+"
            + "-" * max_min_time_length
            + "+"
            + "-" * max_mean_time_length
            + "+"
            + "-" * max_max_time_length
            + "+"
            + "-" * max_percent_length,
            *[
                f"{result.name:<{max_name_length}}|{_time_to_si_string(result.min_time):>{max_min_time_length}}|{_time_to_si_string(result.mean_time):>{max_mean_time_length}}|{_time_to_si_string(result.max_time):>{max_max_time_length}}|{f'{(100 * result.mean_time / self.tottime):.2f}%':>{max_percent_length}}"
                for result in self
            ],
        ]

        return "\n".join(lines)
        # return (
        #     "\n".join(
        #         f"{result.name} {si_format(result.mean_time, precision=3)}s {100 * result.mean_time / self.tottime:.2f}%"
        #         for result in self
        #     )
        #     + f"\n{si_format(self.tottime, precision=3)}s"
        # )

    __str__ = __repr__


def _time_to_si_string(time: float) -> str:
    """Convert a time in seconds to a human-readable string with SI prefix.

    Parameters
    ----------
    time : float
        Time in seconds.

    Returns
    -------
    str
        Time formatted as a string with SI prefix.
    """
    match time:
        case time if time < 1e-9:
            return f"{time * 1e12:.3f} ps"
        case time if time < 1e-6:
            return f"{time * 1e9:.3f} ns"
        case time if time < 1e-3:
            return f"{time * 1e6:.3f} µs"
        case time if time < 1:
            return f"{time * 1e3:.3f} ms"
        case _:
            return f"{time:.3f} s"

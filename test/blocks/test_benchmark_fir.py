from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pytest

from npdsp import FIR

if TYPE_CHECKING:
    from pytest_benchmark.fixture import BenchmarkFixture

RNG = np.random.default_rng(0)


@pytest.mark.benchmark(max_time=0.2, warmup=True, group="fir-taps")
@pytest.mark.parametrize(
    "taps", [3, 16, 64, 256, 1024, 4096], ids=lambda t: f"taps={t}"
)
def test_benchmark_fir_tap_scaling(
    benchmark: BenchmarkFixture, taps: int
) -> None:
    """FIR throughput as filter length grows, fixed signal length."""
    coefs = RNG.normal(size=taps)
    fir = FIR(coefs, name=f"fir_taps_{taps}")
    x = RNG.normal(size=8192)
    benchmark(fir, x)


@pytest.mark.benchmark(max_time=0.2, warmup=True, group="fir-channels")
@pytest.mark.parametrize(
    "channels", [1, 2, 8, 32, 128, 512], ids=lambda c: f"ch={c}"
)
def test_benchmark_fir_wide_channels(
    benchmark: BenchmarkFixture, channels: int
) -> None:
    """FIR with many independent channels processed in one call."""
    fir = FIR([1, 2, 3, 4, 5], name=f"fir_wide_{channels}")
    x = RNG.normal(size=(channels, 4096))
    benchmark(fir, x)


@pytest.mark.benchmark(max_time=0.2, warmup=True, group="fir-taps")
def test_benchmark_fir_single_tap_gain(benchmark: BenchmarkFixture) -> None:
    """Taps == 1 takes the no-history elementwise-gain shortcut path."""
    fir = FIR([2.5], name="fir_single_tap")
    x = RNG.normal(size=(256, 8192))
    benchmark(fir, x)


@pytest.mark.benchmark(max_time=0.2, warmup=True, group="fir-streaming")
def test_benchmark_fir_streaming_tiny_chunks(
    benchmark: BenchmarkFixture,
) -> None:
    """Many 1-sample calls (worst case for per-call overhead) vs. throughput.

    Simulates a real-time loop feeding one sample at a time, 4096 times,
    through a single persistent FIR instance.
    """
    fir = FIR(RNG.normal(size=32), name="fir_stream")
    x = RNG.normal(size=4096)

    def run() -> None:
        for i in range(x.shape[0]):
            fir(x[i : i + 1])

    benchmark(run)


@pytest.mark.benchmark(max_time=0.2, warmup=True, group="fir-cold")
def test_benchmark_fir_cold_first_call(benchmark: BenchmarkFixture) -> None:
    """Cost of the first call (dtype/broadcast cache priming) in isolation.

    Uses pedantic with a fresh FIR per round so each timed call is a true
    "cold start," separate from steady-state throughput measured above.
    """

    def setup() -> tuple[tuple[FIR], dict[str, Any]]:
        return ((FIR(RNG.normal(size=64), name="fir_cold"),), {})

    def run(fir: FIR) -> None:
        fir(RNG.normal(size=256))

    cast("Any", benchmark).pedantic(run, setup=setup, rounds=50, iterations=1)

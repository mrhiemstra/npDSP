from __future__ import annotations

import numpy as np
import pytest
from pytest_benchmark.fixture import BenchmarkFixture

from npdsp import IIR

RNG = np.random.default_rng(0)


@pytest.mark.benchmark(group="iir-steady-state")
@pytest.mark.parametrize("channels", [1, 2, 8, 64], ids=lambda c: f"ch={c}")
def test_benchmark_iir_steady_state_channels(
    benchmark: BenchmarkFixture, channels: int
) -> None:
    """Steady-state throughput across channel counts."""
    b, a = [0.2, 0.4, 0.2], [1.0, -0.5, 0.1]
    iir = IIR(b, a)
    x = (
        RNG.normal(size=4096).astype(np.float64)
        if channels == 1
        else RNG.normal(size=(channels, 4096)).astype(np.float64)
    )

    # Warm up
    iir(x)

    def run() -> None:
        iir(x)

    benchmark(run)


@pytest.mark.benchmark(group="iir-steady-state")
@pytest.mark.parametrize("chunk_size", [1, 16, 256, 4096], ids=lambda s: f"chunk={s}")
def test_benchmark_iir_steady_state_chunk_size(
    benchmark: BenchmarkFixture, chunk_size: int
) -> None:
    """Verify throughput doesn't degrade with small chunks."""
    b, a = [0.2, 0.4, 0.2], [1.0, -0.5, 0.1]
    iir = IIR(b, a)
    x = RNG.normal(size=(8, chunk_size)).astype(np.float64)

    iir(x)  # Warm up

    def run() -> None:
        iir(x)

    benchmark(run)


@pytest.mark.benchmark(group="iir-steady-state")
@pytest.mark.parametrize("order", [1, 2, 4, 8], ids=lambda o: f"order={o}")
def test_benchmark_iir_steady_state_order(
    benchmark: BenchmarkFixture, order: int
) -> None:
    """Steady-state scales with filter order."""
    a = np.concatenate([[1.0], RNG.normal(scale=0.05, size=order)]).astype(np.float64)
    b = RNG.normal(size=order + 1).astype(np.float64)
    iir = IIR(b, a)
    x = RNG.normal(size=(8, 4096)).astype(np.float64)

    iir(x)  # Warm up

    def run() -> None:
        iir(x)

    benchmark(run)

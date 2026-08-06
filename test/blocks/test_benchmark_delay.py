from __future__ import annotations

import numpy as np
import pytest
from pytest_benchmark.fixture import BenchmarkFixture

from npdsp import Delay

RNG = np.random.default_rng(0)


@pytest.mark.benchmark(max_time=0.2, warmup=True, group="delay")
@pytest.mark.parametrize(
    "samples", [1, 64, 1024, 16_384, 262_144], ids=lambda s: f"delay={s}"
)
def test_benchmark_delay_length_scaling(
    benchmark: BenchmarkFixture, samples: int
) -> None:
    """Delay throughput as buffer length grows, fixed input length.

    Every call still does one concatenate + one trailing-slice copy sized
    by `samples`, so this should scale with delay length, not just input
    length. To confirm that relationship is linear, not worse.
    """
    delay = Delay(samples, name=f"delay_{samples}")
    x = RNG.normal(size=4096)
    benchmark(delay, x)


@pytest.mark.benchmark(max_time=0.2, warmup=True, group="delay-streaming")
def test_benchmark_delay_streaming_tiny_chunks(benchmark: BenchmarkFixture) -> None:
    """Many 1-sample calls through a moderately long delay line.

    Stresses the concatenate-per-call allocation pattern directly: each
    call reallocates a (samples+1)-length array just to shift by one
    sample, which is the known remaining inefficiency in SlidingBuffer.
    """
    delay = Delay(512, name="delay_stream")
    x = RNG.normal(size=4096)

    def run() -> None:
        for i in range(x.shape[0]):
            delay(x[i : i + 1])

    benchmark(run)


@pytest.mark.benchmark(max_time=0.2, warmup=True, group="delay-wide")
@pytest.mark.parametrize("channels", [1, 16, 256], ids=lambda c: f"ch={c}")
def test_benchmark_delay_wide_channels(
    benchmark: BenchmarkFixture, channels: int
) -> None:
    """Delay with many channels along the (non-sample) trailing axes."""
    delay = Delay(128, name=f"delay_wide_{channels}")
    x = RNG.normal(size=(4096, channels))  # Delay's sample axis is axis 0
    benchmark(delay, x)

import pytest

from npdsp import *


def test_sample_rate(
    pipeline: Pipeline,
) -> None:
    assert pipeline.sample_rate == 1


def test_sample_rate2() -> None:
    pipeline = Add(0) >> Upsample(4)
    assert pipeline.sample_rate == 4


def test_sample_rate3() -> None:
    pipeline = Add(0) >> Downsample(4)
    assert pipeline.sample_rate == 1 / 4


def test_sample_rate4() -> None:
    pipeline = Add(0) >> Upsample(4) >> Downsample(4)
    assert pipeline.sample_rate == 1


def test_sample_latency() -> None:
    pipeline = Add(0) >> Downsample(4)
    assert pipeline.latency_samples == 0


def test_sample_latency2() -> None:
    pipeline = Add(0) >> FIR([1, 2, 3])
    assert pipeline.latency_samples == 1


def test_has_frequency_dependent_latency() -> None:
    pipeline = SampleRate(5) >> Add(0) >> FIR([1, 2, 3])
    assert not pipeline.has_frequency_dependent_latency
    assert pipeline.latency == 1 / 5


def test_has_frequency_dependent_latency2() -> None:
    pipeline = SampleRate(5) >> IIR([1, 2, 3], [1, 2, 3])

    assert pipeline.has_frequency_dependent_latency
    with pytest.raises(
        NotImplementedError,
        match="Frequency dependent latency reporting is not yet implemented",
    ):
        pipeline.latency  # noqa: B018


def test_pipeline_reset() -> None:
    pipeline = Add(0) >> Downsample(4)
    pipeline.reset()

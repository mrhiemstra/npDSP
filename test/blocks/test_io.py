import numpy as np

from npdsp import *


def test_samplerate_is_feed_thru():
    sr = SampleRate(44100)
    x = np.arange(0, 10, 100)
    y = sr(x)
    assert np.array_equal(x, y)

"""npdsp: A Python library for digital signal processing."""

from importlib.metadata import version

from .blocks import *
from .core import *

__version__ = version("npdsp")

__all__ = [
    "FIR",
    "IIR",
    "Absolute",
    "Add",
    "Block",
    "Clip",
    "Conjugate",
    "Convert",
    "Copy",
    "Delay",
    "Divide",
    "Downsample",
    "Floor",
    "Lambda",
    "Maximum",
    "Minimum",
    "Modulo",
    "Multiply",
    "Negate",
    "Pipeline",
    "Power",
    "ProfileResult",
    "ProfileResults",
    "ResetCounter",
    "SampleRate",
    "Signal",
    "SignalLike",
    "Subtract",
    "Tap",
    "Upsample",
    "design",
    "impulse_response",
    "window",
]

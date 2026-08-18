"""Blocks for npdsp."""

from .conversion import Convert, Downsample, Upsample
from .filters import design, impulse_response, window
from .fir import FIR
from .iir import IIR
from .io import SampleRate
from .math import (
    Absolute,
    Add,
    Clip,
    Conjugate,
    Divide,
    Floor,
    Maximum,
    Minimum,
    Modulo,
    Multiply,
    Negate,
    Power,
    Subtract,
)
from .timing import Delay
from .utility import Copy, Lambda, ResetCounter, Tap

__all__ = [
    "FIR",
    "IIR",
    "Absolute",
    "Add",
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
    "SampleRate",
    "Multiply",
    "Negate",
    "Power",
    "ResetCounter",
    "Subtract",
    "Tap",
    "Upsample",
    "design",
    "impulse_response",
    "window",
]

"""Typing definitions for npdsp."""

from typing import Any, TypeAlias

import numpy.typing as npt

Signal: TypeAlias = npt.NDArray[Any]
SignalLike: TypeAlias = npt.ArrayLike

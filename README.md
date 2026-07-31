# npDSP

A small, composable digital signal processing library built on NumPy.

npDSP provides simple DSP blocks that can be combined into pipelines using Python's `>>` operator.

## Installation

Install from PyPI:

```bash
pip install npDSP
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv add npDSP
```

## Quick start

```python
import numpy as np
import npdsp

pipeline = (
    npdsp.Add(1)
    >> npdsp.Multiply(2)
)

x = np.array([1, 2, 3])

y = pipeline(x)

print(y)
# [ 4  6  8]
```

Blocks can also be given names and accessed through a pipeline:

```python
pipeline = (
    npdsp.Add(1, name="offset")
    >> npdsp.Multiply(2, name="gain")
)

print(pipeline["gain"])
```

## Features

* NumPy-based signal processing
* Composable processing blocks
* Pipeline construction with `>>`
* Stateful processing blocks such as `Delay`
* Mathematical operations
* Conversion and utility blocks
* Type hints
* Block profiling
* Sphinx API documentation

## Documentation

Documentation is available on Read the Docs.

The documentation includes:

* Getting Started
* Concepts
* API Reference
* Block reference
* Core API reference
* Examples

## Development

npDSP uses [uv](https://docs.astral.sh/uv/) for development and dependency management.

Clone the repository:

```bash
git clone https://github.com/mrhiemstra/npDSP.git
cd npDSP
```

Install the development dependencies:

```bash
uv sync
```

Run the test suite:

```bash
uv run pytest
```

Build the documentation:

```bash
uv run sphinx-build -b html docs/source docs/build/html
```

Build the package:

```bash
uv build
```

## Requirements

* Python 3.10+
* NumPy 2.2.6+
* si-prefix 1.3.3+
* typing-extensions 4.16.0+

## License

npDSP is released under the MIT License.

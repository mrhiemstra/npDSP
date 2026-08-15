npDSP
=====

**Composable digital signal processing in NumPy.**

``npDSP`` is a lightweight Python library for building digital signal
processing systems from composable processing blocks.

The core idea is simple: **build DSP as a pipeline of blocks** and connect
them with Python's ``>>`` operator.

.. code-block:: python

   import numpy as np
   import npdsp

   pipeline = npdsp.Add(1) >> npdsp.Multiply(2)

   x = np.array([1, 2, 3])
   y = pipeline(x)

   print(y)
   # [4 6 8]

npDSP provides mathematical blocks, **FIR and IIR filters**, and stateful
blocks that can be used naturally in **streaming applications**.

Features
--------

* **Composable DSP blocks** — build processing chains from small, reusable
  components.
* **``>>`` pipeline composition** — express a DSP chain directly in Python.
* **NumPy-based** — process NumPy arrays without introducing a separate signal
  representation.
* **FIR filters** — finite impulse response filtering.
* **IIR filters** — infinite impulse response filtering with persistent state.
* **Streaming processing** — process successive chunks of samples while
  stateful blocks retain their state.
* **Mathematical blocks** — use mathematical operations as composable DSP
  blocks.
* **Stateful blocks** — blocks can maintain state between calls.
* **Named blocks** — name and access blocks within a pipeline.
* **Profiling** — inspect processing performance.

Installation
------------

Install from PyPI:

.. code-block:: bash

   pip install npDSP

Or with ``uv``:

.. code-block:: bash

   uv add npDSP

Pipelines
---------

The fundamental building block in npDSP is the **processing block**.

Blocks can be composed with ``>>``:

.. code-block:: python

   pipeline = npdsp.Add(1) >> npdsp.Multiply(2) >> some_filter

The resulting pipeline is callable:

.. code-block:: python

   output = pipeline(input)

This keeps a DSP system readable: the pipeline definition describes the order in
which the signal is processed.

FIR and IIR filters
-------------------

npDSP includes both **FIR** and **IIR** filtering.

Because filters are ordinary npDSP blocks, they can be combined directly with
mathematical operations and other processing blocks.

For example:

.. code-block:: python

   pipeline = preprocessing >> fir_filter >> iir_filter >> postprocessing

The FIR/IIR case is particularly useful for streaming because the filter's
internal state can persist between successive calls.

Streaming
---------

npDSP is designed to work naturally with streaming data.

Suppose a device continuously provides chunks of samples. You can put an npDSP
pipeline directly between the device and whatever consumes the processed signal:

.. code-block:: python

   while True:
       samples = streaming_device.get_samples()

       output = pipeline(samples)

       do_something_with(output)

The pipeline does not need to know where the samples came from. Each call
processes the next chunk.

For stateful blocks, such as IIR filters, the state is retained between calls:

.. code-block:: text

   Streaming device
         │
         │  get_samples()
         ▼
   ┌─────────────┐
   │   samples   │
   └──────┬──────┘
          │
          ▼
   ┌─────────────────────────────┐
   │        npDSP pipeline       │
   │                             │
   │  block → FIR → IIR → block  │
   │             │               │
   │             └── state ──────┤
   └─────────────┬───────────────┘
                 │
                 ▼
              output
                 │
                 ▼
          your application
                 │
                 │
                 └─────── repeat

So a stream can be processed incrementally:

.. code-block:: python

   while True:
       samples = streaming_device.get_samples()
       output = pipeline(samples)

       # Write to an output device, analyse it,
       # visualise it, encode it, etc.
       consume(output)

The important part is that **the pipeline persists across iterations**. A
stateful block sees the chunks as consecutive parts of the same signal rather
than independent signals.

This makes the same DSP components useful for applications such as:

* real-time audio processing
* data acquisition
* sensor processing
* streaming analysis
* hardware I/O
* other applications where samples arrive continuously

Batch processing
----------------

The same pipeline can also be used on a complete NumPy array:

.. code-block:: python

   output = pipeline(samples)

There is no separate streaming API that you need to learn. Streaming simply
means calling the same pipeline repeatedly as new chunks arrive.

Mathematical blocks
-------------------

Mathematical operations are also available as blocks.

For example:

.. code-block:: python

   pipeline = npdsp.Add(1) >> npdsp.Multiply(2)

This allows simple mathematical transformations to be combined with filters and
other DSP operations without leaving the pipeline abstraction.

Stateful processing
-------------------

Some DSP operations need to remember previous samples or previous processing
state.

npDSP blocks can be stateful, allowing them to maintain this information
between calls.

For example, an IIR filter can be called repeatedly:

.. code-block:: python

   while True:
       samples = streaming_device.get_samples()
       output = iir_filter(samples)

       consume(output)

The next call continues from the state established by the previous call.

This is especially important when a signal is split into chunks. Processing each
chunk independently would introduce discontinuities at the chunk boundaries; a
stateful block can instead carry the required state from one chunk to the next.

Named blocks
------------

Blocks can be given names:

.. code-block:: python

   pipeline = npdsp.Add(1, name="offset") >> npdsp.Multiply(2, name="gain")

Named blocks can then be accessed from the pipeline:

.. code-block:: python

   gain = pipeline["gain"]

This can be useful when inspecting or working with larger processing chains.

Example
-------

A complete streaming DSP application can be as simple as:

.. code-block:: python

   pipeline = preprocessing >> fir_filter >> iir_filter >> postprocessing

   while True:
       samples = streaming_device.get_samples()
       output = pipeline(samples)

       output_device.write(output)

The application controls the stream. npDSP handles the processing.

Documentation
-------------

Documentation is available at:

**https://npdsp.readthedocs.io/**

It includes the API reference, concepts, examples, and block documentation.

Development
-----------

Clone the repository:

.. code-block:: bash

   git clone https://github.com/mrhiemstra/npDSP.git
   cd npDSP

Install the development environment:

.. code-block:: bash

   uv sync

Run the tests:

.. code-block:: bash

   uv run pytest

Build the documentation:

.. code-block:: bash

   uv run sphinx-build -b html docs/source docs/build/html

Build the package:

.. code-block:: bash

   uv build

Requirements
------------

See ``pyproject.toml`` for the complete dependency specification.

Project status
--------------

npDSP is currently in **Alpha**. The API may change between releases.

Links
-----

* **PyPI:** https://pypi.org/project/npDSP/
* **GitHub:** https://github.com/mrhiemstra/npDSP
* **Documentation:** https://npdsp.readthedocs.io/

License
-------

npDSP is released under the **MIT License**.
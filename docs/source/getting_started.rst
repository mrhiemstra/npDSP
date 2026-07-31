Getting Started
===============

Installation
------------

Install npDSP using pip::

    pip install npDSP

Quick Start
-----------

Blocks can be composed into pipelines using the ``>>`` operator::

    import npdsp as dsp

    pipeline = (
        dsp.Multiply(2)
        >> dsp.Add(1)
    )

    output = pipeline([1, 2, 3])

    #Expected output = [3, 5, 7]

The pipeline applies each block in sequence.
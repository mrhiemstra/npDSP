Getting Started
===============

Installation
------------

Install npDSP using pip::

    pip install npDSP

Quick Start
-----------

Blocks can be composed into pipelines using the ``>>`` operator::
    
    >>> import npdsp as dsp
    >>> pipeline = dsp.Multiply(2) >> dsp.Add(1)
    >>> pipeline([1, 2, 3])
    array([3, 5, 7])

The pipeline applies each block in sequence. It is also possible to process only a slice of the pipeline::

    >>> import npdsp as dsp
    >>> pipeline = dsp.Multiply(2, name='mult_2') >> dsp.Add(1, name='add_1')
    >>> pipeline[:'add_1']([1,2,3]) # Processes from input pipeline up to input of 'add_1'
    array([2, 4, 6])
    >>> pipeline[:'mult_2',...]([1,2,3])  # Processes from input pipeline up to output of 'mult_2'
    array([2, 4, 6])
    >>> pipeline['add_1':]([1,2,3]) # Processes from input 'add_1' up to output of pipeline
    array([2, 3, 4])

Concepts
========

Blocks
------

A :class:`~npdsp.Block` is the fundamental processing unit in npDSP.
A block receives a signal and produces a signal.

Pipelines
---------

A :class:`~npdsp.Pipeline` connects multiple blocks together into a
processing chain.

Blocks can be composed into a :class:`~npdsp.Pipeline` using the ``>>`` operator::

    pipeline = dsp.Multiply(2) >> dsp.Add(1)

Signals
-------

npDSP uses NumPy arrays as its signal representation. Blocks operate on
signals and can maintain internal state when required.

The definition of Signal/SignalLike is as follows::

    Signal: TypeAlias = npt.NDArray[Any]
    SignalLike: TypeAlias = npt.ArrayLike

Stateful Blocks
---------------

Some blocks maintain state between calls. For example, :class:`~npdsp.Delay`
stores samples from previous calls.

State can be cleared using :meth:`~npdsp.Block.reset`.
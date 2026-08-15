02_named_blocks
===============
Code
----

.. code-block:: python
   :linenos:

   import numpy as np
   
   import npdsp as dsp
   
   
   def main() -> None:
       pipeline = (
           dsp.Add(2, name="offset")
           >> dsp.Multiply(0.5, name="gain")
           >> dsp.Delay(3, name="delay")
       )
   
       print(pipeline)
       print(pipeline["gain"])
   
       x = np.arange(5)
       print(pipeline(x))
       print(pipeline[:"gain"](x))  # Pipeline data upto the input of "gain"
       print(pipeline[:"gain", ...](x))  # Pipeline data upto the output of "gain"
       print(pipeline["gain":](x))  # From input "gain" upto end of pipeline
       print(pipeline["gain"](x))  # From input "gain" to output of "gain"
   
   
   if __name__ == "__main__":
       main()

Output
------

.. code-block:: text

   Add(name='offset', value=2) >> Multiply(name='gain', _sample_rate=1.0, value=0.5) >> Delay(name='delay', _sample_rate=1.0, samples=3, _history=<npdsp.core.buffer.SlidingBuffer object at 0x0000017DBEA8F230>)
   Multiply(name='gain', _sample_rate=1.0, value=0.5)
   [0.  0.  0.  1.  1.5]
   [2 3 4 5 6]
   [1.  1.5 2.  2.5 3. ]
   [2.  2.5 3.  0.  0.5]
   [0.  0.5 1.  1.5 2. ]
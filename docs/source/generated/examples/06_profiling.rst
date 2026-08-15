06_profiling
============
Code
----

.. code-block:: python
   :linenos:

   import numpy as np
   
   import npdsp as dsp
   
   
   def main() -> None:
       pipeline = (
           dsp.Add(1, name="offset")
           >> dsp.Multiply(2, name="gain")
           >> dsp.Delay(128, name="delay")
       )
   
       x = np.random.randn(100_000)
   
       results = pipeline.profile(x, runs=1_000)
   
       print(results)
   
   
   if __name__ == "__main__":
       main()

Output
------

.. code-block:: text

   name  | min_time|mean_time|  max_time|percent
   ------+---------+---------+----------+-------
   offset|30.300 µs|33.757 µs|373.500 µs| 30.93%
   gain  |30.300 µs|33.334 µs|342.800 µs| 30.54%
   delay |34.000 µs|42.061 µs|464.500 µs| 38.53%
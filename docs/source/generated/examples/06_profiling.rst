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
   
       x = np.random.Generator(np.random.PCG64()).standard_normal(100_000)
   
       results = pipeline.profile(x, runs=1_000)
   
       print(results)
   
   
   if __name__ == "__main__":
       main()

Output
------

.. code-block:: text

   name  | min_time|mean_time|  max_time|percent
   ------+---------+---------+----------+-------
   offset|30.800 µs|37.773 µs|636.400 µs| 31.59%
   gain  |31.000 µs|37.293 µs|360.100 µs| 31.18%
   delay |35.300 µs|44.524 µs|914.000 µs| 37.23%
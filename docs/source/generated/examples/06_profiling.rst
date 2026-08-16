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
   offset|31.200 µs|35.944 µs|407.300 µs| 30.62%
   gain  |31.100 µs|35.795 µs|378.800 µs| 30.49%
   delay |35.500 µs|45.664 µs|883.000 µs| 38.90%
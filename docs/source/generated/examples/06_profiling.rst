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
   offset|32.200 µs|36.932 µs|398.100 µs| 28.75%
   gain  |32.400 µs|40.980 µs|403.400 µs| 31.90%
   delay |37.400 µs|50.545 µs|771.400 µs| 39.35%
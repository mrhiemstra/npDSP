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
   offset|30.700 µs|37.901 µs|379.600 µs| 30.89%
   gain  |31.200 µs|39.836 µs|385.600 µs| 32.47%
   delay |35.100 µs|44.960 µs|523.100 µs| 36.64%
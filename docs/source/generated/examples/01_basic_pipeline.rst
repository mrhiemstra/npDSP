01_basic_pipeline
=================
Code
----

.. code-block:: python
   :linenos:

   import numpy as np
   
   import npdsp as dsp
   
   
   def main() -> None:
       pipeline = dsp.Add(2) >> dsp.Multiply(0.5) >> dsp.Delay(2)
   
       x = np.array([1, 2, 3, 4, 5])
   
       print("Input: ", x)
       print("Output:", pipeline(x))
   
   
   if __name__ == "__main__":
       main()

Output
------

.. code-block:: text

   Input:  [1 2 3 4 5]
   Output: [0.  0.  1.5 2.  2.5]
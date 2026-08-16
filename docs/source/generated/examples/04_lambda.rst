04_lambda
=========
Code
----

.. code-block:: python
   :linenos:

   import numpy as np
   
   import npdsp as dsp
   
   
   def main() -> None:
       pipeline = (
           dsp.Add(1)
           >> dsp.Lambda(lambda x: x**2, name="square")
           >> dsp.Multiply(0.5)
       )
   
       x = np.arange(5)
   
       print(pipeline(x))
   
   
   if __name__ == "__main__":
       main()

Output
------

.. code-block:: text

   [ 0.5  2.   4.5  8.  12.5]
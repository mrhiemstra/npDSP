07_fir
======
Code
----

.. code-block:: python
   :linenos:

   import numpy as np
   
   from npdsp import FIR
   
   
   def main() -> None:
       # FIR coefficients are given in ascending delay order:
       #
       # y[n] = b[0] * x[n] + b[1] * x[n-1] + ...
       fir = FIR(
           [1.0, 0.5, 0.25],
           name="fir",
       )
   
       x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
   
       y = fir(x)
   
       print("Input: ", x)
       print("Output:", y)
   
       # A separate FIR instance is used because the leading input shape
       # is fixed for the lifetime of a block.
       #
       # The last dimension is the sample dimension.
       fir_channels = FIR(
           [1.0, 0.5, 0.25],
           name="fir_channels",
       )
   
       x_channels = np.array(
           [
               [1.0, 2.0, 3.0, 4.0],
               [5.0, 6.0, 7.0, 8.0],
           ]
       )
   
       y_channels = fir_channels(x_channels)
   
       print()
       print("Multichannel input:")
       print(x_channels)
       print("Multichannel output:")
       print(y_channels)
   
   
   if __name__ == "__main__":
       main()

Output
------

.. code-block:: text

   Input:  [1. 2. 3. 4. 5.]
   Output: [1.   2.5  4.25 6.   7.75]
   
   Multichannel input:
   [[1. 2. 3. 4.]
    [5. 6. 7. 8.]]
   Multichannel output:
   [[ 1.    2.5   4.25  6.  ]
    [ 5.    8.5  11.25 13.  ]]
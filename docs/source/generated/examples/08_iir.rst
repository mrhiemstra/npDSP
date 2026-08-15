08_iir
======
Code
----

.. code-block:: python
   :linenos:

   import numpy as np
   
   from npdsp import IIR
   
   
   def main() -> None:
       # IIR coefficients follow the standard difference equation:
       #
       # a[0] * y[n] =
       #     b[0] * x[n]
       #   + b[1] * x[n-1]
       #   + ...
       #   - a[1] * y[n-1]
       #   - ...
       #
       # The block keeps its internal state between calls.
       iir = IIR(
           b=[1.0],
           a=[1.0, -0.5],
           name="iir",
       )
   
       x = np.ones(5)
   
       y = iir(x)
   
       print("Input: ", x)
       print("Output:", y)
   
       # State is retained between calls.
       y_next = iir(np.ones(5))
   
       print()
       print("Second call:")
       print(y_next)
   
       # Reset returns the IIR to its initial state.
       iir.reset()
   
       y_reset = iir(np.ones(5))
   
       print()
       print("After reset:")
       print(y_reset)
   
       # IIR also supports multiple leading dimensions.
       # The last dimension is the sample dimension.
       iir.reset()
   
       x_channels = np.array(
           [
               [1.0, 2.0, 3.0, 4.0],
               [5.0, 6.0, 7.0, 8.0],
           ]
       )
   
       y_channels = iir(x_channels)
   
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

   Input:  [1. 1. 1. 1. 1.]
   Output: [1.     1.5    1.75   1.875  1.9375]
   
   Second call:
   [1.96875    1.984375   1.9921875  1.99609375 1.99804688]
   
   After reset:
   [1.     1.5    1.75   1.875  1.9375]
   
   Multichannel input:
   [[1. 2. 3. 4.]
    [5. 6. 7. 8.]]
   Multichannel output:
   [[ 1.     2.5    4.25   6.125]
    [ 5.     8.5   11.25  13.625]]
import numpy as np

import npdsp as dsp


def main() -> None:

    delay = dsp.Delay(2)

    x = np.array([
        [1, 10],
        [2, 20],
        [3, 30],
        [4, 40],
        [5, 50]
    ])

    print(delay(x))

if __name__ == "__main__":
    main()
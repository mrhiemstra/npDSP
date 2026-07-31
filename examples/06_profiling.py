import numpy as np

import npdsp as dsp


def main() -> None:
    pipeline = (
        dsp.Add(1, name="offset")
        >> dsp.Multiply(2, name="gain")
        >> dsp.Delay(128, name="delay")
    )

    x = np.random.randn(100_000)

    results = pipeline.profile(x, runs=10)

    print(results)


if __name__ == "__main__":
    main()

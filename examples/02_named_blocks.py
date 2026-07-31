import numpy as np

import npdsp as dsp


def main() -> None:
    pipeline = (
        dsp.Add(2, name="offset")
        >> dsp.Multiply(0.5, name="gain")
        >> dsp.Delay(3, name="delay")
    )

    print(pipeline)
    print(pipeline["gain"])

    x = np.arange(5)
    print(pipeline(x))
    print(pipeline[:"gain"](x)) #Pipeline data upto the input of "gain"
    print(pipeline[:"gain",...](x)) #Pipeline data upto the output of "gain"
    print(pipeline["gain":](x)) #From input "gain" upto end of pipeline
    print(pipeline["gain"](x)) #From input "gain" to output of "gain"

if __name__ == "__main__":
    main()
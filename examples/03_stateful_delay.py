import npdsp as dsp


def main() -> None:
    delay = dsp.Delay(2)

    print(delay([1, 2, 3]))
    print(delay([4, 5, 6]))

    delay.reset()

    print(delay([7, 8, 9]))
    print(delay([10])) # Unequal subsequent input signal lengths are allowed

if __name__ == "__main__":
    main()
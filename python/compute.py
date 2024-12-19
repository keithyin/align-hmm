import numpy as np

if __name__ == "__main__":

    data = np.array([0.0, -7.90896292751124, -6.407116341293261, -8.465948168721212])
    data = np.exp(data)
    print(data / data.sum())

import numpy as np

def franke_function(x, y):
    term1 = 0.75 * np.exp(-( (9*x - 2)**2 / 4.0 + (9*y - 2)**2 / 4.0))
    term2 = 0.75 * np.exp(-( (9*x + 1)**2 / 49.0 + (9*y + 1)**2 / 10.0))
    term3 = 0.5 * np.exp(-( (9*x - 7)**2 / 4.0 + (9*y - 3)**2 / 4.0))
    term4 = -0.2 * np.exp(-(9*x - 4)**2 - (9*y - 7)**2)
    return np.ravel(term1 + term2 + term3 + term4)

def SkrankeFunction(x, y):
    return np.ravel(0 + 1 * x + 2 * y + 3 * x ** 2 + 4 * x * y + 5 * y ** 2)

def create_X(x, y, n):
    if len(x.shape) > 1:
        x = np.ravel(x)
        y = np.ravel(y)

    N = len(x)
    l = int((n + 1) * (n + 2) / 2)  # Number of elements in beta
    X = np.ones((N, l))

    for i in range(1, n + 1):
        q = int((i) * (i + 1) / 2)
        for k in range(i + 1):
            X[:, q + k] = (x ** (i - k)) * (y ** k)

    return X


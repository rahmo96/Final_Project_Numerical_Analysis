import numpy as np


def calculate_first_column(func, a, b, n, R):
    h = b - a
    R[0, 0] = 0.5 * h * (func(a) + func(b))

    for i in range(1, n):
        h /= 2
        sum_term = sum(func(a + k * h) for k in range(1, 2 ** i, 2))
        R[i, 0] = 0.5 * R[i - 1, 0] + h * sum_term


def calculate_remaining_columns(n, R):
    for i in range(1, n):
        for j in range(1, i + 1):
            R[i, j] = R[i, j - 1] + (R[i, j - 1] - R[i - 1, j - 1]) / ((4 ** j) - 1)


def romberg_integration(func, lower_limit, upper_limit, iterations):
    R = np.zeros((iterations, iterations), dtype=float)
    calculate_first_column(func, lower_limit, upper_limit, iterations, R)
    calculate_remaining_columns(iterations, R)
    return R[iterations - 1, iterations - 1]


def f(x):
    return 1 / (2 + x ** 4)


if __name__ == '__main__':
    a = 40
    b = 100
    n = 6
    integral = romberg_integration(f, a, b, n)

    print(f"Division into n={n} sections")
    print(f"Approximate integral in range [{a},{b}] is {integral}")

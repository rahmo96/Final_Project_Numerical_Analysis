from colors import bcolors


def newton_raphson_method(function, derivative_function, initial_guess, tolerance, max_iterations=50):

    print_iteration_header()
    current_guess = initial_guess

    for iteration in range(max_iterations):
        if is_derivative_zero(derivative_function, current_guess):
            print("Derivative is zero at current guess, method cannot continue.")
            return None

        next_guess = calculate_next_guess(function, derivative_function, current_guess)

        if has_converged(current_guess, next_guess, tolerance):
            return next_guess

        print_iteration_details(iteration, current_guess, next_guess)
        current_guess = next_guess

    return current_guess


def print_iteration_header():
    print("{:<10} {:<15} {:<15}".format("Iteration", "p0", "p1"))


def is_derivative_zero(derivative_function, point):
    return derivative_function(point) == 0


def calculate_next_guess(function, derivative_function, current_guess):
    return current_guess - function(current_guess) / derivative_function(current_guess)


def has_converged(current_guess, next_guess, tolerance):
    return abs(next_guess - current_guess) < tolerance


def print_iteration_details(iteration, current_guess, next_guess):
    print("{:<10} {:<15.9f} {:<15.9f}".format(iteration, current_guess, next_guess))


if __name__ == '__main__':
    function = lambda x: x ** 3 - 3 * x ** 2
    derivative_function = lambda x: 3 * x ** 2 - 6 * x
    initial_guess = -5
    tolerance = 1e-6
    max_iterations = 100
    root = newton_raphson_method(function, derivative_function, initial_guess, tolerance, max_iterations)
    if root is not None:
        print(bcolors.OKBLUE, "\nThe equation f(x) has an approximate root at x = {:<15.9f}".format(root), bcolors.ENDC)
    else:
        print(bcolors.FAIL, "\nThe method failed to find a root.", bcolors.ENDC)

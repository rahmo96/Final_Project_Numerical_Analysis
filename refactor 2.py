import numpy as np
class MonteCarloRefactored:
    def __init__(self, dimensions, samples, batch_size, lower_bound, upper_bound):
        self.dimensions = dimensions
        self.samples = samples
        self.batch_size = batch_size
        self.lower_bound = np.array(lower_bound)
        self.upper_bound = np.array(upper_bound)

    def _get_random_numbers(self):
        """Generates stratified random points between 0 and 1 uniformly.

        Returns:
          A tuple of (random_numbers, weights).
            - random_numbers: A 2D array of shape (samples, dimensions) containing random points.
            - weights: A 1D array of shape (samples,) containing weights (all ones for uniform).
        """
        strata_count = int(np.sqrt(self.samples))
        stratum_size = self.samples // strata_count
        random_numbers = np.zeros((self.samples, self.dimensions))

        for i in range(strata_count):
            lower = i / strata_count
            upper = (i + 1) / strata_count
            random_numbers[i * stratum_size:(i + 1) * stratum_size] = np.random.uniform(lower, upper,
                                                                                        (stratum_size, self.dimensions))

        weights = np.ones(self.samples)
        return random_numbers, weights

    def apply_bound(self, random_numbers):
        """Transforms random numbers from [0, 1] to [lower_bound, upper_bound].

        Args:
          random_numbers: A 2D array of shape (samples, dimensions) containing random points between 0 and 1.

        Returns:
          A 2D array of shape (samples, dimensions) containing transformed random numbers within limits.
        """
        return self.lower_bound + (self.upper_bound - self.lower_bound) * random_numbers

    def evaluate_function(self, random_numbers):
        """Your specific integrand function.

        Args:
          random_numbers: A 2D array of shape (samples, dimensions) containing random points.

        Returns:
          A 1D array of shape (samples,) containing the values of the integrand.
        """
        return np.cos(random_numbers[:, 0])  # Integrate sin(x)

    def process_sample_batch(self, random_numbers):
        """Calculates the integrand.

        Args:
          random_numbers

        Returns:
          A tuple of (result, error_squared).
            - result: The average value of the integrand evaluated at random_numbers.
            - error_squared: The squared error of the estimate (implementation may vary).
        """
        results = self.evaluate_function(random_numbers)
        return np.mean(results), np.var(results)  # Simple estimate for error

    def run_integration(self, iterations):
        """Performs Monte Carlo integration for a specified number of iterations."""
        for i in range(iterations):
            samples = self.generate_and_transform_samples()
            integral_estimate, error_estimate = self.accumulate_results(samples)
            self.print_iteration_results(i + 1, integral_estimate, error_estimate)
        return integral_estimate, error_estimate

    def generate_and_transform_samples(self):
        """Generates stratified random numbers and transforms them to the integration domain."""
        random_numbers, _ = self._get_random_numbers()
        return self.apply_bound(random_numbers)

    def accumulate_results(self, transformed_numbers):
        """Accumulates results from batch processing of samples."""
        total_result, total_error_squared = 0.0, 0.0
        for _ in range(self.samples // self.batch_size):
            batch_result, batch_error_squared = self.process_sample_batch(transformed_numbers)
            total_result += batch_result
            total_error_squared += batch_error_squared

        integral_estimate = total_result * (self.upper_bound - self.lower_bound) / (
            self.samples // self.batch_size
        )
        error_estimate = np.sqrt(total_error_squared / (self.samples // self.batch_size))
        return integral_estimate, error_estimate

    def print_iteration_results(self, iteration, estimate, error):
        """Prints the results of a single iteration in a formatted way."""
        print(f"Iteration: {iteration} - Estimate: {estimate:.4f}, Error: {error:.4f}")
if __name__ == "__main__":
    # Define number of dimensions, events, iterations, etc.
    dimensions = 1
    samples = 10000
    batch_size = 100
    lower_bound = 0
    upper_bound = 3
    iterations = 20

    # Create the Monte CarloFlow object
    mc_flow = MonteCarloRefactored(dimensions, samples, batch_size, lower_bound, upper_bound)

    # Run the integration
    results = mc_flow.run_integration(iterations)

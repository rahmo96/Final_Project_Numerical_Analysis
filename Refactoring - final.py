import numpy as np
class MonteCarloRefactored:
    def __init__(self, dimensions, samples, batch_size, lower_bound, upper_bound):
        """
                Initializes the Monte Carlo integration object.

                Args:
                    dimensions: The number of dimensions in the integration domain (int).
                    samples: The total number of random samples to generate (int).
                    batch_size: The number of samples to process in each batch (int).
                    lower_bound: An array-like object specifying the lower limits of the integration domain in each dimension.
                    upper_bound: An array-like object specifying the upper limits of the integration domain in each dimension.
                """
        self.dimensions = dimensions
        self.samples = samples
        self.batch_size = batch_size
        self.lower_bound = np.array(lower_bound)
        self.upper_bound = np.array(upper_bound)

    def _get_random_numbers(self):
        """
        Generates stratified random points within the unit hypercube [0, 1]^dimensions.

        Returns:
            random_numbers: A 2D NumPy array of shape (samples, dimensions) where each row is a random sample point.
            weights: A 1D NumPy array of shape (samples,) containing weights (all ones for uniform distribution).
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
        """
        Scales and shifts random numbers from the unit hypercube to the actual integration domain.

        Args:
            random_numbers: A 2D NumPy array of shape (samples, dimensions) containing random points between 0 and 1.

        Returns:
            A 2D NumPy array of shape (samples, dimensions) containing transformed random numbers within the specified bounds.
        """
        return self.lower_bound + (self.upper_bound - self.lower_bound) * random_numbers

    def evaluate_function(self, random_numbers):
        """
        Evaluates the integrand function at the given sample points.

        Args:
            random_numbers: A 2D NumPy array of shape (samples, dimensions) containing sample points within the integration domain.

        Returns:
            A 1D NumPy array of shape (samples,) containing the function values at the sample points.
        """
        return random_numbers[: , 0] **2  # Integrate sin(x)

    def process_sample_batch(self, random_numbers):
        """
        Processes a batch of samples, evaluating the integrand function and calculating statistics.

        Args:
            random_numbers: A 2D NumPy array of shape (batch_size, dimensions) containing sample points.

        Returns:
            result: The mean value of the integrand function for the batch.
            error_squared: The variance of the integrand function for the batch (estimate of error).
        """
        results = self.evaluate_function(random_numbers)
        return np.mean(results), np.var(results)  # Simple estimate for error

    def run_integration(self, iterations):
        """
        Performs Monte Carlo integration for the specified number of iterations.

        Args:
            iterations: The number of iterations to run (int).

        Returns:
            integral_estimate: The final estimated value of the integral.
            error_estimate: The estimated error of the integral.
        """
        for i in range(iterations):
            samples = self.generate_and_transform_samples()
            integral_estimate, error_estimate = self.accumulate_results(samples)
            self.print_iteration_results(i + 1, integral_estimate, error_estimate)
        return integral_estimate, error_estimate

    def generate_and_transform_samples(self):
        """
        Generates stratified random samples and transforms them to the integration domain.

        Returns:
            A 2D NumPy array of shape (samples, dimensions) containing transformed random numbers within the bounds.
        """
        random_numbers, _ = self._get_random_numbers()
        return self.apply_bound(random_numbers)

    def accumulate_results(self, transformed_numbers):
        """
        Accumulates results from batches, calculates the integral estimate, and estimates the error.

        Args:
            transformed_numbers: A 2D NumPy array of shape (samples, dimensions) containing transformed sample points.

        Returns:
            integral_estimate: The estimated value of the integral.
            error_estimate: The estimated error of the integral.
        """
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
        """
        Prints the results of a single iteration in a formatted way.

        Args:
            iteration: The iteration number (int).
            estimate: The estimated integral value for this iteration (float).
            error: The estimated error for this iteration (float).
        """
        print(f"Iteration: {iteration} - Estimate: {estimate:.4f}, Error: {error:.4f}")
if __name__ == "__main__":
    # Define number of dimensions, events, iterations, etc.
    dimensions = 1
    samples = 10000
    batch_size = 100
    lower_bound = 0
    upper_bound = 1
    iterations = 20

    # Create the Monte CarloFlow object
    mc_flow = MonteCarloRefactored(dimensions, samples, batch_size, lower_bound, upper_bound)

    # Run the integration
    results = mc_flow.run_integration(iterations)

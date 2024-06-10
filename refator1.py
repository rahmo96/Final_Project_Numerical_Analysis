import numpy as np
class MonteCarloFlow:
    def __init__(self, n_dim, n_events, events_limit, lower_limit, upper_limit, verbose=0):
        self.n_dim = n_dim
        self.n_events = n_events
        self.events_limit = events_limit
        self.lower_limit = lower_limit
        self.upper_limit = upper_limit
        self.verbose = verbose
        self._history = []  # Optional: store history if needed

    def _get_random_numbers(self):
        """Generates stratified random points between 0 and 1 uniformly.

        Returns:
          A tuple of (random_numbers, weights).
            - random_numbers: A 2D array of shape (n_events, n_dim) containing random points.
            - weights: A 1D array of shape (n_events,) containing weights (all ones for uniform).
        """
        strata_count = int(np.sqrt(self.n_events))
        stratum_size = self.n_events // strata_count
        random_numbers = np.zeros((self.n_events, self.n_dim))

        for i in range(strata_count):
            lower = i / strata_count
            upper = (i + 1) / strata_count
            random_numbers[i * stratum_size:(i + 1) * stratum_size] = np.random.uniform(lower, upper,
                                                                                        (stratum_size, self.n_dim))

        weights = np.ones(self.n_events)
        return random_numbers, weights

    def _apply_limits(self, random_numbers):
        """Transforms random numbers from [0, 1] to [lower_limit, upper_limit].

        Args:
          random_numbers: A 2D array of shape (n_events, n_dim) containing random points between 0 and 1.

        Returns:
          A 2D array of shape (n_events, n_dim) containing transformed random numbers within limits.
        """
        return self.lower_limit + (self.upper_limit - self.lower_limit) * random_numbers

    def _integrand(self, random_numbers):
        """Your specific integrand function.

        Args:
          random_numbers: A 2D array of shape (n_events, n_dim) containing random points.

        Returns:
          A 1D array of shape (n_events,) containing the values of the integrand.
        """
        return np.cos(random_numbers[:, 0])  # Integrate sin(x)

    def _run_event(self, random_numbers):
        """Calculates the integrand.

        Args:
          random_numbers: A 2D array of shape (n_events, n_dim) containing random points.

        Returns:
          A tuple of (result, error_squared).
            - result: The average value of the integrand evaluated at random_numbers.
            - error_squared: The squared error of the estimate (implementation may vary).
        """
        results = self._integrand(random_numbers)
        return np.mean(results), np.var(results)  # Simple estimate for error

    def run_integration(self, n_iterations):
        for i in range(n_iterations):
            # Generate random numbers and apply limits
            random_numbers, weights = self._get_random_numbers()
            transformed_numbers = self._apply_limits(random_numbers)

            # Run a single event (multiple calls) and accumulate results
            total_result, total_error_squared = 0.0, 0.0
            for _ in range(self.n_events // self.events_limit):
                batch_result, batch_error_squared = self._run_event(transformed_numbers)
                total_result += batch_result
                total_error_squared += batch_error_squared

            # Adjust results based on weights and number of events
            integral_estimate = total_result * (self.upper_limit - self.lower_limit) / (self.n_events // self.events_limit)
            error_estimate = np.sqrt(total_error_squared / (self.n_events // self.events_limit))

            # Optional: store history
            self._history.append((integral_estimate, error_estimate))

            if self.verbose > 0:
                # Print iteration progress
                print(f"Iteration: {i + 1} - Estimate: {integral_estimate:.4f}, Error: {error_estimate:.4f}")

        # Return final results or history if collected
        if len(self._history) > 0:
            return self._history
        else:
            return integral_estimate, error_estimate

if __name__ == "__main__":
    # Define number of dimensions, events, iterations, etc.
    n_dim = 1
    n_events = 10000
    events_limit = 100
    n_iterations = 20
    lower_limit = 0
    upper_limit = 3

    # Create the Monte CarloFlow object
    mc_flow = MonteCarloFlow(n_dim, n_events, events_limit, lower_limit, upper_limit, verbose=1)

    # Run the integration
    results = mc_flow.run_integration(n_iterations)

    if isinstance(results[0], tuple):  # Check if history is returned
        # Access results from each iteration (integral estimate, error)
        for integral_estimate, error_estimate in results:
            print(f"Iteration - Estimate: {integral_estimate:.4f}, Error: {error_estimate:.4f}")
    else:
        # Access final results (integral estimate, error)
        integral_estimate, error_estimate = results
        print(f"Integral Estimate: {integral_estimate:.4f}, Error: {error_estimate:.4f}")

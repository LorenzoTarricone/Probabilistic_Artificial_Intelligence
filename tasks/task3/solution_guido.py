"""Solution."""
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
import math
import os
import time
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, DotProduct, Matern, RBF
from scipy.stats import norm
import matplotlib.pyplot as plt


# global variables
DOMAIN = np.array([[0, 10]])  # restrict \theta in [0, 10]
SAFETY_THRESHOLD = 4  # threshold, upper bound of SA


# DONE: implement a self-contained solution in the BO_algo class.
# NOTE: main() is not called by the checker.
class BO_algo():
    i = -1

    def __init__(self):
        """Initializes the algorithm with a parameter configuration."""

        # DONE: Define all relevant class members for your BO algorithm here.

        self.points = []

        f_variance = 0.5
        f_length_scale = 10.0
        # f_smoothness = 2.5
        f_noise_variance = 0.15 ** 2
        self.f_prior_mean = 0.0
        self.f_gp = GaussianProcessRegressor(
            kernel=(
                ConstantKernel(f_variance) *
                RBF(f_length_scale, length_scale_bounds="fixed")
            ),
            # kernel=(
            #     Matern(nu=f_smoothness, length_scale_bounds="fixed")
            # ),
            alpha=f_noise_variance,
        )

        # v_variance = math.sqrt(2)
        # v_length_scale = 10.0
        v_smoothness = 2.5
        v_noise_variance = 0.0001 ** 2
        self.v_prior_mean = 4.0
        self.v_gp = GaussianProcessRegressor(
            # kernel=(
            #     DotProduct(0, sigma_0_bounds="fixed") + (
            #         ConstantKernel(v_variance) *
            #         RBF(v_length_scale, length_scale_bounds="fixed")
            #     )
            # ),
            kernel=(
                DotProduct(0, sigma_0_bounds="fixed") +
                Matern(nu=v_smoothness, length_scale_bounds="fixed")
            ),
            alpha=v_noise_variance,
        )

        BO_algo.i += 1

    def next_recommendation(self):
        """
        Recommend the next input to sample.

        Returns
        -------
        recommendation: float
            the next point to evaluate
        """
        # DONE: Implement the function which recommends the next point to query
        # using functions f and v.
        # In implementing this function, you may use
        # optimize_acquisition_function() defined below.

        return self.optimize_acquisition_function()

        # suggestion = self.optimize_acquisition_function()
        # N = 8
        # tol = .01
        # if len(self.previous_points) < N:
        #     return suggestion
        # min_distance = np.max(self.get_distances(suggestion, N=N))
        # if min_distance < tol:
        #     return self.random_point_from_domain()
        # else:
        #     return suggestion

    def optimize_acquisition_function(self):
        """Optimizes the acquisition function defined below (DO NOT MODIFY).

        Returns
        -------
        x_opt: float
            the point that maximizes the acquisition function, where
            x_opt in range of DOMAIN
        """

        def objective(x):
            return -self.acquisition_function(x)

        f_values = []
        x_values = []

        # Restarts the optimization 20 times and pick best solution
        for _ in range(20):
            x0 = DOMAIN[:, 0] + (DOMAIN[:, 1] - DOMAIN[:, 0]) * \
                 np.random.rand(DOMAIN.shape[0])
            result = fmin_l_bfgs_b(objective, x0=x0, bounds=DOMAIN,
                                   approx_grad=True)
            x_values.append(np.clip(result[0], *DOMAIN[0]))
            f_values.append(-result[1])

        ind = np.argmax(f_values)
        x_opt = x_values[ind].item()

        return x_opt

    def acquisition_function(self, x: np.ndarray):
        """Compute the acquisition function for x.

        Parameters
        ----------
        x: np.ndarray
            x in domain of f, has shape (N, 1)

        Returns
        ------
        af_value: np.ndarray
            shape (N, 1)
            Value of the acquisition function at x
        """
        x = np.atleast_2d(x)

        # DONE: Implement the acquisition function you want to optimize.

        # safe_points = [point for point in self.points if point[2] <= SAFETY_THRESHOLD]
        safe_points = self.points

        def f_ei(x: np.ndarray, xi: float):
            f_max = np.max(safe_points, axis=0)[1]
            f_mean, f_std = self.f_gp.predict(x, return_std=True)
            z = (f_max - f_mean - xi) / f_std
            return (f_max - f_mean - xi) * norm.cdf(z) + f_std * norm.pdf(z)

        def v_pr(x: np.ndarray):
            v_mean, v_std = self.v_gp.predict(x, return_std=True)
            return norm.cdf(SAFETY_THRESHOLD - self.v_prior_mean, v_mean, v_std)

        return f_ei(x, xi=0.01) * v_pr(x) if safe_points else v_pr(x)

    def add_data_point(self, x: float, f: float, v: float):
        """
        Add data points to the model.

        Parameters
        ----------
        x: float
            structural features
        f: float
            logP obj func
        v: float
            SA constraint func
        """
        x, f, v = float(x), float(f), float(v)

        # DONE: Add the observed data {x, f, v} to your model.

        self.points.append([x, f, v])

        x_values = [[point[0]] for point in self.points]
        f_values = [point[1] - self.f_prior_mean for point in self.points]
        v_values = [point[2] - self.v_prior_mean for point in self.points]

        self.f_gp.fit(x_values, f_values)
        self.v_gp.fit(x_values, v_values)

    def get_solution(self):
        """
        Return x_opt that is believed to be the maximizer of f.

        Returns
        -------
        solution: float
            the optimal solution of the problem
        """

        # DONE: Return your predicted safe optimum of f.

        # self.plot(plot_recommendation=True)

        safe_points = [point for point in self.points if point[2] <= SAFETY_THRESHOLD]
        f_max_idx = np.argmax(safe_points, axis=0)[1]
        f_max_x = safe_points[f_max_idx][0]
        return f_max_x

    def plot(self, plot_recommendation: bool = True):
        """Plot objective and constraint posterior for debugging (OPTIONAL).

        Parameters
        ----------
        plot_recommendation: bool
            Plots the recommended point if True.
        """

        np_indices = [[i] for i in range(len(self.points))]
        np_points = np.hstack((self.points, np_indices))
        np_not_safe = np_points[np_points[:, 2] >= SAFETY_THRESHOLD]
        np_safe = np_points[np_points[:, 2] < SAFETY_THRESHOLD]

        x_plot = np.linspace(0, 10, 1000)[:, np.newaxis]

        f_mean, f_std = self.f_gp.predict(x_plot, return_std=True)
        f_mean += self.f_prior_mean
        f_plot = [f(i) for i in x_plot]

        v_mean, v_std = self.v_gp.predict(x_plot, return_std=True)
        v_mean += self.v_prior_mean
        v_plot = [v(i) for i in x_plot]

        cmap_not_safe = plt.cm.Reds(np.linspace(0.2, 1, len(np_points)))
        cmap_safe = plt.cm.Blues(np.linspace(0.2, 1, len(np_points)))

        def plot_1(ax):
            ax.set_title("Objective Function Posterior")
            ax.scatter(np_not_safe[:, 0], np_not_safe[:, 1],
                       c=cmap_not_safe[np_not_safe[:, 3].astype(int)], marker="o",
                       label="Unsafe (Above Threshold)")
            ax.scatter(np_safe[:, 0], np_safe[:, 1],
                       c=cmap_safe[np_safe[:, 3].astype(int)], marker="o",
                       label="Safe (Below Threshold)")
            ax.plot(x_plot, f_mean, "k", label="Objective Function Posterior (Mean)")
            ax.plot(x_plot, f_plot, "k", alpha=0.2, label="Objective Function")
            ax.fill_between(x_plot.ravel(), f_mean - 1.96 * f_std,
                            f_mean + 1.96 * f_std, alpha=0.2, color="k",
                            label="95% Confidence Interval")
            ax.legend()

        def plot_2(ax):
            ax.set_title("Constraint Function Posterior")
            ax.scatter(np_not_safe[:, 0], np_not_safe[:, 2],
                       c=cmap_not_safe[np_not_safe[:, 3].astype(int)], marker="o",
                       label="Unsafe (Above Threshold)")
            ax.scatter(np_safe[:, 0], np_safe[:, 2],
                       c=cmap_safe[np_safe[:, 3].astype(int)], marker="o",
                       label="Safe (Below Threshold)")
            ax.plot(x_plot, v_mean, "k", label="Constraint Function Posterior (Mean)")
            ax.plot(x_plot, v_plot, "k", alpha=0.2, label="Constraint Function")
            ax.axhline(y=SAFETY_THRESHOLD, color="k", linestyle="dotted", label="Safe Threshold")
            ax.fill_between(x_plot.ravel(), v_mean - 1.96 * v_std,
                            v_mean + 1.96 * v_std, alpha=0.2, color="k",
                            label="95% Confidence Interval")
            ax.legend()

        _, axs = plt.subplots(2, figsize=(15, 15))

        plot_1(axs[0])
        plot_2(axs[1])

        plt.savefig(f"out/{BO_algo.i}.png")
        plt.show()


# ---
# TOY PROBLEM. To check your code works as expected (ignored by checker).
# ---

def check_in_domain(x: float):
    """Validate input"""
    x = np.atleast_2d(x)
    return np.all(x >= DOMAIN[None, :, 0]) and np.all(x <= DOMAIN[None, :, 1])


def f(x: float):
    """Dummy logP objective"""
    mid_point = DOMAIN[:, 0] + 0.5 * (DOMAIN[:, 1] - DOMAIN[:, 0])
    return - np.linalg.norm(x - mid_point, 2)


def v(x: float):
    """Dummy SA"""
    return np.random.uniform(0, 8)


def get_initial_safe_point():
    """Return initial safe point"""
    x_domain = np.linspace(*DOMAIN[0], 4000)[:, None]
    c_val = np.vectorize(v)(x_domain)
    x_valid = x_domain[c_val < SAFETY_THRESHOLD]
    np.random.seed(0)
    np.random.shuffle(x_valid)
    x_init = x_valid[0].item()
    print('x_init', x_init)

    return x_init


def main():
    """FOR ILLUSTRATION / TESTING ONLY (NOT CALLED BY CHECKER)."""
    # Init problem
    agent = BO_algo()

    # Add initial safe point
    x_init = get_initial_safe_point()
    obj_val = f(x_init)
    cost_val = v(x_init)
    agent.add_data_point(x_init, obj_val, cost_val)

    # Loop until budget is exhausted
    for j in range(20):
        # Get next recommendation
        x = agent.next_recommendation()

        # Check for valid shape
        # assert x.shape == (1, DOMAIN.shape[0]), \
        #     f"The function next recommendation must return a numpy array of " \
        #     f"shape (1, {DOMAIN.shape[0]})"

        # Obtain objective and constraint observation
        obj_val = f(x) + np.random.normal(scale=0.15)
        cost_val = v(x) + np.random.normal(scale=0.0001)
        agent.add_data_point(x, obj_val, cost_val)

    # Validate solution
    solution = agent.get_solution()
    assert check_in_domain(solution), \
        f'The function get solution must return a point within the' \
        f'DOMAIN, {solution} returned instead'

    # Compute regret
    regret = (0 - f(solution))

    print(f'Optimal value: 0\nProposed solution {solution}\nSolution value '
          f'{f(solution)}\nRegret {regret}\nUnsafe-evals TODO\n')


if __name__ == "__main__":
    main()

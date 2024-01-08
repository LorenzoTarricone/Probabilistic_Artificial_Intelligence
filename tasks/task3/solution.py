"""Solution."""
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
import math
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF, DotProduct, ConstantKernel
from scipy import stats

# import additional ...


# global variables
DOMAIN = np.array([[0, 10]])  # restrict \theta in [0, 10]
SAFETY_THRESHOLD = 4  # threshold, upper bound of SA


# TODO: implement a self-contained solution in the BO_algo class.
# NOTE: main() is not called by the checker.
class BO_algo():
    def __init__(self):
        """Initializes the algorithm with a parameter configuration."""
        # TODO: Define all relevant class members for your BO algorithm here.
        self.previous_points = []
        self.f_kernel = ConstantKernel(0.5) * RBF(length_scale=10.0, length_scale_bounds=(10, 1e8))
        # self.v_kernel = DotProduct(sigma_0=0.0, sigma_0_bounds='fixed') + math.sqrt(2.0) * RBF(length_scale=10.0)
        self.v_kernel = DotProduct(sigma_0=0.0, sigma_0_bounds='fixed') + Matern(nu = 2.5, length_scale_bounds='fixed')
        self.f_model = GaussianProcessRegressor(kernel=self.f_kernel, alpha = 0.15**2)
        self.v_model = GaussianProcessRegressor(kernel=self.v_kernel, alpha = 0.0001**2)
        self.lam = 0.4
        self.prior_mean = 4.0
        self.beta1 = 0.475
        self.beta2 = 1


    def next_recommendation(self):
        """
        Recommend the next input to sample.

        Returns
        -------
        recommendation: float
            the next point to evaluate
        """
        # TODO: Implement the function which recommends the next point to query
        # using functions f and v.
        # In implementing this function, you may use
        # optimize_acquisition_function() defined below.
        if len(self.previous_points) == 0:
            #Return initial safe point
            x_domain = np.linspace(*DOMAIN[0], 4000)[:, None]
            c_val = np.vectorize(v)(x_domain)
            x_valid = x_domain[c_val < SAFETY_THRESHOLD]
            np.random.seed(0)
            np.random.shuffle(x_valid)
            x_init = x_valid[0]

            return x_init
        else:
            return self.optimize_acquisition_function()

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
        
        f_mean, f_std = self.f_model.predict(x, return_std=True)
        v_mean, v_std = self.v_model.predict(x, return_std=True)

        af_value = f_mean + self.beta1 * f_std - self.lam * (v_mean + self.beta2 * v_std)
        return af_value
    
        # t = float('-inf')
        # for point in self.previous_points:
        #     if point[2] < SAFETY_THRESHOLD and point[1] > t:
        #         t = point[1]
        # z = (t - f_mean)/f_std
        
        # expected_improvement = f_std * ( z * stats.norm.cdf(z) + stats.norm.pdf(z))
        # constr_prob = stats.norm.cdf(SAFETY_THRESHOLD - self.prior_mean, v_mean, v_std)
        
        # return expected_improvement*constr_prob

        # t = np.array(self.previous_points)[:,1].min()
        # prob_constraint = norm.cdf(self.prior_mean, loc=v_mean, scale=v_std)

        # z_x = (t - f_mean - 0.01)/f_std

        # ei_x =  f_std *(z_x * norm.cdf(z_x) + norm.pdf(z_x))
            
        # return prob_constraint * ei_x


        

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
        # TODO: Add the observed data {x, f, v} to your model.
        self.previous_points.append([float(x), float(f), float(v)])

        data = np.array(self.previous_points)

        # self.f_model = GaussianProcessRegressor(kernel=self.f_kernel, alpha = 0.15**2)
        # self.v_model = GaussianProcessRegressor(kernel=self.v_kernel, alpha = 0.0001**2)

        self.f_model.fit(data[:, 0].reshape(-1, 1), data[:, 1])  # Reshape the input to a 2D array
        self.v_model.fit(data[:, 0].reshape(-1, 1), data[:, 2] - self.prior_mean)
        #self.v_model.fit(data[:, 0].reshape(-1, 1), data[:, 2])

        self.green_intervals()

    def find_extremes_above_reference(self, data, reference):
        below_reference = False
        intervals = []

        for i, value in enumerate(data):
            if value > reference:
                if not below_reference:
                    below_reference = True
                    start_index = i
            else:
                if below_reference:
                    below_reference = False
                    end_index = i - 1
                    intervals.append((start_index, end_index))

        # Check if the last interval extends to the end of the list
        if below_reference:
            intervals.append((start_index, len(data) - 1))
        
        return intervals

    def green_intervals(self):
        DOMAIN = np.array([[0, 10]])  # restrict \theta in [0, 10]
        x_domain = np.linspace(*DOMAIN[0], 4000)[:, None]
        f_mean, f_std = self.f_model.predict(x_domain, return_std=True)
        v_mean, v_std = self.v_model.predict(x_domain, return_std=True)

        lower_bound = f_mean - f_std - self.lam * (v_mean - v_std)
        upper_bound = f_mean + f_std - self.lam * (v_mean + v_std)

        # find maximum of lower bound
        blb = np.max(lower_bound)

        extremes = self.find_extremes_above_reference(list(upper_bound), float(blb))
        extremes = np.array(extremes) / 4000 * 10

        DOMAIN = np.array(extremes)


    def get_solution(self):
        """
        Return x_opt that is believed to be the maximizer of f.

        Returns
        -------
        solution: float
            the optimal solution of the problem
        """
        # TODO: Return your predicted safe optimum of f.
        # data = np.array(self.previous_points)
        # data = data[data[:,-1]<=SAFETY_THRESHOLD]
        
        # return data[data[:,1].argmax(),0]

        max = float('-inf')
        x_max = 0
        for point in self.previous_points:
            if point[2] < SAFETY_THRESHOLD and point[1] > max:
                max = point[1]
                x_max = point[0]
        return x_max

    def plot(self, plot_recommendation: bool = True):
        """Plot objective and constraint posterior for debugging (OPTIONAL).

        Parameters
        ----------
        plot_recommendation: bool
            Plots the recommended point if True.
        """
        pass


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
    return 2.0


def get_initial_safe_point():
    """Return initial safe point"""
    x_domain = np.linspace(*DOMAIN[0], 4000)[:, None]
    c_val = np.vectorize(v)(x_domain)
    x_valid = x_domain[c_val < SAFETY_THRESHOLD]
    np.random.seed(0)
    np.random.shuffle(x_valid)
    x_init = x_valid[0]

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
        #assert x.shape == (1, DOMAIN.shape[0]), \
        #    f"The function next recommendation must return a numpy array of " \
        #    f"shape (1, {DOMAIN.shape[0]})"

        # Obtain objective and constraint observation
        obj_val = f(x) + np.random.randn()
        cost_val = v(x) + np.random.randn()
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

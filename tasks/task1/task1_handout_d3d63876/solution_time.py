import os
import typing
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, RBF, ConstantKernel as C, RationalQuadratic
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
import matplotlib.pyplot as plt
from matplotlib import cm

# aggiunta per Nystroem method
from sklearn import pipeline
from sklearn.kernel_approximation import Nystroem
from sklearn.model_selection import train_test_split, GridSearchCV, ParameterGrid, KFold
from sklearn.preprocessing import StandardScaler
import random

from tqdm.notebook import tqdm_notebook
import time


# Set `EXTENDED_EVALUATION` to `True` in order to visualize your predictions.
EXTENDED_EVALUATION = False
EVALUATION_GRID_POINTS = 300  # Number of grid points used in extended evaluation

# Cost function constants
COST_W_UNDERPREDICT = 50.0
COST_W_NORMAL = 1.0


class Model(object):
    """
    Model for this task.
    You need to implement the fit_model and predict methods
    without changing their signatures, but are allowed to create additional methods.
    """

    def __init__(self, kernel = C() * Matern(length_scale=0.05, nu=2.5) + WhiteKernel(noise_level=1e-05)):
        """
        Initialize your model here.
        We already provide a random number generator for reproducibility.
        """
        self.rng = np.random.default_rng(seed=0)

        # TODO: Add custom initialization for your model here if necessary
        #self.kernel = RBF(length_scale=0.5) + ConstantKernel() + Matern(length_scale=0.5, nu=1.5) + WhiteKernel(noise_level=0.5)
        # kernels: RBF(0.5), Matern(length_scale = 0.5, nu = 1.5), WhiteKernel
        # nu = 0.5 once differentiable, nu = 1.5 twice differentiable, nu = 2.5 three times differentiable
        self.gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=3, random_state=self.rng.integers(0,100))

        #feature_map_nystroem = Nystroem(kernel=kernel, random_state=1)
        #self.model = pipeline.Pipeline([("feature_map", feature_map_nystroem),
        #                                ("GP", self.gp)])
        
        self.std_scaler = StandardScaler(with_std=False)
        self.num_post_samples = 1000

    def make_predictions(self, test_x_2D: np.ndarray, test_x_AREA: np.ndarray, alpha = 0.1) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict the pollution concentration for a given set of city_areas.
        :param test_x_2D: city_areas as a 2d NumPy float array of shape (NUM_SAMPLES, 2)
        :param test_x_AREA: city_area info for every sample in a form of a bool array (NUM_SAMPLES,)
        :return:
            Tuple of three 1d NumPy float arrays, each of shape (NUM_SAMPLES,),
            containing your predictions, the GP posterior mean, and the GP posterior stddev (in that order)
        """

        # TODO: Use your GP to estimate the posterior mean and stddev for each city_area here
        gp_mean = np.zeros(test_x_2D.shape[0], dtype=float)
        gp_std = np.zeros(test_x_2D.shape[0], dtype=float)

        gp_mean, gp_std = self.gp.predict(test_x_2D, return_std=True)
        gp_mean = self.std_scaler.inverse_transform(np.tile(gp_mean, (2,1)).reshape(test_x_2D.shape[0],2))[:,0]
        #gp_mean, gp_std = self.model.predict(test_x_2D, return_std=True)

        # TODO: Use the GP posterior to form your predictions here
        #gp_samples = self.rng.normal(loc=gp_mean, scale=gp_std, size=(self.num_post_samples, len(gp_mean)))
        #gp_potential_pred = np.linspace(gp_mean - gp_std, gp_mean + gp_std, num=10)

        #costs = np.zeros((gp_samples.shape[0], gp_potential_pred[1]))
        #for i in range(gp_samples.shape[0]):
        #    for j in range(gp_potential_pred[1]):
        #        costs[i,j] = cost_function(gp_samples[i, :], gp_potential_pred[:, j])

        #costs = cost_function(np.expand_dims(gp_samples, 1), np.expand_dims(gp_potential_pred, 2), test_x_AREA)
        #predictions = gp_samples[np.argmin(costs, axis=0), np.arange(len(gp_mean))]
        predictions = gp_mean + test_x_AREA * alpha * gp_std

        return predictions, gp_mean, gp_std

    def fitting_model(self, train_y: np.ndarray,train_x_2D: np.ndarray):
        """
        Fit your model on the given training data.
        :param train_x_2D: Training features as a 2d NumPy float array of shape (NUM_SAMPLES, 2)
        :param train_y: Training pollution concentrations as a 1d NumPy float array of shape (NUM_SAMPLES,)
        """

        # TODO: Fit your model here
        # preprocess data by standardizing them
        self.std_scaler.fit(train_x_2D)
        train_x_2D = self.std_scaler.transform(train_x_2D)

        self.gp.fit(train_x_2D, train_y)

        # Using Nystroem method
        #feature_map_nystroem = Nystroem(kernel=kernel, random_state=1)
        #self.model = pipeline.Pipeline([("feature_map", feature_map_nystroem),
        #                                ("GP", self.gp)])
        #self.model.set_params(feature_map__n_components=500)
        #self.model.fit(train_x_2D, train_y)
        #nystroem_score = nystroem_approx_gp.score(test_x_2D, test_y)

        #train_x_2D_transformed = feature_map_nystroem.fit_transform(train_x_2D)
        #model.fit(train_x_2D_transformed, train_y)
        #model.score(train_x_2D_transformed, train_y)

# Compare different kernels 
def optimal_kernel(gp, train_y: np.ndarray,train_x_2D: np.ndarray):
    
    kernels = [
    RBF(length_scale=1.0),
    Matern(length_scale=1.0, nu=1.5),
    WhiteKernel(noise_level=1e-5)
    ]

    param_grid = {
    'kernel': kernels,
    }

    grid_search = GridSearchCV(estimator=gp, param_grid=param_grid, scoring='neg_mean_squared_error', cv=5)
    grid_search.fit(train_x_2D, train_y)

    # Retrieve the Best Kernel and Hyperparameters
    best_kernel = grid_search.best_params_['kernel']
    print(best_kernel)
    return best_kernel

# You don't have to change this function
def cost_function(ground_truth: np.ndarray, predictions: np.ndarray, AREA_idxs: np.ndarray) -> float:
    """
    Calculates the cost of a set of predictions.

    :param ground_truth: Ground truth pollution levels as a 1d NumPy float array
    :param predictions: Predicted pollution levels as a 1d NumPy float array
    :param AREA_idxs: city_area info for every sample in a form of a bool array (NUM_SAMPLES,)
    :return: Total cost of all predictions as a single float
    """
    assert ground_truth.ndim == 1 and predictions.ndim == 1 and ground_truth.shape == predictions.shape

    # Unweighted cost
    cost = (ground_truth - predictions) ** 2
    weights = np.ones_like(cost) * COST_W_NORMAL

    # Case i): underprediction
    mask = (predictions < ground_truth) & [bool(AREA_idx) for AREA_idx in AREA_idxs]
    weights[mask] = COST_W_UNDERPREDICT

    # Weigh the cost and return the average
    return np.mean(cost * weights)


# You don't have to change this function
def is_in_circle(coor, circle_coor):
    """
    Checks if a coordinate is inside a circle.
    :param coor: 2D coordinate
    :param circle_coor: 3D coordinate of the circle center and its radius
    :return: True if the coordinate is inside the circle, False otherwise
    """
    return (coor[0] - circle_coor[0])**2 + (coor[1] - circle_coor[1])**2 < circle_coor[2]**2

# You don't have to change this function 
def determine_city_area_idx(visualization_xs_2D):
    """
    Determines the city_area index for each coordinate in the visualization grid.
    :param visualization_xs_2D: 2D coordinates of the visualization grid
    :return: 1D array of city_area indexes
    """
    # Circles coordinates
    circles = np.array([[0.5488135, 0.71518937, 0.17167342],
                    [0.79915856, 0.46147936, 0.1567626 ],
                    [0.26455561, 0.77423369, 0.10298338],
                    [0.6976312,  0.06022547, 0.04015634],
                    [0.31542835, 0.36371077, 0.17985623],
                    [0.15896958, 0.11037514, 0.07244247],
                    [0.82099323, 0.09710128, 0.08136552],
                    [0.41426299, 0.0641475,  0.04442035],
                    [0.09394051, 0.5759465,  0.08729856],
                    [0.84640867, 0.69947928, 0.04568374],
                    [0.23789282, 0.934214,   0.04039037],
                    [0.82076712, 0.90884372, 0.07434012],
                    [0.09961493, 0.94530153, 0.04755969],
                    [0.88172021, 0.2724369,  0.04483477],
                    [0.9425836,  0.6339977,  0.04979664]])
    
    visualization_xs_AREA = np.zeros((visualization_xs_2D.shape[0],))

    for i,coor in enumerate(visualization_xs_2D):
        visualization_xs_AREA[i] = any([is_in_circle(coor, circ) for circ in circles])

    return visualization_xs_AREA

# You don't have to change this function
def perform_extended_evaluation(model: Model, output_dir: str = '/results'):
    """
    Visualizes the predictions of a fitted model.
    :param model: Fitted model to be visualized
    :param output_dir: Directory in which the visualizations will be stored
    """
    print('Performing extended evaluation')

    # Visualize on a uniform grid over the entire coordinate system
    grid_lat, grid_lon = np.meshgrid(
        np.linspace(0, EVALUATION_GRID_POINTS - 1, num=EVALUATION_GRID_POINTS) / EVALUATION_GRID_POINTS,
        np.linspace(0, EVALUATION_GRID_POINTS - 1, num=EVALUATION_GRID_POINTS) / EVALUATION_GRID_POINTS,
    )
    visualization_xs_2D = np.stack((grid_lon.flatten(), grid_lat.flatten()), axis=1)
    visualization_xs_AREA = determine_city_area_idx(visualization_xs_2D)
    
    # Obtain predictions, means, and stddevs over the entire map
    predictions, gp_mean, gp_stddev = model.make_predictions(visualization_xs_2D, visualization_xs_AREA)
    predictions = np.reshape(predictions, (EVALUATION_GRID_POINTS, EVALUATION_GRID_POINTS))
    gp_mean = np.reshape(gp_mean, (EVALUATION_GRID_POINTS, EVALUATION_GRID_POINTS))

    vmin, vmax = 0.0, 65.0

    # Plot the actual predictions
    fig, ax = plt.subplots()
    ax.set_title('Extended visualization of task 1')
    im = ax.imshow(predictions, vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(im, ax = ax)

    # Save figure to pdf
    figure_path = os.path.join(output_dir, 'extended_evaluation.pdf')
    fig.savefig(figure_path)
    print(f'Saved extended evaluation to {figure_path}')

    plt.show()


def extract_city_area_information(train_x: np.ndarray, test_x: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extracts the city_area information from the training and test features.
    :param train_x: Training features
    :param test_x: Test features
    :return: Tuple of (training features' 2D coordinates, training features' city_area information,
        test features' 2D coordinates, test features' city_area information)
    """
    train_x_2D = np.zeros((train_x.shape[0], 2), dtype=float)
    train_x_AREA = np.zeros((train_x.shape[0],), dtype=bool)
    test_x_2D = np.zeros((test_x.shape[0], 2), dtype=float)
    test_x_AREA = np.zeros((test_x.shape[0],), dtype=bool)

    #TODO: Extract the city_area information from the training and test features
    train_x_2D = train_x[:, :2]
    train_x_AREA = train_x[:, 2]
    test_x_2D = test_x[:, :2]
    test_x_AREA = test_x[:, 2]

    assert train_x_2D.shape[0] == train_x_AREA.shape[0] and test_x_2D.shape[0] == test_x_AREA.shape[0]
    assert train_x_2D.shape[1] == 2 and test_x_2D.shape[1] == 2
    assert train_x_AREA.ndim == 1 and test_x_AREA.ndim == 1

    return train_x_2D, train_x_AREA, test_x_2D, test_x_AREA

def subsample(train_array, test_array, perc):
  assert train_array.shape[0] == test_array.shape[0]

  len = train_array.shape[0]
  n_samples = round(len * perc / 100)

  lista = [random.randrange(0,len) for i in range(n_samples)]

  return train_array[lista, :], test_array[lista]

def cross_validation(train_y: np.ndarray,train_x_2D: np.ndarray, train_x_AREA: np.ndarray, nFolds = 5):
    folds = KFold(n_splits=nFolds, shuffle=True)
    
    cost = 0

    for train_index, validation_index in folds.split(train_x_2D):
        X_train_2D, X_validation_2D = train_x_2D[train_index, :], train_x_2D[validation_index, :]
        X_train_AREA, X_validation_AREA = train_x_AREA[train_index], train_x_AREA[validation_index]
        Y_train, Y_validation = train_y[train_index], train_y[validation_index]
    
        model = Model()
        model.fitting_model(Y_train, X_train_2D)

        predictions = model.make_predictions(X_validation_2D, X_validation_AREA)

        cost_fold = cost_function(Y_validation, predictions[0], X_validation_AREA)
        print(f'Cost fold: {cost_fold}')
        cost += cost_fold

    cost /= nFolds

    return cost


# you don't have to change this function
def main():
    # Load the training dateset and test features
    train_x = np.loadtxt('train_x.csv', delimiter=',', skiprows=1)
    train_y = np.loadtxt('train_y.csv', delimiter=',', skiprows=1)
    test_x = np.loadtxt('test_x.csv', delimiter=',', skiprows=1)

    # Subsample data
    #train_x_sub, train_y_sub = subsample(train_x, train_y, 40)
    # divide data into training and validation sets
    N,_ = train_x.shape
    train_size = int(0.05 * N)

    train_x_n = train_x[:train_size]
    train_x_v = train_x[train_size:]
    train_y_n = train_y[:train_size]
    train_y_v = train_y[train_size:]

    # Extract the city_area information
    #train_x_2D, train_x_AREA, test_x_2D, test_x_AREA = extract_city_area_information(train_x, test_x)
    train_x_2D, train_x_AREA, test_x_2D, test_x_AREA = extract_city_area_information(train_x_n, train_x_v)

    # best_kernel = optimal_kernel(GaussianProcessRegressor(kernel=None), train_y_sub, train_x_2D)

    # Fit the model
    print('Fitting model')
    model = Model()
    #model.fitting_model(train_y,train_x_2D)
    model.fitting_model(train_y_n ,train_x_2D)
    
    #print('Cross validating')
    #cv_results = cross_validation(train_y, train_x_2D, train_x_AREA)
    #cv_results = cross_validation(train_y_n, train_x_2D, train_x_AREA)
    #print('Cross validation result: ', cv_results)

    # Predict on the test features
    print('Predicting on test features')
    predictions = model.make_predictions(test_x_2D, test_x_AREA)
    print(predictions)

    cost = cost_function(train_y_v, predictions[0], test_x_AREA)
    print(cost)

    # A METHOD TO EVALUATE THE MODEL?

    if EXTENDED_EVALUATION:
        perform_extended_evaluation(model, output_dir='.')


if __name__ == "__main__":
    main()

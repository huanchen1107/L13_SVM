import numpy as np
from sklearn.svm import SVC


def train_svm(X, y, kernel="rbf", C=10.0, gamma=1.0, degree=3):
    """Train an sklearn SVC model on the given data.

    Returns
    -------
    sklearn.svm.SVC
        Trained model with decision_function available.
    """
    model = SVC(kernel=kernel, C=C, gamma=gamma, degree=degree)
    model.fit(X, y)
    return model


def make_decision_grid(x_range, y_range, resolution=80):
    """Build a 2D mesh grid for evaluating the decision function.

    Returns
    -------
    xx : np.ndarray
        2D array of x coordinates.
    yy : np.ndarray
        2D array of y coordinates.
    grid_points : np.ndarray of shape (resolution * resolution, 2)
        Flattened (x, y) pairs for model.decision_function.
    """
    x_vals = np.linspace(x_range[0], x_range[1], resolution)
    y_vals = np.linspace(y_range[0], y_range[1], resolution)
    xx, yy = np.meshgrid(x_vals, y_vals)
    grid_points = np.column_stack([xx.ravel(), yy.ravel()])
    return xx, yy, grid_points


def compute_decision_surface(model, grid_points, xx_shape):
    """Evaluate decision_function on grid and reshape to 2D.

    Returns
    -------
    Z : np.ndarray
        Decision function values reshaped to match xx / yy shape.
    """
    Z = model.decision_function(grid_points)
    return Z.reshape(xx_shape)

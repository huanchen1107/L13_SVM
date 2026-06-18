import numpy as np


def generate_ring_dataset(n_inner=35, n_outer=45, inner_radius_range=(0.0, 1.0),
                          outer_radius_range=(1.6, 2.5), noise=0.08, random_seed=7):
    """Generate a 2D ring dataset: blue points near origin, red points in outer ring.

    Parameters
    ----------
    n_inner : int
        Number of inner (blue) points.
    n_outer : int
        Number of outer (red) points.
    inner_radius_range : tuple[float, float]
        (min, max) radius for inner points.
    outer_radius_range : tuple[float, float]
        (min, max) radius for outer points.
    noise : float
        Gaussian noise std added to x and y coordinates.
    random_seed : int
        Seed for reproducibility.

    Returns
    -------
    X : np.ndarray of shape (n_inner + n_outer, 2)
        Feature matrix (x, y coordinates).
    y : np.ndarray of shape (n_inner + n_outer,)
        Labels: 0 for inner (blue), 1 for outer (red).
    """
    rng = np.random.default_rng(random_seed)

    # Inner: random angles, radius uniform in inner_radius_range
    angles_inner = rng.uniform(0, 2 * np.pi, n_inner)
    radii_inner = rng.uniform(*inner_radius_range, n_inner)
    X_inner = np.column_stack([radii_inner * np.cos(angles_inner),
                                radii_inner * np.sin(angles_inner)])

    # Outer: random angles, radius uniform in outer_radius_range
    angles_outer = rng.uniform(0, 2 * np.pi, n_outer)
    radii_outer = rng.uniform(*outer_radius_range, n_outer)
    X_outer = np.column_stack([radii_outer * np.cos(angles_outer),
                                radii_outer * np.sin(angles_outer)])

    X = np.vstack([X_inner, X_outer])
    y = np.hstack([np.zeros(n_inner, dtype=int), np.ones(n_outer, dtype=int)])

    if noise > 0:
        X += rng.normal(0, noise, X.shape)

    return X, y

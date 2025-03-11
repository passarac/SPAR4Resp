import numpy as np
import scipy.spatial
import scipy.stats
import scipy.linalg
import nolds

def box_counting_dimension(ak: np.ndarray, bk: np.ndarray, epsilons=np.logspace(-2, 0, 10)) -> float:
    """
    Computes the Box-Counting Fractal Dimension of the SPAR attractor projection.

    Parameters:
    -----------
    ak : np.ndarray
        1D array representing the `a_k` coordinate from the SPAR projection.
    bk : np.ndarray
        1D array representing the `b_k` coordinate from the SPAR projection.
    epsilons : np.ndarray, optional
        Log-spaced range of box sizes to test (default is `np.logspace(-2, 0, 10)`).

    Returns:
    --------
    float
        Estimated Box-Counting Fractal Dimension, indicating the complexity of the attractor's geometry.
    """
    spar_data = np.vstack((ak, bk)).T  # Convert to 2D array
    counts = []

    for eps in epsilons:
        grid, _ = np.histogramdd(spar_data, bins=int(1.0 / eps))
        counts.append(np.count_nonzero(grid))

    coeffs = np.polyfit(np.log(epsilons), np.log(counts), 1)
    return -coeffs[0]



def correlation_dimension(ak: np.ndarray, bk: np.ndarray) -> float:
    """
    Computes the Correlation Dimension (D2) of the SPAR attractor projection.

    Parameters:
    -----------
    ak : np.ndarray
        1D array representing the `a_k` coordinate from the SPAR projection.
    bk : np.ndarray
        1D array representing the `b_k` coordinate from the SPAR projection.

    Returns:
    --------
    float
        Estimated Correlation Dimension (D2), which quantifies how densely points cluster in the attractor.
    """
    spar_data = np.vstack((ak, bk)).T
    return nolds.corr_dim(spar_data, 2)




def attractor_volume(ak: np.ndarray, bk: np.ndarray) -> float:
    """
    Computes the convex hull volume occupied by the SPAR attractor projection.

    Parameters:
    -----------
    ak : np.ndarray
        1D array representing the `a_k` coordinate from the SPAR projection.
    bk : np.ndarray
        1D array representing the `b_k` coordinate from the SPAR projection.

    Returns:
    --------
    float
        Volume of the convex hull surrounding the attractor.
    """
    spar_data = np.vstack((ak, bk)).T
    hull = scipy.spatial.ConvexHull(spar_data)
    return hull.volume
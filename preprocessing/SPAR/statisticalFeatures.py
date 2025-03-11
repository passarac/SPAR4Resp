import numpy as np
import scipy.stats
import scipy.spatial
from sklearn.decomposition import PCA


def mean_distance(ak: np.ndarray, bk: np.ndarray) -> float:
    """
    Computes the mean Euclidean distance between points in the SPAR attractor projection.

    Parameters:
    -----------
    ak : np.ndarray
        1D NumPy array representing the `a_k` coordinate from the SPAR projection.
    bk : np.ndarray
        1D NumPy array representing the `b_k` coordinate from the SPAR projection.

    Returns:
    --------
    float
        Mean Euclidean distance between all pairs of points in the attractor.
    """
    spar_data = np.vstack((ak, bk)).T  # Convert to 2D array
    distances = scipy.spatial.distance.pdist(spar_data, metric='euclidean')
    return np.mean(distances)




def std_distance(ak: np.ndarray, bk: np.ndarray) -> float:
    """
    Computes the standard deviation of Euclidean distances between points in the SPAR attractor.

    Parameters:
    -----------
    ak : np.ndarray
        1D NumPy array representing the `a_k` coordinate.
    bk : np.ndarray
        1D NumPy array representing the `b_k` coordinate.

    Returns:
    --------
    float
        Standard deviation of the Euclidean distances.
    """
    spar_data = np.vstack((ak, bk)).T
    distances = scipy.spatial.distance.pdist(spar_data, metric='euclidean')
    return np.std(distances)




def skewness_distance(ak: np.ndarray, bk: np.ndarray) -> float:
    """
    Computes the skewness of the Euclidean distance distribution in the SPAR attractor.

    Parameters:
    -----------
    ak : np.ndarray
        1D NumPy array representing the `a_k` coordinate.
    bk : np.ndarray
        1D NumPy array representing the `b_k` coordinate.

    Returns:
    --------
    float
        Skewness of the Euclidean distance distribution, indicating asymmetry.
    """
    spar_data = np.vstack((ak, bk)).T
    distances = scipy.spatial.distance.pdist(spar_data, metric='euclidean')
    return scipy.stats.skew(distances)




def kurtosis_distance(ak: np.ndarray, bk: np.ndarray) -> float:
    """
    Computes the kurtosis of the Euclidean distance distribution in the SPAR attractor.

    Parameters:
    -----------
    ak : np.ndarray
        1D NumPy array representing the `a_k` coordinate.
    bk : np.ndarray
        1D NumPy array representing the `b_k` coordinate.

    Returns:
    --------
    float
        Kurtosis of the Euclidean distance distribution. Higher values indicate the presence of extreme distances.
    """
    spar_data = np.vstack((ak, bk)).T
    distances = scipy.spatial.distance.pdist(spar_data, metric='euclidean')
    return scipy.stats.kurtosis(distances)




def principal_component_variance(ak: np.ndarray, bk: np.ndarray) -> float:
    """
    Computes the variance explained by the first principal component in the SPAR attractor projection.

    Parameters:
    -----------
    ak : np.ndarray
        1D NumPy array representing the `a_k` coordinate.
    bk : np.ndarray
        1D NumPy array representing the `b_k` coordinate.

    Returns:
    --------
    float
        Variance explained by the first principal component, indicating the dominant axis of variation.
    """
    spar_data = np.vstack((ak, bk)).T
    pca = PCA(n_components=1)
    pca.fit(spar_data)
    return pca.explained_variance_ratio_[0]
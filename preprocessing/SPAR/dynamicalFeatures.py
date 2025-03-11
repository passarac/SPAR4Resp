import numpy as np
import scipy.stats
import nolds
from pyrqa import *


def recurrence_rate(ak: np.ndarray, bk: np.ndarray, threshold: float = 0.1) -> float:
    """
    Computes the Recurrence Rate (RR) from the SPAR attractor projection.

    Parameters:
    -----------
    ak : np.ndarray
        1D array representing the a_k coordinate of the SPAR projection.
    bk : np.ndarray
        1D array representing the b_k coordinate of the SPAR projection.
    threshold : float, optional
        Similarity threshold for recurrence detection (default is 0.1).

    Returns:
    --------
    float
        Recurrence Rate (RR), which quantifies how frequently the attractor revisits previous states.
    """
    spar_data = np.vstack((ak, bk)).T  # Convert to 2D
    rp = RecurrencePlot(RQAParameters(spar_data, time_delay=1, embedding_dimension=2, similarity_threshold=threshold))
    rqa = RQA(rp)
    return rqa.recurrence_rate




def determinism(ak: np.ndarray, bk: np.ndarray, time_delay: int, threshold: float = 0.1) -> float:
    """
    Computes Determinism (DET) from the SPAR attractor projection.

    Parameters:
    -----------
    ak : np.ndarray
        1D array representing the a_k coordinate of the SPAR projection.
    bk : np.ndarray
        1D array representing the b_k coordinate of the SPAR projection.
    threshold : float, optional
        Similarity threshold for recurrence detection (default is 0.1).

    Returns:
    --------
    float
        Determinism (DET), which quantifies the proportion of recurrence forming diagonal structures, indicating predictability.
    """
    spar_data = np.vstack((ak, bk)).T
    rp = RecurrencePlot(RQAParameters(spar_data, time_delay=time_delay, embedding_dimension=2, similarity_threshold=threshold))
    rqa = RQA(rp)
    return rqa.determinism




def trapping_time(ak: np.ndarray, bk: np.ndarray, time_delay: int, threshold: float = 0.1) -> float:
    """
    Computes the Trapping Time (TT) from the SPAR attractor projection.

    Parameters:
    -----------
    ak : np.ndarray
        1D array representing the a_k coordinate of the SPAR projection.
    bk : np.ndarray
        1D array representing the b_k coordinate of the SPAR projection.
    threshold : float, optional
        Similarity threshold for recurrence detection (default is 0.1).

    Returns:
    --------
    float
        Trapping Time (TT), which indicates the average time the system remains in a recurrent state before changing.
    """
    spar_data = np.vstack((ak, bk)).T
    rp = RecurrencePlot(RQAParameters(spar_data, time_delay=time_delay, embedding_dimension=2, similarity_threshold=threshold))
    rqa = RQA(rp)
    return rqa.trapping_time




def lyapunov_exponent(ak: np.ndarray, bk: np.ndarray) -> float:
    """
    Computes the Largest Lyapunov Exponent (LLE) from the SPAR attractor projection.

    Parameters:
    -----------
    ak : np.ndarray
        1D array representing the a_k coordinate of the SPAR projection.
    bk : np.ndarray
        1D array representing the b_k coordinate of the SPAR projection.

    Returns:
    --------
    float
        Largest Lyapunov Exponent (LLE), which measures the system's sensitivity to initial conditions.
    """
    spar_data = np.vstack((ak, bk)).T
    return nolds.lyap_r(spar_data)




def shannon_entropy(ak: np.ndarray, bk: np.ndarray, bins: int = 20) -> float:
    """
    Computes Shannon Entropy from the SPAR attractor projection.

    Parameters:
    -----------
    ak : np.ndarray
        1D array representing the a_k coordinate of the SPAR projection.
    bk : np.ndarray
        1D array representing the b_k coordinate of the SPAR projection.
    bins : int, optional
        Number of bins for histogram estimation (default is 20).

    Returns:
    --------
    float
        Shannon Entropy (SE), which measures the unpredictability of the attractor distribution.
    """
    spar_data = np.vstack((ak, bk)).T
    hist, _ = np.histogramdd(spar_data, bins=bins, density=True)
    return -np.sum(hist * np.log(hist + 1e-10))  # Avoid log(0)




def permutation_entropy(ak: np.ndarray, bk: np.ndarray, order: int = 3) -> float:
    """
    Computes Permutation Entropy from the SPAR attractor projection.

    Parameters:
    -----------
    ak : np.ndarray
        1D array representing the a_k coordinate of the SPAR projection.
    bk : np.ndarray
        1D array representing the b_k coordinate of the SPAR projection.
    order : int, optional
        Order of the permutation entropy (default is 3).

    Returns:
    --------
    float
        Permutation Entropy (PE), which quantifies randomness in the trajectory.
    """
    spar_data = np.vstack((ak, bk)).T
    return nolds.perm_entropy(spar_data, order=order)

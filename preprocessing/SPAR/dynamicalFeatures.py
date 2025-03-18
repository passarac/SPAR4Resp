import numpy as np
import scipy.stats
import nolds
'''
from pyrqa.computation import RQA
from pyrqa.time_series import TimeSeries
from pyrqa.settings import RQAParameters
from pyrqa.analysis_type import Classic
from pyrqa.metric import EuclideanMetric
from pyrqa.recurrence_plot import RecurrencePlot'''


def recurrence_rate(ak: np.ndarray, bk: np.ndarray, threshold: float = 0.1) -> float:
    """
    Computes the Recurrence Rate (RR) from the SPAR attractor projection without pyrqa.

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
    # Stack the coordinates to form a 2D trajectory
    spar_data = np.vstack((ak, bk)).T  # Shape: (N, 2)

    # Compute pairwise Euclidean distances
    N = spar_data.shape[0]
    distance_matrix = np.sqrt(np.sum((spar_data[:, np.newaxis, :] - spar_data[np.newaxis, :, :])**2, axis=2))  # Shape (N, N)

    # Construct recurrence matrix: 1 if distance < threshold, 0 otherwise
    recurrence_matrix = (distance_matrix < threshold).astype(int)

    # Compute recurrence rate: ratio of recurrent points to total points
    recurrence_rate = np.sum(recurrence_matrix) / (N * N)  # Normalize by total elements

    return recurrence_rate




def determinism(ak: np.ndarray, bk: np.ndarray, time_delay: int,
                threshold: float = 0.1, min_diag_length: int = 2) -> float:
    """
    Computes Determinism (DET) from the SPAR attractor projection without pyrqa.

    Parameters:
    -----------
    ak : np.ndarray
        1D array representing the a_k coordinate of the SPAR projection.
    bk : np.ndarray
        1D array representing the b_k coordinate of the SPAR projection.
    time_delay : int
        Time delay used for recurrence calculations.
    threshold : float, optional
        Similarity threshold for recurrence detection (default is 0.1).
    min_diag_length : int, optional
        Minimum length of diagonal lines to be considered in DET calculation (default is 2).

    Returns:
    --------
    float
        Determinism (DET), which quantifies the proportion of recurrence forming diagonal structures.
    """
    # Stack the coordinates to form a 2D trajectory
    spar_data = np.vstack((ak, bk)).T  # Shape: (N, 2)

    # Compute pairwise Euclidean distances
    N = spar_data.shape[0]
    distance_matrix = np.sqrt(np.sum((spar_data[:, np.newaxis, :] - spar_data[np.newaxis, :, :])**2, axis=2))  # Shape (N, N)

    # Construct recurrence matrix: 1 if distance < threshold, 0 otherwise
    recurrence_matrix = (distance_matrix < threshold).astype(int)

    # Identify diagonal structures (lines of 1s)
    diag_lengths = []
    for diag in range(-N+1, N):  # Loop over diagonals
        line = np.diag(recurrence_matrix, k=diag)  # Extract diagonal
        line_lengths = np.diff(np.where(np.concatenate(([0], line, [0])) == 0)[0]) - 1  # Compute consecutive 1s
        diag_lengths.extend(line_lengths[line_lengths >= min_diag_length])  # Filter lines >= min length

    # Total number of recurrence points in diagonals
    recurrent_diag_points = np.sum(diag_lengths)
    # Total number of recurrence points in the entire matrix
    total_recurrent_points = np.sum(recurrence_matrix)

    # Compute DET: ratio of diagonal recurrence points to total recurrence points
    determinism = recurrent_diag_points / total_recurrent_points if total_recurrent_points > 0 else 0.0

    return determinism




def trapping_time(ak: np.ndarray, bk: np.ndarray, time_delay: int, threshold: float = 0.1, min_vert_length: int = 2) -> float:
    """
    Computes Trapping Time (TT) from the SPAR attractor projection without pyrqa.

    Parameters:
    -----------
    ak : np.ndarray
        1D array representing the a_k coordinate of the SPAR projection.
    bk : np.ndarray
        1D array representing the b_k coordinate of the SPAR projection.
    time_delay : int
        Time delay used for recurrence calculations.
    threshold : float, optional
        Similarity threshold for recurrence detection (default is 0.1).
    min_vert_length : int, optional
        Minimum length of vertical lines to be considered in TT calculation (default is 2).

    Returns:
    --------
    float
        Trapping Time (TT), which indicates how long the system remains in a recurrent state.
    """
    # Stack the coordinates to form a 2D trajectory
    spar_data = np.vstack((ak, bk)).T  # Shape: (N, 2)

    # Compute pairwise Euclidean distances
    N = spar_data.shape[0]
    distance_matrix = np.sqrt(np.sum((spar_data[:, np.newaxis, :] - spar_data[np.newaxis, :, :])**2, axis=2))  # Shape (N, N)

    # Construct recurrence matrix: 1 if distance < threshold, 0 otherwise
    recurrence_matrix = (distance_matrix < threshold).astype(int)

    # Identify vertical structures (sequences of 1s in columns)
    vert_lengths = []
    for col in range(N):  # Iterate over columns
        line = recurrence_matrix[:, col]  # Extract column
        line_lengths = np.diff(np.where(np.concatenate(([0], line, [0])) == 0)[0]) - 1  # Find vertical sequences
        vert_lengths.extend(line_lengths[line_lengths >= min_vert_length])  # Filter vertical lines

    # Compute TT: average length of vertical lines
    trapping_time = np.mean(vert_lengths) if len(vert_lengths) > 0 else 0.0

    return trapping_time



# HAS ISSUES
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



# HAS ISSUES
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

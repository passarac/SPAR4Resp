import numpy as np
from scipy.spatial import ConvexHull
from sklearn.decomposition import PCA
import scipy.signal as signal
import networkx as nx
from scipy.spatial.distance import pdist, squareform
from scipy.stats import skew, kurtosis
from pyunicorn.timeseries import RecurrencePlot
from scipy.stats import linregress

from misc.utils import *



def compute_statistical_features(a_k, b_k):
    """
    Computes basic statistical features of the SPAR projection.

    Parameters:
    - a_k (np.ndarray): SPAR projection in the first dimension.
    - b_k (np.ndarray): SPAR projection in the second dimension.

    Returns:
    - dict: A dictionary containing:
        - "variance_a": Variance of a_k (spread of data along this axis).
        - "variance_b": Variance of b_k.
        - "skewness_a": Skewness of a_k (asymmetry in the distribution).
        - "skewness_b": Skewness of b_k.
        - "kurtosis_a": Kurtosis of a_k (tailedness of the distribution).
        - "kurtosis_b": Kurtosis of b_k.
    """
    return {
        "variance_a": np.var(a_k),
        "variance_b": np.var(b_k),
        "skewness_a": skew(a_k),
        "skewness_b": skew(b_k),
        "kurtosis_a": kurtosis(a_k),
        "kurtosis_b": kurtosis(b_k),
    }



def compute_geometric_features(a_k, b_k):
    """
    Computes geometric properties of the SPAR attractor.

    Parameters:
    - a_k (np.ndarray): SPAR projection in the first dimension.
    - b_k (np.ndarray): SPAR projection in the second dimension.

    Returns:
    - dict: A dictionary containing:
        - "convex_hull_area": The area of the convex hull, representing the attractor's spread.
        - "aspect_ratio": The ratio of max range of a_k to max range of b_k.
        - "radius_of_gyration": The root mean square distance from centroid.
        - "attractor_compactness": The standard deviation to mean distance ratio from centroid.
    """
    points = np.column_stack((a_k, b_k))
    hull = ConvexHull(points)
    centroid = np.mean(points, axis=0)
    distances = np.linalg.norm(points - centroid, axis=1)

    return {
        "convex_hull_area": hull.volume,
        "aspect_ratio": np.ptp(a_k) / np.ptp(b_k),
        "radius_of_gyration": np.sqrt(np.mean(distances**2)),
        "attractor_compactness": np.std(distances) / np.mean(distances)
    }


def box_counting_dimension(a_k, b_k, num_scales=10):
    """
    Computes the box-counting dimension of a 2D attractor.

    Parameters:
    - a_k (np.ndarray): SPAR projection in the first dimension.
    - b_k (np.ndarray): SPAR projection in the second dimension.
    - num_scales (int): Number of different box sizes to consider.

    Returns:
    - float: Estimated box-counting dimension.
    """
    # Convert points to 2D array
    points = np.column_stack((a_k, b_k))
    
    # Determine the range of the attractor
    min_x, max_x = np.min(a_k), np.max(a_k)
    min_y, max_y = np.min(b_k), np.max(b_k)
    
    # Generate box sizes (logarithmically spaced)
    min_size = min(max_x - min_x, max_y - min_y) / 2
    max_size = min_size / 100  # Smallest box size
    box_sizes = np.logspace(np.log10(min_size), np.log10(max_size), num_scales)
    
    # Count number of occupied boxes for each size
    N = []
    for box_size in box_sizes:
        # Create a grid
        x_bins = np.arange(min_x, max_x, box_size)
        y_bins = np.arange(min_y, max_y, box_size)
        
        # Assign each point to a grid cell
        grid = set()
        for x, y in points:
            x_idx = np.digitize(x, x_bins)
            y_idx = np.digitize(y, y_bins)
            grid.add((x_idx, y_idx))
        
        # Count occupied boxes
        N.append(len(grid))
    
    # Perform linear regression on log-log plot
    log_N = np.log(N)
    log_box_sizes = np.log(1 / box_sizes)
    
    coeffs = np.polyfit(log_box_sizes, log_N, 1)
    fractal_dim = coeffs[0]  # Slope of the log-log plot
    
    return {
        "box_counting_dim": fractal_dim
    }



def compute_rqa_features(a_k, b_k, threshold=0.1):
    """
    Computes Recurrence Quantification Analysis (RQA) features of the attractor.

    Parameters:
    - a_k (np.ndarray): SPAR projection in the first dimension.
    - b_k (np.ndarray): SPAR projection in the second dimension.
    - threshold (float): Recurrence threshold for defining recurrence points.

    Returns:
    - dict: A dictionary containing:
        - "recurrence_rate": Density of recurrent points in the attractor.
        - "determinism": Ratio of recurrence points forming diagonal lines.
        - "laminarity": Measure of the proportion of vertical structures.
        - "trapping_time": Average length of vertical lines.
    """
    points = np.column_stack((a_k, b_k))
    rp = RecurrencePlot(points, threshold=threshold)

    return {
        "recurrence_rate": rp.recurrence_rate(),
        "determinism": rp.determinism(),
        "laminarity": rp.laminarity(),
        "trapping_time": rp.trapping_time()
    }



def compute_entropy_features(a_k, b_k, bins=20):
    """
    Computes entropy-based features using 2D histogram.

    Parameters:
    - a_k (np.ndarray): SPAR projection in the first dimension.
    - b_k (np.ndarray): SPAR projection in the second dimension.
    - bins (int): Number of histogram bins.

    Returns:
    - dict: A dictionary containing:
        - "joint_entropy": Entropy of the 2D histogram.
    """
    hist_2d, _, _ = np.histogram2d(a_k, b_k, bins=bins, density=True)
    hist_2d = hist_2d / np.sum(hist_2d)
    hist_2d = hist_2d[hist_2d > 0]
    joint_entropy = -np.sum(hist_2d * np.log(hist_2d))

    return {"joint_entropy": joint_entropy}



def compute_lyapunov_exponent(a_k, b_k, dt=1.0, theiler_window=10, max_time_steps=None):
    """
    Estimate the largest Lyapunov exponent from 2D state-space projections.

    Parameters:
    -----------
    a_k : np.ndarray
        SPAR projection in the first dimension.
    b_k : np.ndarray
        SPAR projection in the second dimension.
    dt : float, optional
        Sampling time interval (default is 1.0).
    theiler_window : int, optional
        Minimum time separation between pairs to avoid temporal correlations (default is 10).
    max_time_steps : int, optional
        Maximum number of time steps over which to track divergence.
        If None, it defaults to half the length of the trajectory.

    Returns:
    --------
    lambda_max : float
        Estimated largest Lyapunov exponent.
    time_fit : np.ndarray
        Time vector over which the linear fit was performed.
    divergence_fit : np.ndarray
        Average logarithmic divergence used for the linear fit.
    """
    # Combine the two projections into a 2D trajectory:
    X = np.column_stack((a_k, b_k))
    N = X.shape[0]

    # Set max_time_steps if not provided.
    if max_time_steps is None:
        max_time_steps = N // 2

    # Initialize arrays to accumulate log distances and counts
    L = np.zeros(max_time_steps)  # sum of log separations at each evolution step
    counts = np.zeros(max_time_steps)  # number of pairs contributing at each step

    # For each point, find its nearest neighbor with time separation > theiler_window.
    for i in range(N - max_time_steps):
        # Compute distances to all other points with index difference > theiler_window
        # We'll exclude indices too close in time.
        valid_indices = np.concatenate((np.arange(0, i - theiler_window + 1),
                                        np.arange(i + theiler_window, N)))
        if valid_indices.size == 0:
            continue  # No valid neighbor found
        # Compute Euclidean distances from point i to all valid candidates:
        distances = np.linalg.norm(X[valid_indices] - X[i], axis=1)
        j_index = valid_indices[np.argmin(distances)]

        # For each time step k, track the divergence if both trajectories remain in bounds.
        for k in range(max_time_steps):
            if i + k < N and j_index + k < N:
                dist = np.linalg.norm(X[i + k] - X[j_index + k])
                # To avoid log(0) issues:
                if dist > 0:
                    L[k] += np.log(dist)
                    counts[k] += 1

    # Compute the average log divergence at each time step.
    with np.errstate(divide='ignore', invalid='ignore'):
        L_avg = np.where(counts > 0, L / counts, 0)

    # Choose a fitting region (typically the early part of the divergence curve)
    # Here, we choose up to half of the max_time_steps.
    k_fit = np.arange(max_time_steps // 2)
    t_fit = k_fit * dt
    divergence_fit = L_avg[k_fit]

    # Perform linear regression on the fitting region.
    # The slope is an estimate of the largest Lyapunov exponent.
    slope, intercept, r_value, p_value, std_err = linregress(t_fit, divergence_fit)

    lambda_max = slope

    return {
        "lle": lambda_max
    }


def compute_spectral_features(a_k, b_k, fs=1):
    """
    Computes frequency domain features from power spectral density.

    Parameters:
    - a_k (np.ndarray): SPAR projection in the first dimension.
    - b_k (np.ndarray): SPAR projection in the second dimension.
    - fs (int): Sampling frequency.

    Returns:
    - dict: A dictionary containing:
        - "psd_mean_a": Mean power spectral density of a_k.
        - "psd_mean_b": Mean power spectral density of b_k.
        - "psd_peak_a": Frequency of peak power in a_k.
        - "psd_peak_b": Frequency of peak power in b_k.
        - "peak_to_peak_a": Peak-to-peak amplitude of a_k.
        - "peak_to_peak_b": Peak-to-peak amplitude of b_k.
    """
    freqs_a, psd_a = signal.welch(a_k, fs=fs)
    freqs_b, psd_b = signal.welch(b_k, fs=fs)

    return {
        "psd_mean_a": np.mean(psd_a),
        "psd_mean_b": np.mean(psd_b),
        "psd_peak_a": freqs_a[np.argmax(psd_a)],
        "psd_peak_b": freqs_b[np.argmax(psd_b)],
        "peak_to_peak_a": np.ptp(a_k),
        "peak_to_peak_b": np.ptp(b_k)
    }


def compute_graph_entropy(a_k, b_k, k=5):
    """
    Computes graph entropy based on a k-nearest neighbor graph.

    Parameters:
    - a_k (np.ndarray): SPAR projection in the first dimension.
    - b_k (np.ndarray): SPAR projection in the second dimension.
    - k (int): Number of nearest neighbors.

    Returns:
    - dict: A dictionary containing:
        - "graph_entropy": Shannon entropy of the degree distribution.
    """
    points = np.column_stack((a_k, b_k))
    G = nx.Graph()

    for i, p1 in enumerate(points):
        distances = np.linalg.norm(points - p1, axis=1)
        nearest_neighbors = np.argsort(distances)[1:k+1]
        for j in nearest_neighbors:
            G.add_edge(i, j)

    degrees = np.array([d for _, d in G.degree()])
    prob = degrees / np.sum(degrees)
    return {"graph_entropy": -np.sum(prob * np.log(prob))}




### Extract All Features
def extract_all_features(a_k, b_k):
    """
    Computes all extracted features from SPAR projections.

    Parameters:
    - a_k (np.ndarray): SPAR projection in the first dimension.
    - b_k (np.ndarray): SPAR projection in the second dimension.

    Returns:
    - dict: A dictionary containing all computed features.
    """
    features = {}
    features.update(compute_statistical_features(a_k, b_k))
    features.update(compute_geometric_features(a_k, b_k))

    with suppress_output():
        features.update(compute_rqa_features(a_k, b_k))

    features.update(compute_entropy_features(a_k, b_k))
    features.update(compute_lyapunov_exponent(a_k, b_k))
    features.update(compute_spectral_features(a_k, b_k))
    features.update(compute_graph_entropy(a_k, b_k))
    features.update(box_counting_dimension(a_k, b_k))
    return features

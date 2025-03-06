import numpy as np
from sklearn.decomposition import PCA
from scipy.fftpack import fft
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.stats import entropy, gaussian_kde
from antropy import *
from scipy.spatial import ConvexHull


# ------------------------------
# 1. GEOMETRIC FEATURES
# ------------------------------
def max_radius(a_k, b_k):
    """Computes the maximal radius of the SPAR attractor."""
    return np.max(np.sqrt(a_k**2 + b_k**2))

def mean_radius(a_k, b_k):
    """Computes the mean radius of the SPAR attractor."""
    return np.mean(np.sqrt(a_k**2 + b_k**2))

def convex_hull_area(a_k, b_k):
    """Computes the convex hull area of the attractor."""
    points = np.vstack((a_k, b_k)).T
    hull = ConvexHull(points)
    return hull.area

def circularity(a_k, b_k):
    """Computes the circularity (eccentricity) of the SPAR attractor using PCA."""
    pca = PCA(n_components=2)
    data = np.vstack((a_k, b_k)).T
    pca.fit(data)
    lambda1, lambda2 = pca.explained_variance_
    return np.sqrt(1 - lambda2 / lambda1)

def aspect_ratio(a_k, b_k):
    """Computes the aspect ratio of the attractor (height-to-width)."""
    return (np.max(b_k) - np.min(b_k)) / (np.max(a_k) - np.min(a_k))

# ------------------------------
# 2. VARIABILITY & STABILITY FEATURES
# ------------------------------
def dispersion_index(a_k, b_k):
    """Computes the dispersion index (variance-to-mean ratio of radius)."""
    r = np.sqrt(a_k**2 + b_k**2)
    return np.var(r) / np.mean(r)

def poincare_sd1(a_k, b_k):
    """Computes short-term variability (SD1) using Poincaré analysis."""
    r = np.sqrt(a_k**2 + b_k**2)
    r_diff = np.diff(r)
    return np.std(r_diff) / np.sqrt(2)

def poincare_sd2(a_k, b_k):
    """Computes long-term variability (SD2) using Poincaré analysis."""
    r = np.sqrt(a_k**2 + b_k**2)
    return np.std(r) * np.sqrt(2)

def recurrence_rate(a_k, b_k, threshold=0.1):
    """Computes the recurrence rate of the attractor."""
    data = np.vstack((a_k, b_k)).T
    distance_matrix = squareform(pdist(data, metric='euclidean'))
    recurrence_matrix = (distance_matrix < threshold).astype(int)
    return np.sum(recurrence_matrix) / (len(a_k) ** 2)

# ------------------------------
# 3. FREQUENCY & SPECTRAL FEATURES
# ------------------------------
def dominant_frequency(a_k, b_k):
    """Finds the dominant frequency in the attractor."""
    r = np.sqrt(a_k**2 + b_k**2)
    radius_fft = np.abs(fft(r))[:len(r) // 2]
    return np.argmax(radius_fft)

def spectral_entropy(a_k, b_k):
    """Computes spectral entropy of the attractor."""
    r = np.sqrt(a_k**2 + b_k**2)
    radius_fft = np.abs(fft(r))[:len(r) // 2]
    radius_fft /= np.sum(radius_fft) + 1e-10
    return -np.sum(radius_fft * np.log2(radius_fft + 1e-10))

def spectral_centroid(a_k, b_k):
    """Computes spectral centroid (weighted mean frequency)."""
    r = np.sqrt(a_k**2 + b_k**2)
    fft_freqs = np.fft.fftfreq(len(r))[:len(r) // 2]
    radius_fft = np.abs(fft(r))[:len(r) // 2]
    return np.sum(fft_freqs * radius_fft) / np.sum(radius_fft)

def spectral_spread(a_k, b_k):
    """Computes spectral spread (variance of frequency distribution)."""
    centroid = spectral_centroid(a_k, b_k)
    r = np.sqrt(a_k**2 + b_k**2)
    fft_freqs = np.fft.fftfreq(len(r))[:len(r) // 2]
    radius_fft = np.abs(fft(r))[:len(r) // 2]
    return np.sqrt(np.sum(((fft_freqs - centroid) ** 2) * radius_fft) / np.sum(radius_fft))

# ------------------------------
# 4. COMPLEXITY & NONLINEAR FEATURES
# ------------------------------
def fractal_dimension(a_k, b_k):
    """Computes fractal dimension of the attractor using the box-counting method."""
    data = np.vstack((a_k, b_k)).T
    epsilon_range = np.logspace(-2, 0, 10)
    counts = [np.sum(np.histogram2d(data[:, 0], data[:, 1], bins=int(1/eps))[0] > 0) for eps in epsilon_range]
    coeffs = np.polyfit(np.log(epsilon_range), np.log(counts), 1)
    return -coeffs[0]

def sample_entropy(a_k, b_k):
    """Computes sample entropy to measure attractor predictability."""
    r = np.sqrt(a_k**2 + b_k**2)
    return sampen(r)

def shannon_entropy(a_k, b_k, num_bins=30):
    """Computes Shannon entropy of the attractor."""
    hist_2d, _, _ = np.histogram2d(a_k, b_k, bins=num_bins)
    prob_dist = hist_2d.flatten() / np.sum(hist_2d)
    return entropy(prob_dist)

# ------------------------------
# 5. GAPS, DENSITY, & STABILITY
# ------------------------------
def lacunarity(a_k, b_k):
    """Computes lacunarity (presence of gaps in attractor structure)."""
    data = np.vstack((a_k, b_k)).T
    box_sizes = np.logspace(-2, 0, 10)
    lac_values = []
    for eps in box_sizes:
        grid = np.histogram2d(data[:, 0], data[:, 1], bins=int(1 / eps))[0]
        mean = np.mean(grid)
        var = np.var(grid)
        lac_values.append(var / (mean**2) if mean > 0 else 0)
    return np.mean(lac_values)

def density_variability(a_k, b_k):
    """Computes density variability using Kernel Density Estimation (KDE)."""
    xy = np.vstack([a_k, b_k])
    density = gaussian_kde(xy)(xy)
    return np.var(density)

# ------------------------------
# 6. TEMPORAL EVOLUTION FEATURES
# ------------------------------
def drift_over_time(a_k, b_k):
    """Computes the drift of the attractor over time."""
    drift_x = np.std(np.diff(a_k))
    drift_y = np.std(np.diff(b_k))
    return np.sqrt(drift_x**2 + drift_y**2)

def phase_transition_frequency(a_k, b_k):
    """Counts abrupt transitions in attractor states."""
    r = np.sqrt(a_k**2 + b_k**2)
    transitions = np.abs(np.diff(r))
    return np.sum(transitions > np.mean(transitions) + 2 * np.std(transitions))

# ------------------------------
# Combine Function
# ------------------------------


# Compute Features

def compute_features(a_k, b_k):
    """Computes all SPAR features and returns a dictionary."""
    features = {
        # Geometric Features
        "Max Radius": max_radius(a_k, b_k),
        "Mean Radius": mean_radius(a_k, b_k),
        "Convex Hull Area": convex_hull_area(a_k, b_k),
        "Circularity": circularity(a_k, b_k),
        "Aspect Ratio": aspect_ratio(a_k, b_k),

        # Variability & Stability Features
        "Dispersion Index": dispersion_index(a_k, b_k),
        "Poincaré SD1": poincare_sd1(a_k, b_k),
        "Poincaré SD2": poincare_sd2(a_k, b_k),
        "Recurrence Rate": recurrence_rate(a_k, b_k),

        # Frequency & Spectral Features
        "Dominant Frequency": dominant_frequency(a_k, b_k),
        "Spectral Entropy": spectral_entropy(a_k, b_k),
        "Spectral Centroid": spectral_centroid(a_k, b_k),
        "Spectral Spread": spectral_spread(a_k, b_k),

        # Complexity & Nonlinear Features
        "Fractal Dimension": fractal_dimension(a_k, b_k),
        "Sample Entropy": sample_entropy(a_k, b_k),
        "Shannon Entropy": shannon_entropy(a_k, b_k),

        # Gaps, Density & Stability Features
        "Lacunarity": lacunarity(a_k, b_k),
        "Density Variability": density_variability(a_k, b_k),

        # Temporal Evolution Features
        "Attractor Drift": drift_over_time(a_k, b_k),
        "Phase Transition Frequency": phase_transition_frequency(a_k, b_k)
    }
    
    return features

from dynamicalFeatures import *
from geometricFeatures import *
from statisticalFeatures import *
from topologicalFeatures import *


def extract_all_features(data):
    """
    Extracts all features and returns them as a dictionary.
    """
    return {
        "Box Counting Dimension": box_counting_dimension(data),
        "Correlation Dimension": correlation_dimension(data),
        "Attractor Volume": attractor_volume(data),
        "Mean Distance": mean_distance(data),
        "Std Distance": std_distance(data),
        "Skewness Distance": skewness_distance(data),
        "Kurtosis Distance": kurtosis_distance(data),
        "PCA Variance Explained": principal_component_variance(data),
        "Recurrence Rate": recurrence_rate(data),
        "Determinism": determinism(data),
        "Trapping Time": trapping_time(data),
        "Lyapunov Exponent": lyapunov_exponent(data),
        "Shannon Entropy": shannon_entropy(data),
        "Permutation Entropy": permutation_entropy(data),
        "Betti Numbers": betti_numbers(data),
        "Persistent Homology Entropy": persistent_entropy(data)
    }
from .dynamicalFeatures import *
from .geometricFeatures import *
from .statisticalFeatures import *
from .topologicalFeatures import *


def extract_all_features(a_k, b_k):
    """
    Extracts all features and returns them as a dictionary.
    """
    β0, β1 = betti_numbers(a_k, a_k)
    return {
        "Box Counting Dimension": box_counting_dimension(a_k, b_k),
        #"Correlation Dimension": correlation_dimension(a_k, b_k),
        "Attractor Volume": attractor_volume(a_k, b_k),
        "Mean Distance": mean_distance(a_k, b_k),
        "Std Distance": std_distance(a_k, b_k),
        "Skewness Distance": skewness_distance(a_k, b_k),
        "Kurtosis Distance": kurtosis_distance(a_k, b_k),
        "PCA Variance Explained": principal_component_variance(a_k, b_k),
        "Recurrence Rate": recurrence_rate(a_k, b_k),
        "Determinism": determinism(a_k, b_k, time_delay=5),
        "Trapping Time": trapping_time(a_k, b_k, time_delay=5),
        #"Lyapunov Exponent": lyapunov_exponent(a_k, b_k),
        "Shannon Entropy": shannon_entropy(a_k, b_k),
        #"Permutation Entropy": permutation_entropy(a_k, b_k),
        "Betti0": β0,
        "Betti1": β1,
        "Persistent Homology Entropy": persistent_entropy(a_k, b_k)
    }
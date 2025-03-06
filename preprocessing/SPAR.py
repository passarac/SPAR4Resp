import numpy as np
import neurokit2 as nk


def embed_time_series(ts_signal, N, tau):
    """
    Get the delayed embeddings of the data (for a single data window).

    Parameters:
    - ts_signal (np.ndarray): The input data, either a 1D or 2D array.
      If 1D, it is treated as a single feature.
    - N (int): The dimension of the embedding.
    - tau (int): The time delay for the embedding - the number of samples to shift the data.

    Returns:
    - np.ndarray: The delayed embeddings of the data.
    """
    # Ensure ts_signal is at least 2D (convert 1D to 2D with one feature)
    if ts_signal.ndim == 1:
        ts_signal = ts_signal[:, np.newaxis]
    
    # Number of samples and features in the data
    num_samples, num_features = ts_signal.shape
    
    # Initialize a list to store the embeddings for each feature
    embeddings = []
    for j in range(num_features):
        # Compute delayed embedding for each feature independently
        embedding = nk.complexity_embedding(ts_signal[:, j], delay=tau, dimension=N)
        embeddings.append(embedding)
    
    # Concatenate embeddings for all features along the second axis (columns) to get a single array
    embeddings_np = np.concatenate(embeddings, axis=1)

    return embeddings_np



def compute_spar_projection(embedded_data, N, k=1):
    """
    Computes the SPAR attractor projection in 2D using Fourier-like basis vectors.
    
    Parameters:
        embedded_data (numpy.ndarray): Delay embedded data of shape (num_points, N).
        N (int): Embedding dimension.
        k (int): Projection parameter (default is k=1).
    
    Returns:
        numpy.ndarray: 2D projection of the attractor (a_k, b_k).
    """
    num_points = embedded_data.shape[0]
    
    # Compute the projection coefficients
    a_k = np.zeros(num_points)
    b_k = np.zeros(num_points)

    for j in range(N):
        cos_term = np.cos(2 * np.pi * j * k / N)
        sin_term = np.sin(2 * np.pi * j * k / N)
        
        a_k += (1 / np.sqrt(N)) * cos_term * embedded_data[:, j]
        b_k += (-1 / np.sqrt(N)) * sin_term * embedded_data[:, j]

    return a_k, b_k
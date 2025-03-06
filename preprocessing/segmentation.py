import numpy as np

def generate_sliding_windows(arr, window_size, overlap=0.0):
    """
    Generates sliding windows from a given numpy array with a specified overlap.

    Parameters:
    - arr (np.ndarray): Input numpy array. Can be:
        - 1D array: Shape (N,), where N is the length of the array.
        - 2D array: Shape (N, M), where N is the number of rows (samples) and M is the number of features per sample.
    - window_size (int): Size of each sliding window (number of rows in the window).
        - Must be a positive integer and not exceed the length of `arr` (or the number of rows for 2D arrays).
    - overlap (float): Fraction of overlap between consecutive windows. Must be a float between 0.0 (no overlap) and 1.0 (complete overlap).
        - Default is 0.0 (no overlap).

    Returns:
    - np.ndarray: Array of sliding windows. The output shape depends on the input `arr`:
        - For 1D input: Shape (num_windows, window_size).
        - For 2D input: Shape (num_windows, window_size, M), where M is the number of features per sample.

    Raises:
    - ValueError: If `window_size` is not a positive integer or exceeds the length of `arr`.
    - ValueError: If `overlap` is not a float between 0.0 and 1.0.
    - ValueError: If the calculated step size based on `overlap` results in a non-positive value.

    Example:
    >>> arr = np.array([1, 2, 3, 4, 5, 6])
    >>> generate_sliding_windows(arr, window_size=3, overlap=0.5)
    array([[1, 2, 3],
           [3, 4, 5],
           [5, 6, ...]])

    >>> arr = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
    >>> generate_sliding_windows(arr, window_size=2, overlap=0.5)
    array([[[1, 2], [3, 4]],
           [[3, 4], [5, 6]],
           [[5, 6], [7, 8]],
           ...])
    """
    # Validate window size
    if not isinstance(window_size, int) or window_size <= 0:
        raise ValueError("Window size must be a positive integer.")
    if window_size > len(arr):
        raise ValueError("Window size must not exceed the length of the array.")

    # Ensure overlap is a valid fraction
    if not (0.0 <= overlap < 1.0):
        raise ValueError("Overlap must be a float between 0.0 and 1.0.")

    # Calculate the step size based on overlap
    step_size = int(window_size * (1 - overlap))

    # Check for valid step size
    if step_size <= 0:
        raise ValueError("Overlap too high, resulting in a non-positive step size.")

    # Determine the number of windows
    num_windows = (len(arr) - window_size) // step_size + 1

    # Generate the sliding windows
    windows = np.array([arr[i : i + window_size] for i in range(0, num_windows * step_size, step_size)])

    return windows
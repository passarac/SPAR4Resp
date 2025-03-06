import numpy as np
from scipy.signal import butter, filtfilt

def butter_filter(data, btype='low', cutoff=2.0, fs=12.5, order=2):
    """
    Butterworth filter using filtfilt (zero-phase).
    :param data: 1D NumPy array of samples.
    :param btype: filtertype- choose between low or high
    :param cutoff: Cutoff frequency in Hz.
    :param fs: Sampling frequency in Hz.
    :param order: Filter order.
    :return: Filtered 1D array (same length as input).
    """
    # Normalize the frequency
    nyquist = 0.5 * fs
    normalized_cutoff = cutoff / nyquist

    # Design filter
    b, a = butter(order, normalized_cutoff, btype=btype, analog=False)

    # Zero-phase filtering
    filtered_data = filtfilt(b, a, data)
    return filtered_data
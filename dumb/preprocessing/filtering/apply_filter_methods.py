from . import filter_methods


def filter_accelerometer_data(sensor_dat, cutoff=2.0, fs=50.0, order=2):
    """
    Apply low-pass filtering to each axis of a set of signal.
    :param accel_data: NxM NumPy array of raw sensor data.
    :param cutoff: Cutoff frequency in Hz for the low-pass filter.
    :param fs: Sampling frequency in Hz.
    :param order: Filter order for the Butterworth filter.
    :return: Nx3 NumPy array of filtered accelerations.
    """
    filtered = np.zeros_like(sensor_dat)

    # Get the M dimension
    m_feat = filtered.shape[1]

    # Filter each axis (column) independently
    for axis in range(m_feat):
        filtered[:, axis] = butter_lowpass_filter(
            sensor_dat[:, axis],
            cutoff=cutoff,
            fs=fs,
            order=order
        )
    return filtered
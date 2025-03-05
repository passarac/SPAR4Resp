import numpy as np

def load_test_raw_accel_data():
    pass

def load_test_breathing_signal_data():
    pass

def simulate_resp_signal():
    pass

def generate_triaxial_accelerometer_data(N=750, freq=12.5, noise_level=0.2, periodic=True):
    """
    Simulates triaxial accelerometer data with optional periodic motion and noise.
    
    Parameters:
        N (int): Number of samples.
        freq (int): Frequency of simulated periodic motion (Hz).
        noise_level (float): Standard deviation of Gaussian noise.
        periodic (bool): If True, adds a periodic sinusoidal motion.

    Returns:
        np.ndarray: Simulated triaxial accelerometer data of shape (N, 3).
    """
    t = np.linspace(0, N / freq, N)
    
    if periodic:
        x = np.sin(2 * np.pi * 1 * t) + np.random.normal(0, noise_level, N)
        y = np.cos(2 * np.pi * 0.5 * t) + np.random.normal(0, noise_level, N)
        z = np.sin(2 * np.pi * 0.2 * t) + np.random.normal(0, noise_level, N)
    else:
        x = np.random.normal(0, noise_level, N)
        y = np.random.normal(0, noise_level, N)
        z = np.random.normal(0, noise_level, N)
    
    data = np.column_stack((x, y, z))
    return data
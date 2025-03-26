import numpy as np
import glob
import pandas as pd
from sklearn.feature_selection import mutual_info_regression
from scipy.signal import argrelextrema
from matplotlib import pyplot as plt
from tqdm import tqdm
from scipy.stats import mode
from joblib import Parallel, delayed
from pyinform.mutualinfo import mutual_info
from pyunicorn.timeseries import RecurrencePlot
from statsmodels.tsa.stattools import acf
from scipy.spatial import KDTree

def compute_mi_sklearn(signal, max_tau=50):
    """
    Computes Mutual Information (MI) between the original signal and its time-delayed version
    using sklearn's mutual_info_regression (which works for continuous signals). Also finds the
    optimal time delay (τ) as the first local minimum of MI.

    Parameters:
    - signal: 1D NumPy array representing the input time series signal.
    - max_tau: Integer, maximum time delay to evaluate (default is 50).

    Returns:
    - mi_values: NumPy array of MI values for each time delay from 1 to max_tau-1.
    - optimal_tau: Integer, first local minimum of MI (if found), otherwise global min.
    """

    mis = []  # List to store MI values for different time delays

    # Ensure the signal is a NumPy array
    signal = np.array(signal, dtype=np.float64)

    # Loop through different time delays from 1 to max_tau-1
    for tau in range(1, max_tau):
        # Extract original and delayed signal for given tau
        original = signal[:-tau].reshape(-1, 1)  # Reshape for sklearn compatibility
        delayed = signal[tau:]
        # Compute mutual information
        mi_value = mutual_info_regression(original, delayed)[0]
        mis.append(mi_value)

    # Convert to NumPy array
    mi_values = np.array(mis)

    # Find the first local minimum of MI
    minima_indices = argrelextrema(mi_values, np.less)[0]
    optimal_tau = minima_indices[0] + 1 if len(minima_indices) > 0 else np.argmin(mi_values) + 1

    return mi_values, optimal_tau


def compute_avg_mutual_info(signal, max_tau=50, bins=15):
    """
    Computes Average Mutual Information (AMI) between the original signal and its time-delayed version
    using pyinform's mutual_info function (which requires discrete inputs).

    Also finds the optimal time delay (τ) as the first local minimum of AMI.

    Parameters:
    - signal: 1D NumPy array representing the input time series signal.
    - max_tau: Integer, maximum time delay to evaluate (default = 50).
    - bins: Number of discrete bins to use for signal discretization (default = 10).

    Returns:
    - ami_values: NumPy array of AMI values for each time delay from 1 to max_tau-1.
    - optimal_tau: Integer, first local minimum of AMI (if found), otherwise global min.
    """

    # Ensure the signal is a NumPy array
    signal = np.array(signal, dtype=np.float64)

    # Step 1: Normalize the signal between 0 and 1
    signal = (signal - np.min(signal)) / (np.max(signal) - np.min(signal))

    # Step 2: Discretize into bins
    signal = np.round(signal * (bins - 1)).astype(int)  # Convert to integers

    ami_values = []  # List to store AMI values for different time delays

    # Loop through different time delays from 1 to max_tau-1
    for tau in range(1, max_tau):
        # Compute mutual information (discrete)
        ami_value = mutual_info(signal[:-tau], signal[tau:])
        ami_values.append(ami_value)

    # Convert to NumPy array
    ami_values = np.array(ami_values)

    # Find the first local minimum of AMI
    minima_indices = argrelextrema(ami_values, np.less)[0]
    optimal_tau = minima_indices[0] + 1 if len(minima_indices) > 0 else np.argmin(ami_values) + 1

    return ami_values, optimal_tau


def compute_acf_tau(signal, max_lag=50):
    """
    Computes the Autocorrelation Function (ACF) for a given signal and determines the optimal
    time delay (τ) where ACF first drops below 1/e.

    Parameters:
    - signal: 1D NumPy array, the input time series.
    - max_lag: Maximum number of lags to compute ACF (default = 50).

    Returns:
    - acf_values: NumPy array of ACF values for lags 0 to max_lag.
    - tau_acf: Integer, the first lag where ACF drops below 1/e.
    """
    # Compute autocorrelation function (ACF)
    acf_values = acf(signal, nlags=max_lag, fft=True)  # Use FFT for speed
    # Find the first index where ACF drops below 1/e (~0.3679)
    tau_acf = np.where(acf_values < np.exp(-1))[0][0] if np.any(acf_values < np.exp(-1)) else max_lag
    # return ACF values and the optimal tau determined by ACF
    return acf_values, tau_acf


def compute_optimal_tau(signal, max_tau):
    """
    Computes the optimal time delay (τ) for phase-space reconstruction using three different methods:
    - Mutual Information (MI)
    - Average Mutual Information (AMI)
    - Autocorrelation Function (ACF)

    Parameters:
    - signal (numpy.ndarray): 1D NumPy array representing the input time series signal.
    - max_tau (int): Maximum time delay (τ) to evaluate.

    Returns:
    - dict: A dictionary containing:
        - "mi_values" (numpy.ndarray): Mutual Information values for each τ.
        - "tau_mi" (int): Optimal τ determined from the first local minimum of MI.
        - "ami_values" (numpy.ndarray): Average Mutual Information values for each τ.
        - "tau_ami" (int): Optimal τ determined from the first local minimum of AMI.
        - "acf_values" (numpy.ndarray): Autocorrelation values for each τ.
        - "tau_acf" (int): Optimal τ determined as the first point where ACF drops below 1/e.

    Notes:
    - MI and AMI are better for capturing **both linear and nonlinear dependencies**.
    - ACF only detects **linear correlations**, so it may underestimate τ in complex systems.
    - Comparing all three methods helps ensure a robust choice of τ for phase-space embedding.
    """

    # Compute Mutual Information (MI)
    mi_values, tau_mi = compute_mi_sklearn(signal, max_tau=max_tau)

    # Compute Average Mutual Information (AMI)
    ami_values, tau_ami = compute_avg_mutual_info(signal, max_tau=max_tau)

    # Compute Autocorrelation Function (ACF)
    acf_values, tau_acf = compute_acf_tau(signal, max_lag=max_tau)

    # Store results in a dictionary
    results = {
        "mi_values": mi_values,
        "tau_mi": tau_mi,
        "ami_values": ami_values,
        "tau_ami": tau_ami,
        "acf_values": acf_values,
        "tau_acf": tau_acf
    }

    return results

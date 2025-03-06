from sklearn.preprocessing import StandardScaler

def normalize_with_standard_scaler(data, scalers=None):
    """
    Normalizes the data using StandardScaler.

    Parameters:
        data (numpy.ndarray): Input data of shape (samples, time_steps, features).
        scalers (list of StandardScaler): Optional. Pre-fitted StandardScalers for each feature.

    Returns:
        numpy.ndarray: Normalized data.
        list: List of fitted StandardScalers (one for each feature).
    """
    # Initialize scalers if not provided
    if scalers is None:
        # Create a StandardScaler for each feature (axis)
        scalers = [StandardScaler() for _ in range(data.shape[2])]
        # Fit each scaler to the corresponding feature data
        for i in range(data.shape[2]):
            # Extract all data for the current feature, reshape to (samples * time_steps, 1)
            feature_data = data[:, :, i].reshape(-1, 1)
            scalers[i].fit(feature_data)  # Fit the scaler to this feature

    # Apply the fitted scalers to normalize the data
    for i in range(data.shape[2]):
        # Extract all data for the current feature, reshape to (samples * time_steps, 1)
        feature_data = data[:, :, i].reshape(-1, 1)
        # Transform the feature data using the fitted scaler
        data[:, :, i] = scalers[i].transform(feature_data).reshape(data.shape[0], data.shape[1])

    # Return the normalized data and the fitted scalers
    return data, scalers
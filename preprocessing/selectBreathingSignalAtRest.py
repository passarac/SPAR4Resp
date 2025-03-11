import pandas as pd
import numpy as np

from preprocessing.segmentation import generate_sliding_windows

import math

def split_by_time_gaps(df, timestamp_column='timestamp', gap_size=1):
    """
    Split a dataframe into multiple dataframes based on gaps in the timestamp column.

    Parameters:
    - df (pd.DataFrame): The input dataframe.
    - timestamp_column (str): The name of the timestamp column in the dataframe.
    - gap_size (int): The maximum gap size (in seconds) to consider as part of the same group.

    Returns:
    - list: A list of dataframes split based on gaps in the timestamp column.
    """

    # Ensure timestamp is in datetime format
    df[timestamp_column] = pd.to_datetime(df[timestamp_column])

    # Identify gaps greater than 1 second
    df['gap'] = df[timestamp_column].diff().dt.total_seconds() > gap_size

    # Assign group numbers based on gaps
    groups = df['gap'].cumsum()

    # Split into list of dataframes
    split_dfs = [group_df.drop(columns=['gap']) for _, group_df in df.groupby(groups)]

    return split_dfs


def select_and_segment_breathing_signal_at_rest(df, timestamp_column='interpolatedPhoneTimestamp',
                             breathing_signal_column='breathingSignal',
                             mappedActivity_column='mapped_activity',
                             window_size=math.ceil((12.5*60)*1)):
    """
    Selects the breathing signal during stationary activities and segments it into windows.

    Parameters:
    - df (pd.DataFrame): The input dataframe.
    - timestamp_column (str): The name of the timestamp column in the dataframe.
    - breathing_signal_column (str): The name of the breathing signal column in the dataframe.
    - mappedActivity_column (str): The name of the activity column in the dataframe.
    - window_size (int): The size of the sliding window.

    Returns:
    - list_windows: A list of segmented breathing signal windows.

    """

    list_windows = []

    stationary_activities = ["lyingBack", "lyingLeft", "lyingRight", "lyingStomach", "sittingStanding"]

    # Select timestamp, breathing signal, and activity classification columns
    df_oi = df[[timestamp_column, breathing_signal_column, mappedActivity_column]]
    # remove first and last 50 rows because those always seem to be the most 'spikey'
    df_oi = df_oi.iloc[50:-50]

    # select only the stationary activities
    df_oi = df_oi[df_oi[mappedActivity_column].isin(stationary_activities)]

    df_oi['new_timestamp'] = pd.to_datetime(df_oi[timestamp_column], unit='ms')
    split_dfs = split_by_time_gaps(df_oi, timestamp_column='new_timestamp', gap_size=1)

    # iterate over the split dataframes
    for df in split_dfs:
        # check if the length of the dataframe is at least 750 (1 minute)
        if len(df) < 750:
            continue
        # convert to numpy
        bs_np = df.to_numpy()
        # there are 750 data samples in one minute
        data_windows = generate_sliding_windows(bs_np, window_size=window_size)
        list_windows.append(data_windows)

    concat_list_windows = np.concatenate(list_windows, axis=0)
    return concat_list_windows


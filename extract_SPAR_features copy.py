import pandas as pd
import numpy as np
import glob
from pathlib import Path
from dataHandling.dataHandling import get_SPAR_subject_folders, get_SPAR_files
from preprocessing.SPAR.extractFeatures import *
import pickle

def extract_SPAR_features(base_dataset_path: Path):
    subject_folders = get_SPAR_subject_folders(str(base_dataset_path))
    
    for subject_folder in subject_folders[1:]:
        subject_id = subject_folder[-6:]
        print(f"Processing subject {subject_id}...")

        SPAR_files = get_SPAR_files(subject_folder)
        
        for SPAR_file in SPAR_files:
            print(f"Processing file {SPAR_file}...")
            filename = Path(SPAR_file).name

            save_path = base_dataset_path / "SPAR_results" / "features" / subject_id / filename
            save_path.parent.mkdir(parents=True, exist_ok=True)

            if (save_path.with_suffix(".pkl")).exists():
                # print(f"File {save_path} already processed. Skipping...")
                continue

            # Load SPAR data
            with open(SPAR_file, "rb") as f:
                SPAR_dat = pickle.load(f)

            SPAR_features_list = []

            for data in SPAR_dat:
                SPAR_attractor = data['SPAR_projection']
                original_data = data['original_arr']

                # remove any NaN values from SPAR attractor
                SPAR_attractor = SPAR_attractor[~np.isnan(SPAR_attractor).any(axis=1)]

                # get the timestamp column
                timestamp = original_data[:, 0]
                # get the activity column
                activity = original_data[:, 2]

                # get the mode of the activity
                values, counts = np.unique(activity, return_counts=True)
                activity_mode = values[np.argmax(counts)]  # Get the most frequent value

                # get the start and end timestamp
                start_timestamp = timestamp[0]
                end_timestamp = timestamp[-1]

                try:
                    # Extract SPAR features
                    SPAR_features = extract_all_features(SPAR_attractor[:, 0], SPAR_attractor[:, 1])
                except Exception as e:
                    print(f"Error processing file {SPAR_file}: {e}")
                    continue
                
                # add the activity mode, start timestamp, and end timestamp to the features
                SPAR_features["activity_mode"] = activity_mode
                SPAR_features["start_timestamp"] = start_timestamp
                SPAR_features["end_timestamp"] = end_timestamp

                SPAR_features_list.append(SPAR_features)

            print(len(SPAR_dat), len(SPAR_features_list))

            # Save SPAR features
            with open(save_path.with_suffix(".pkl"), "wb") as f:
                pickle.dump(SPAR_features_list, f)


if __name__ == "__main__":
    base_dataset_path = Path("G:\\Shared drives\\PhD\\Data\\SMILE")
    extract_SPAR_features(base_dataset_path)

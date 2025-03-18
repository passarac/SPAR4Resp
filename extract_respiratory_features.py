import pandas as pd
import numpy as np
import glob
import pickle
from pathlib import Path
from dataHandling.dataHandling import *
from preprocessing.respWaveformFeatExtraction.respiratoryFeatures import *
from preprocessing.respWaveformFeatExtraction.calculateContinuousBreathFeatures import *

def extract_respiratory_features(base_dataset_path: Path):
    subject_folders = get_breathingSignal_subject_folders(str(base_dataset_path))

    for subject_folder in subject_folders:
        subject_id = subject_folder[-6:]
        print(f"Processing subject {subject_id}...")

        bs_files = get_breathingSignal_files(subject_folder)

        for bs_file in bs_files:
            print(f"Processing file {bs_file}...")

            # Get filename
            filename = Path(bs_file).name

            # remove extension from filename
            filename = filename.split(".")[0]

            # Define save path
            save_path = base_dataset_path / "respiratoryFeatures" / "continuous" / subject_id / filename
            save_path.parent.mkdir(parents=True, exist_ok=True)

            # Load breathing signal data
            # It has shape (N, 750, 3) where:
            # N is the number of windows, 750 is the window size, and 3 is the number of columns (timestamp, breathing signal, activity)
            np_dat = np.load(bs_file, allow_pickle=True)["array"]

            result_arr = []

            for arr in np_dat:
                # Get the breathing signal (second column)
                rsp_signal = arr[:, 1]
                timestamp = arr[:, 0]
                activity = arr[:, 2]
                try:
                    # Extract respiratory features
                    # Extract continuous respiratory features
                    resp_features = calculate_TS_breathFeatures(timestamps=timestamp, signal=rsp_signal)
                except Exception as e:
                    print("Error encountered, skipping this data window...")
                    continue

                result_arr.append(resp_features)

            # print(len(result_arr))
            # Save respiratory feature data
            with open(str(save_path) + ".pkl", "wb") as f:
                pickle.dump(result_arr, f)
            #break
        #break


def main():
    base_dataset_path = Path("G:\\Shared drives\\PhD\\Data\\SMILE")
    extract_respiratory_features(base_dataset_path)

if __name__ == "__main__":
    main()

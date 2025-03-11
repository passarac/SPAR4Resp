import pandas as pd
import numpy as np
from preprocessing.SPAR.generateSPARAttractor import *
import glob
from pathlib import Path
from dataHandling.dataHandling import *
import pickle

def apply_SPAR(base_dataset_path: Path):
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
            save_path = base_dataset_path / "SPAR_results" / "attractor" / subject_id / filename
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
                    # Apply SPAR
                    # Generate delayed embeddings
                    embedded_rsp = embed_time_series(rsp_signal, N=3, tau=5)
                    # SPAR 2D Projection
                    a_k, b_k = compute_spar_projection(embedded_rsp, N=3, k=1)
                except Exception as e:
                    print("Error encountered, skipping this data window...")
                    continue

                # combine timestamp, rsp_signal, activity, a_k, and b_k
                original_arr = np.column_stack((timestamp, rsp_signal, activity))
                SPAR_projection = np.column_stack((a_k, b_k))
                res = {
                    "original_arr": original_arr,
                    "SPAR_projection": SPAR_projection
                }

                result_arr.append(res)

            # print(len(result_arr))
            # Save SPAR data
            with open(str(save_path) + ".pkl", "wb") as f:
                pickle.dump(result_arr, f)
            #break
        #break


def main():
    base_dataset_path = Path("G:\\Shared drives\\PhD\\Data\\SMILE")
    apply_SPAR(base_dataset_path)

if __name__ == "__main__":
    main()

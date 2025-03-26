import pandas as pd
import numpy as np
from preprocessing.SPAR.generateSPARAttractor import *
from pathlib import Path
from dataHandling.dataHandling import *
import pickle
import argparse

# SPAR parameters
TAU = 9
N = 3
K = 1

def apply_SPAR(base_dataset_path: Path, subject_id: str = None, randomize_folders: bool = False, verbose: bool = True):
    """
    Applies Symmetric Projection Attractor Reconstruction (SPAR) to segmented breathing signals
    for each subject, and saves the resulting 2D attractor projections.

    For each breathing signal window:
        - Extracts the respiratory signal.
        - Applies time-delay embedding and SPAR projection.
        - Saves the original data and the SPAR projection in a .pkl file.

    Args:
        base_dataset_path (Path): Path to the dataset directory containing breathingSignal folders.
        subject_id (str, optional): If provided, only processes the subject with this ID. Defaults to None.
        randomize_folders (bool, optional): If True, shuffles the subject folder order. Defaults to False.
        verbose (bool, optional): If True, prints detailed logs. Defaults to True.

    Returns:
        None
    """
    subject_folders = get_breathingSignal_subject_folders(str(base_dataset_path))

    # Filter for one subject if provided
    if subject_id:
        subject_folders = [folder for folder in subject_folders if folder.endswith(subject_id)]
        if len(subject_folders) == 0:
            print(f"No folder found for subject ID: {subject_id}")
            return

    if randomize_folders:
        subject_folders = np.random.permutation(subject_folders)

    # Iterate over each subject folder
    for subject_folder in subject_folders:

        # Get subject ID
        subject_id = subject_folder[-6:]

        # Print subject ID
        if verbose:
            print(f"Processing subject {subject_id}...")

        # Get all breathing signal files for this subject
        bs_files = get_breathingSignal_files(subject_folder)

        # Iterate over each breathing signal file
        for bs_file in bs_files:

            if verbose:
                print(f">> Processing file {bs_file}...")

            # Get filename
            filename = Path(bs_file).name

            # remove extension from filename
            filename = filename.split(".")[0]

            # Define save path
            save_path = base_dataset_path / "SPAR_results" / "attractor" / subject_id / filename
            save_path.parent.mkdir(parents=True, exist_ok=True)

            # Skip if file already processed
            if (save_path.with_suffix(".pkl")).exists():
                print(f">>>>> File {bs_file} already processed. Skipping...")
                continue

            # Load breathing signal data
            # It has shape (N, 750, 3) where:
            # N is the number of windows, 750 is the window size, and 3 is the number of columns (timestamp, breathing signal, activity)
            np_dat = np.load(bs_file, allow_pickle=True)["array"]

            # Initialize result array to store SPAR projections
            result_arr = []

            # Iterate over each window
            for arr in np_dat:

                # Get the breathing signal (second column)
                rsp_signal = arr[:, 1]
                timestamp = arr[:, 0]
                activity = arr[:, 2]

                try:
                    # Apply SPAR
                    # Generate delayed embeddings
                    embedded_rsp = embed_time_series(rsp_signal, N=N, tau=TAU)
                    # SPAR 2D Projection
                    a_k, b_k = compute_spar_projection(embedded_rsp, N=N, k=K)
                except Exception as e:
                    print(">>>>> Error encountered, skipping this data window...")
                    continue

                # combine timestamp, rsp_signal, activity, a_k, and b_k
                original_arr = np.column_stack((timestamp, rsp_signal, activity))
                SPAR_projection = np.column_stack((a_k, b_k))
                res = {
                    "original_arr": original_arr,
                    "SPAR_projection": SPAR_projection
                }

                # Append to result_arr
                result_arr.append(res)
            
            # Save SPAR data
            with open(str(save_path) + ".pkl", "wb") as f:
                pickle.dump(result_arr, f)


def main():
    parser = argparse.ArgumentParser(description="Apply SPAR projection to breathing signal windows.")
    
    parser.add_argument("base_dataset_path",
                        type=Path,
                        help="Path to the base dataset directory.")
    
    parser.add_argument("--subject_id",
                        type=str,
                        default=None,
                        help="Process only the subject with this ID (e.g., 'INH001').")
    
    parser.add_argument("--randomize_folders",
                        action="store_true",
                        help="Randomize the order of subject folders.")
    
    parser.add_argument("--verbose",
                        action="store_true",
                        help="Print additional information during processing.")

    args = parser.parse_args()

    apply_SPAR(
        base_dataset_path=args.base_dataset_path,
        subject_id=args.subject_id,
        randomize_folders=args.randomize_folders,
        verbose=args.verbose
    )


if __name__ == "__main__":
    main()
import pandas as pd
import numpy as np
from pathlib import Path
from dataHandling.dataHandling import get_SPAR_subject_folders, get_SPAR_files
from preprocessing.SPAR.extractFeatures import *
import pickle
import argparse
from tqdm import tqdm

def extract_SPAR_features(base_dataset_path: Path, subject_id: str = None, randomize_folders: bool = False, verbose: bool = True):
    """
    Extracts SPAR-based features from precomputed SPAR projections for each subject.

    This function processes each subject's SPAR projection files, extracts relevant 
    features from 2D SPAR attractors, and stores the resulting feature dictionaries 
    in a structured directory as pickled `.pkl` files.

    For each SPAR projection in the file:
        - Removes NaN values from the attractor.
        - Extracts geometric features from the SPAR attractor.
        - Records the activity mode, start timestamp, and end timestamp.
        - Appends the result to a list of feature dictionaries.

    Args:
        base_dataset_path (Path): Path to the root dataset directory.
        subject_id (str, optional): If provided, only processes the subject folder ending with this ID. Defaults to None.
        randomize_folders (bool, optional): Whether to randomize the order of subject folders. Defaults to False.
        verbose (bool, optional): Whether to print progress information. Defaults to True.

    Returns:
        None
    """
    # Get the subject folders
    subject_folders = get_SPAR_subject_folders(str(base_dataset_path))

    # Filter by subject_id if specified
    if subject_id:
        subject_folders = [folder for folder in subject_folders if folder.endswith(subject_id)]
        if len(subject_folders) == 0:
            print(f"No folder found for subject ID: {subject_id}")
            return
    
    # Randomize the order of the subject folders if randomize_folders is True
    if randomize_folders:
        subject_folders = np.random.permutation(subject_folders)
    
    # Process each subject folder
    for subject_folder in tqdm(subject_folders, position=0, leave=True, desc="Processing subjects"):

        # Get the subject ID
        subject_id = subject_folder[-6:]

        if verbose:
            print(f"Processing subject {subject_id}...")

        # Get SPAR files
        SPAR_files = get_SPAR_files(subject_folder)
        
        # Process each SPAR file
        for SPAR_file in tqdm(SPAR_files, position=1, leave=True, desc="Processing SPAR files for: " + subject_id):

            if verbose:
                print(f">> Processing file {SPAR_file}...")

            # Get filename
            filename = Path(SPAR_file).name

            # Define save path
            save_path = base_dataset_path / "SPAR_results" / "features" / subject_id / filename
            # Create parent directories if they don't exist
            save_path.parent.mkdir(parents=True, exist_ok=True)

            # Skip if the file has already been processed
            if (save_path.with_suffix(".pkl")).exists():
                print(f">>>>> File {save_path} already processed. Skipping...")
                continue

            # Load SPAR data
            with open(SPAR_file, "rb") as f:
                SPAR_dat = pickle.load(f)

            # List to store SPAR features
            SPAR_features_list = []

            # Process each SPAR projection
            for data in tqdm(SPAR_dat, position=2, leave=False, desc="Processing SPAR projections for file: " + filename):

                # Get SPAR attractor and original data
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
                    a_k = SPAR_attractor[:, 0] 
                    b_k = SPAR_attractor[:, 1]
                    SPAR_features = extract_all_features(a_k, b_k)
                except Exception as e:
                    (f">>>>> Error processing file {SPAR_file}: {e}")
                    continue
                
                # add the activity mode, start timestamp, and end timestamp to the features
                SPAR_features["activity_mode"] = activity_mode
                SPAR_features["start_timestamp"] = start_timestamp
                SPAR_features["end_timestamp"] = end_timestamp

                SPAR_features_list.append(SPAR_features)

            #print(len(SPAR_dat), len(SPAR_features_list))

            # Save SPAR features
            with open(save_path.with_suffix(".pkl"), "wb") as f:
                pickle.dump(SPAR_features_list, f)


def main():
    parser = argparse.ArgumentParser(description="Extract SPAR features from subject data.")

    parser.add_argument("base_dataset_path",
                        type=Path,
                        help="Path to the base dataset directory.")

    parser.add_argument("--subject_id",
                        type=str,
                        default=None,
                        help="Specify a subject ID to process only that subject.")
    
    parser.add_argument("--randomize_folders",
                        action="store_true",
                        help="Randomize the order of subject folders.")
    
    parser.add_argument("--verbose",
                        action="store_true",
                        help="Enable verbose output.")

    args = parser.parse_args()

    extract_SPAR_features(
        base_dataset_path=args.base_dataset_path,
        subject_id=args.subject_id,
        randomize_folders=args.randomize_folders,
        verbose=args.verbose
    )

if __name__ == "__main__":
    main()
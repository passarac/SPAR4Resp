import pandas as pd
import numpy as np
import argparse
from dataHandling.dataHandling import *
from preprocessing.selectBreathingSignalAtRest import *


def process_subject_data(base_dataset_path: Path, randomize_folders: bool = False, verbose: bool = True, subject_id: str = None):

    """
    Processes HAR subject data to extract and segment breathing signals at rest.

    This function iterates through subject folders within a given dataset path,
    loads HAR classification files, and extracts breathing signals at rest using
    a segmentation function. The segmented breathing windows are saved as compressed
    `.npz` files for each subject and file.

    Args:
        base_dataset_path (Path): Path to the root dataset directory containing subject folders.
        randomize_folders (bool, optional): If True, shuffles the order of subject folders before processing. Defaults to False.
        verbose (bool, optional): If True, prints detailed logging messages during processing. Defaults to True.
        subject_id (str, optional): If provided, processes only the subject folder ending with this ID. Defaults to None (i.e., all subjects).

    Returns:
        None
    """

    # Get the subject folders
    subject_folders = get_HAR_subject_folders(str(base_dataset_path))

    # Randomize the order of the subject folders if randommise_folders is True
    if randomize_folders:
        subject_folders = np.random.permutation(subject_folders)

    # If a specific subject ID is provided, filter the list
    if subject_id is not None:
        subject_folders = [folder for folder in subject_folders if folder.endswith(subject_id)]
        if len(subject_folders) == 0:
            print(f"No folder found for subject ID: {subject_id}")
            return
    
    # Process each subject folder
    for subject_folder in subject_folders:

        # Get the subject ID
        subject_id = subject_folder[-6:]

        if verbose:
            print(f"Processing subject {subject_id}...")

        # Get the HAR classification files
        HAR_files = get_HAR_classification_files(subject_folder)
        
        # Process each HAR classification file
        for HAR_file in HAR_files:

            if verbose:
                print(f">> Processing file {HAR_file}...")

            # Get the filename
            filename = Path(HAR_file).name
            # remove .csv from filename
            filename = filename.split(".")[0]

            # Define the save path
            save_path = base_dataset_path / "breathingSignal" / subject_id / filename
            # Create the parent directories if they do not exist
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Skip if the file has already been processed
            if (save_path.with_suffix(".npz")).exists():
                if verbose:
                    print(f">>>>>> File {HAR_file} already processed. Skipping...")
                continue

            # Load HAR classification data
            HAR_data = pd.read_csv(HAR_file, on_bad_lines='skip')

            try:
                # Select and segment breathing signal at rest
                list_windows = select_and_segment_breathing_signal_at_rest(HAR_data)
            except Exception as e:
                if verbose:
                    print(f">>>>> Error processing file {HAR_file}: {e}")
                continue

            # Remove the last column (Pandas timestamp column)
            list_windows = list_windows[:, :, :-1]

            # Save segmented breathing signal windows
            np.savez_compressed(save_path, array=list_windows)


def main():
    parser = argparse.ArgumentParser(
        description="Getting Breathing Signal At Rest from Each Subject."
    )

    # Required positional argument ---------
    parser.add_argument(
        "base_dataset_path",
        type=Path,
        help="Path to the base dataset directory."
    )

    # Optional arguments -------------------
    parser.add_argument(
        "--randomize_folders",
        action="store_true", # if the flag is present, then it will be true, otherwise false
        help="Randomize the order of the subject folders."
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print additional information."
    )

    parser.add_argument(
        "--subject_id",
        type=str,
        default=None,
        help="Process only the subject with this ID (e.g., 'INH001')."
    )  

    args = parser.parse_args()
    process_subject_data(args.base_dataset_path)


if __name__ == "__main__":
    main()
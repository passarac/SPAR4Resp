import pandas as pd
import numpy as np

from dataHandling.dataHandling import *
from preprocessing.selectBreathingSignalAtRest import *


def process_subject_data(base_dataset_path: Path):
    subject_folders = get_HAR_subject_folders(str(base_dataset_path))
    
    for subject_folder in subject_folders:
        subject_id = subject_folder[-6:]
        print(f"Processing subject {subject_id}...")

        HAR_files = get_HAR_classification_files(subject_folder)
        
        for HAR_file in HAR_files:
            print(f"Processing file {HAR_file}...")
            filename = Path(HAR_file).name

            save_path = base_dataset_path / "breathingSignal" / subject_id / filename
            save_path.parent.mkdir(parents=True, exist_ok=True)

            if (save_path / ".npz").exists():
                print(f"File {HAR_file} already processed. Skipping...")
                continue

            # Load HAR classification data
            HAR_data = pd.read_csv(HAR_file, on_bad_lines='skip')

            try:
                # Select and segment breathing signal at rest
                list_windows = select_and_segment_breathing_signal_at_rest(HAR_data)
            except Exception as e:
                print(f"Error processing file {HAR_file}: {e}")
                continue

            # Remove the last column (Pandas timestamp column)
            list_windows = list_windows[:, :, :-1]

            # Save segmented breathing signal windows
            np.savez_compressed(save_path, array=list_windows)

def main():
    base_dataset_path = Path("G:\\Shared drives\\PhD\\Data\\SMILE")
    process_subject_data(base_dataset_path)

if __name__ == "__main__":
    main()

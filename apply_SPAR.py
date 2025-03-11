import pandas as pd
import numpy as np
from preprocessing.SPAR.generateSPARAttractor import *

import glob
from pathlib import Path




def main():
    base_dataset_path = Path("G:\\Shared drives\\PhD\\Data\\SMILE")
    subject_folders = get_breathingSignal_subject_folders(str(base_dataset_path))

    for subject_folder in subject_folders:
        bs_files = get_breathingSignal_files(subject_folder)
        print(bs_files)
        break

if __name__ == "__main__":
    main()

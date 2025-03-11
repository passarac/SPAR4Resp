import glob
from pathlib import Path


def get_raw_respeck_subject_folders(dataset_path: str):
    """
    Get all subject folders containing raw Respeck data.

    Parameters:
    - dataset_path (Path): The base dataset folder.

    Returns:
    - subject_folders (list): List of subject folders containing raw Respeck data.
    """
    dataset_folder = Path(dataset_path)
    raw_respeck_folder = dataset_folder / "rawRespeck"
    subject_folders = glob.glob(str(raw_respeck_folder / "*"))

    return subject_folders



def get_HAR_subject_folders(dataset_path: str):
    """
    Get all subject folders containing HAR classification data.

    Parameters:
    - dataset_path (Path): The base dataset folder.

    Returns:
    - subject_folders (list): List of subject folders containing HAR classification data.
    """
    dataset_folder = Path(dataset_path)
    HAR_folder = dataset_folder / "classification_results" / "HAR"
    subject_folders = glob.glob(str(HAR_folder / "*"))

    return subject_folders



def get_breathingSignal_subject_folders(base_dataset_path: str):
    """
    Get all subject folders containing breathing signal data.

    Parameters:
    - base_dataset_path (Path): The base dataset folder.

    Returns:
    - subject_folders (list): List of subject folders containing breathing signal data
    """
    base_dataset_path = Path(base_dataset_path)
    breathingSignal_folder = base_dataset_path / "breathingSignal"
    subject_folders = glob.glob(str(breathingSignal_folder / "*"))
    return subject_folders




def get_raw_respeck_files(subject_folder: str):
    """
    Get all raw Respeck files for a given subject folder.

    Parameters:
    - subject_folder (Path): The subject folder containing raw Respeck data.

    Returns:
    - respeck_files (list): List of raw Respeck files for the subject.
    """
    subject_folder = Path(subject_folder)  # Convert to pathlib.Path if it is a string
    respeck_files = glob.glob(str(subject_folder / "*.csv"))
    return respeck_files



def get_HAR_classification_files(subject_folder: str):
    """
    Get all HAR classification files for a given subject folder.

    Parameters:
    - subject_folder (Path): The subject folder containing HAR classification data.

    Returns:
    - HAR_files (list): List of HAR classification files for the subject.
    """
    subject_folder = Path(subject_folder)  # Convert to pathlib.Path if it is a string
    HAR_files = glob.glob(str(subject_folder / "*.csv"))
    return HAR_files



def get_breathingSignal_files(subject_folder: str):
    """
    Get all breathing signal files for a given subject folder.

    Parameters:
    - subject_folder (Path): The subject folder containing breathing signal data.

    Returns:
    - breathingSignal_files (list): List of breathing signal filepaths for the subject.
    """
    subject_folder = Path(subject_folder)
    breathingSignal_files = glob.glob(str(subject_folder / "*.npz"))
    return breathingSignal_files
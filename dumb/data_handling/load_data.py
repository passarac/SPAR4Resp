import glob


def get_rawRespeck_filepaths(dataset_folder_path):
    """
    How data is expected to be organized:
    rawRespeck/
    ├── subject001/
    └── fileContainingRawRespeckDat1.csv
    └── fileContainingRawRespeckDat2.csv
    └── fileContainingRawRespeckDat3.csv
    ├── subject002/
    ├── subject003/
    ...
    ├── subjectN/
    """

    subject_folders = glob.glob(dataset_folder_path + '/*')

    raw_respeck_filepaths = []

    for folder in subject_folders:
        files = glob.glob(folder + '/*')
        raw_respeck_filepaths += files
    
    return(raw_respeck_filepaths)
from applyClassification.HARClassifier import *
from dataHandling.dataHandling import *
from pathlib import Path
import pandas as pd

import argparse

def main(base_dataset_path):

    problematic_files = []

    # Define paths
    base_dataset_path = Path(base_dataset_path)
    scaler_path = Path("G:\\Shared drives\\PhD\\speckled_dev\\applyClassification\\scalers\\scaler.pkl")
    model_path = Path("G:\\Shared drives\\PhD\\speckled_dev\\applyClassification\\pretrainedModels\\bigru.keras")
    classification_results_base = base_dataset_path / "classification_results" / "HAR"
    
    # Get subject folders
    subject_folders = get_raw_respeck_subject_folders(str(base_dataset_path))
    
    for subject_folder in subject_folders:
        subject_id = subject_folder[-6:]
        print(f"Beginning HAR Classification for {subject_id}...")
        
        respeck_files = get_raw_respeck_files(subject_folder)
        
        # Initialize classifier
        HAR_classifier = HARClassifier(window_size=25,
                                       scaler_path=str(scaler_path),
                                       model_path=str(model_path))
        
        # Create folder for classification results
        subject_results_folder = classification_results_base / subject_id
        subject_results_folder.mkdir(parents=True, exist_ok=True)
        
        for respeck_file in respeck_files:
            print(f"Processing file {respeck_file}...")
            
            try:
                # Extract filename
                filename = Path(respeck_file).name

                # Define save path
                save_path = subject_results_folder / filename

                # Check if file has already been processed
                if save_path.exists():
                    print(f"File {respeck_file} already processed. Skipping...")
                    continue
                
                # Load the raw respeck data
                respeck_data = pd.read_csv(respeck_file, on_bad_lines='skip')
                
                # Run classification
                HAR_classifier.preprocess(respeck_data)
                HAR_classifier.apply_algorithm()
                HAR_classifier.postprocess()
                res_df = HAR_classifier.result_df
                
                # Save results
                res_df.to_csv(save_path, index=False)
                print(f"Results saved to {save_path}")
                
                # Reset classifier for next file
                HAR_classifier.reset()
                
            except Exception as e:
                print(f"Error processing file {respeck_file}: {e}")
                problematic_files.append(respeck_file)
                continue
            
        #break
    
    print("HAR Classification complete.") 
    # save problematic filenames as a text file
    problematic_file_log = classification_results_base / "problematic_files.txt"
    with open(problematic_file_log, "w") as f:
        for file in problematic_files:
            f.write(file + "\n")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run HAR Classification on Respeck Data")
    parser.add_argument("base_dataset_path", type=str, help="Path to the base data directory")
    args = parser.parse_args()
    
    main(args.base_dataset_path)
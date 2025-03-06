import joblib
import numpy as np
import pandas as pd
import sklearn
from tensorflow.keras.models import load_model

from preprocessing import normalisation
from preprocessing import segmentation

class HARClassifier():
    def __init__(self, window_size, scaler_path, model_path):

        self.window_size = window_size
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.scaler = joblib.load(self.scaler_path)
        self.model = load_model(self.model_path)

        self.input_data = None
        self.preprocessed_data = None
        self.output_data = None
        self.data_windows = None
        self.pred_probabilities = None
        self.pred_labels = None
        self.result_df = None

        self.original_columns = ['interpolatedPhoneTimestamp', 'breathingSignal', 'breathingRate',
                                 'activityLevel', 'x', 'y', 'z',]

        self.output_columns = self.original_columns + ['0_prob', '1_prob', '2_prob', '3_prob',
                               '4_prob', '5_prob', '6_prob', '7_prob', '8_prob', '9_prob', '10_prob',
                               '11_prob','final_label', 'mapped_activity']

        # Define the mapping of integer labels to activity names
        self.activity_labels = np.array([
            "ascending", "descending", "lyingBack", "lyingLeft", "lyingRight",
            "lyingStomach", "miscMovement", "normalWalking", "notWorn", "running",
            "shuffleWalking", "sittingStanding"
        ])

    def reset(self):
        self.input_data = None
        self.preprocessed_data = None
        self.output_data = None
        self.data_windows = None
        self.pred_probabilities = None
        self.pred_labels = None
        self.result_df = None

    def preprocess(self, data_df):
        """
        Processes the data into a format that is ready for ML classification

        Parameters:
        - data_df: dataframe containing raw respeck data
        """
        # Columns of interest - these are columns from the original data that we will want to keep in the output
        data_df = data_df[self.original_columns]
        # Assign input dataframe to self.input_data
        self.input_data = data_df
        # Convert the dataframe to a numpy array
        dat_np = self.input_data.to_numpy()
        # Extract sliding windows using the function generate_sliding_windows
        self.data_windows = segmentation.generate_sliding_windows(dat_np, self.WINDOW_SIZE, 0)
        # Seperate accelerometer data from the other columns
        accelerometer_data_windows = self.data_windows[:, :, len(self.original_columns)-3:]
        # Normalize accelerometer data
        normalized_windows,_ = normalisation.normalise_with_standard_scaler(accelerometer_data_windows, self.scaler)
        # Assign normalized data to self.preprocessed_data
        # this is now ready to be used for classification
        self.preprocessed_data = normalized_windows

    def apply_algorithm(self):
        # Get model predictions
        self.pred_probabilities = self.model.predict(self.preprocessed_data)
        # Convert predicted probabilities to class labels
        self.pred_labels = np.argmax(self.pred_probabilities, axis=1)

    def postprocess(self):
        '''
        In this function, we want to generate an output dataframe
        '''
        # Repeating the rows in self.pred_probabilities and self.pred_labels by window size
        pred_probabilities = np.repeat(self.pred_probabilities, self.WINDOW_SIZE, axis=0)
        pred_labels = np.repeat(self.pred_labels, self.WINDOW_SIZE)
        # reshape data windows
        original_dat = self.data_windows.reshape(-1, len(self.original_columns))
        # get corresponding activity labels
        activity_labels = self.activity_labels[pred_labels]
        # Concatenate arrays side by side
        result = np.column_stack((original_dat, pred_probabilities, pred_labels, activity_labels))
        # Convert to pandas DataFrame
        res_df = pd.DataFrame(result, columns=self.output_columns)
        self.result_df = res_df
import preprocessing
import apply_classification
from get_test_data import *
import glob

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

raw_respeck_folder = ""

def classify_activities():
    pass

def main():
    accel_signal = generate_triaxial_accelerometer_data(N=750, freq=12.5, noise_level=0.2, periodic=True)
    print(accel_signal)

if __name__ == "__main__":
    main()
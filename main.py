import preprocessing
from get_test_data import *

def main():
    accel_signal = generate_triaxial_accelerometer_data(N=750, freq=12.5, noise_level=0.2, periodic=True)
    print(accel_signal)

if __name__ == "__main__":
    main()
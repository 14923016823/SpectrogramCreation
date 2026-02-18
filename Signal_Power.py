### This module uses STFT 2D matrix to comopute the signal power, set noise floor to 0 and eliminate the noise power

import numpy as np

def signal_power(stft_matrix):
    # Compute the power of the signal by taking the magnitude squared of the STFT matrix
    power = np.abs(stft_matrix) ** 2
    
    return power
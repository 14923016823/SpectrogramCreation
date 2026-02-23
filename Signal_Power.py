### This module uses STFT 2D matrix to comopute the signal power, and compute the noise floor.

import numpy as np

def signal_noise_power(stft_matrix):
    # Compute the power of the signal by taking the magnitude squared of the STFT matrix
    power = 10*np.log10(np.abs(stft_matrix) ** 2)
    noise = 0
    sig_power_median = 0
    
    # Parse out the signal from the STFT frame
    for i in range(power.shape[0]):
        power_max = np.max(power[i, :])
        mask = (power[i, :] <= power_max - 13)  # Adjust the threshold as needed
        noise += np.median(power[i, mask])
        sig_power_median += np.median(power[i, ~mask])
    
    # Compute the noise floor as the median of the power values
    noise_floor = noise / power.shape[0]
    sig_power_median = sig_power_median / power.shape[0]
    
    return power, noise_floor, sig_power_median
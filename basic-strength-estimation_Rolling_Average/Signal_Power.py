### This module uses STFT 2D matrix to comopute the signal power, and compute the noise floor.



import numpy as np



def signal_noise_power(stft_matrix):

###############################################################

    # 1. Compute the power of the signal by taking the magnitude squared of the STFT matrix

    power = 10*np.log10(np.abs(stft_matrix) ** 2)

    noise = 0

    signal_power_median = 0

   

    # Parse out the signal from the STFT frame

    for i in range(power.shape[0]):

        power_max = np.max(power[i, :]) #Find the maximum power in the current time frame

        mask = (power[i, :] <= power_max - 10)  # Adjust the threshold as needed

        noise += np.median(power[i, mask])

        signal_power_median += np.median(power[i, ~mask])

   

    # Compute the noise floor as the median of the power values

    noise_floor = noise / power.shape[0]

    signal_power_median = signal_power_median / power.shape[0]

   

    return power, noise_floor, signal_power_median


def new_signal_noise_power(stft_matrix):
    """
    1.Calculates the median of the signal for each bin 
    2. Calculates the power matrix by taking the magnitude squared of the STFT matrix
    3. For each entry in the power matrix the respective noise floor (median of the signal) is subtracted to get the SNR in dB.
    Alternative method to compute signal power and noise floor using a more robust approach.
    """
    

    ###############################################################

    # 1. Compute the power of the signal by taking the magnitude squared of the STFT matrix

    power = np.abs(stft_matrix) ** 2 # shape: (Time, Freq)
    column_medians = np.median(power, axis=0) # This results in the vector with the median of the signal for each freq. bin

    noise = 0

    signal_power_median = 0

    snr = 10*np.log10(power / column_medians) # This results in the SNR matrix in dB for each time-frequency bin

    # Parse out the signal from the STFT frame

    for i in range(power.shape[0]):

        power_max = np.max(power[i, :]) #Find the maximum power in the current time frame

        mask = (power[i, :] <= power_max - 10)  # Adjust the threshold as needed

        noise += np.median(power[i, mask])

        signal_power_median += np.median(power[i, ~mask])

   

    # Compute the noise floor as the median of the power values

    noise_floor = noise / power.shape[0]

    signal_power_median = signal_power_median / power.shape[0]

   

    return snr, noise_floor, signal_power_median
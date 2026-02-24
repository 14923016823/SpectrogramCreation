### This module defines a function to plot the spectrogram of a signal using the 
# Power Spectral Density (PSD) computed from the Short-Time Fourier Transform (STFT).

import matplotlib.pyplot as plt

def plot_spectrogram(power, time, frequency, noise_floor=None, sig_power_median=None):
    """
    Plots the spectrogram of a signal.

    Parameters:
    power (2D np.ndarray): The power spectral density (PSD) matrix obtained from the STFT.
    time (1D np.ndarray): The array of time values corresponding to the rows of the power matrix.
    frequency (1D np.ndarray): The array of frequency values corresponding to the columns of the power matrix.
    noise_floor (float, optional): The noise floor value in dB. Defaults to None.
    sig_power_median (float, optional): The median signal power value in dB. Defaults to None.
    """
    ### Compute the noise floor and set it to 0 in the power matrix
    plt.figure(figsize=(10, 6))
    plt.imshow(power, cmap='viridis', aspect='auto', origin='upper',
               vmin=noise_floor, vmax=sig_power_median,
               extent=[time.min(), time.max(), frequency.max(), frequency.min()])
    plt.colorbar(label='Power/Frequency (dB/Hz)')
    plt.title('Spectrogram')
    plt.ylabel('Time (s)')
    plt.xlabel('Frequency (Hz)')
    plt.show()
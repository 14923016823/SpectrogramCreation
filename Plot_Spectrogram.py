### This module defines a function to plot the spectrogram of a signal using the 
# Power Spectral Density (PSD) computed from the Short-Time Fourier Transform (STFT).

import matplotlib.pyplot as plt
import numpy as np

def plot_spectrogram(power, time, frequency):
    """
    Plots the spectrogram of a signal.

    Parameters:
    power (2D np.ndarray): The power spectral density (PSD) matrix obtained from the STFT.
    time (1D np.ndarray): The array of time values corresponding to the rows of the power matrix.
    frequency (1D np.ndarray): The array of frequency values corresponding to the columns of the power matrix.
    """
    plt.figure(figsize=(10, 6))
    plt.imshow(10 * np.log10(power), cmap='viridis', aspect='auto', origin='upper',
               extent=[time.min(), time.max(), frequency.min(), frequency.max()])
    plt.colorbar(label='Power/Frequency (dB/Hz)')
    plt.title('Spectrogram')
    plt.ylabel('Time (s)')
    plt.xlabel('Frequency (Hz)')
    plt.grid()
    plt.show()
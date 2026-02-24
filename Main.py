from Read_data import read_data
from STFT import stft_band
from Signal_Power import signal_noise_power
from Plot_Spectrogram import plot_spectrogram

#from STFT import stft
#from Plot_Spectrogram import plot_spectrogram

import numpy as np

# Add path of the data file

path = "/home/nziubrys/Linux/GitHub/FFT_DATA/FUNcube-1_39444_202601010247.fc32"

# Define Macros
f_tuning = 145_970_000
f_sampeling = 25_000

# Define read parameters
dtype = np.complex64
read_count = -1

# Call data reading function

signal = read_data(path, dtype = dtype, count = read_count)

# Call STFT function

frame_size = int(8192 / 2)
overlap_size = 2048
stft_matrix, time, frequency = stft_band(signal, frame_size, overlap_size, window_function=np.hanning, f_sampeling=f_sampeling)

# Call signal power function

power, noise_floor, sig_power_median = signal_noise_power(stft_matrix)

# Call spectrogram plotting function

plot_spectrogram(power, time, frequency, noise_floor=noise_floor, sig_power_median=sig_power_median)
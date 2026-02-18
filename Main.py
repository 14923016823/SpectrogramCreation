from Read_data import read_data
from STFT import stft_band
from Signal_Power import signal_power
from Plot_Spectrogram import plot_spectrogram

#from STFT import stft
#from Plot_Spectrogram import plot_spectrogram

import numpy as np
import matplotlib.pyplot as plt

# Add path of the data file

path = "/home/nziubrys/Linux/GitHub/FFT_DATA/FUNcube-1_39444_202601040540.32fc"

# Define Macros
f_tuning = 145_970_000
f_sampeling = 250_000
f_relevant = 15_000

# Define read parameters
dtype = np.complex64
read_count = 20_000_000

# Call data reading function

signal = read_data(path, dtype = dtype, count = read_count)

# Call STFT function

frame_size = 8192
overlap_size = 4048
stft_matrix, time, frequency = stft_band(signal, frame_size, overlap_size, window_function=np.hanning, f_sampeling=f_sampeling, f_relevant=f_relevant)

# Call signal power function

power = signal_power(stft_matrix)

# Call spectrogram plotting function

plot_spectrogram(power, time, frequency)
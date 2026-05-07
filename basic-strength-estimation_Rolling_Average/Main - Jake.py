from Read_data import read_data
from STFT import stft_band
from Signal_Power import signal_noise_power
from Signal_Power import new_signal_noise_power
from Plot_Spectrogram import plot_spectrogram
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
from interp import interp
# from strest import strest
from strest import newstrest


np.set_printoptions(threshold=np.inf)
################################################
# 1. SETUP PATHS
path = r"C:\Users\jacob\Desktop\TUD\2 year\Q3\Project\L0_datasamples\Delfi-C3_32789_202304160811.fc32"
path2 = r"C:\Users\jacob\Desktop\TUD\2 year\Q3\Project\L1B_datasamples\Delfi-C3_32789_202304160811.dat.txt"


###################################################

# 2. Define Macros
f_tuning = 145869000
f_sampeling = 25000
frame_size = 2**10
overlap_size = frame_size // 2

###################################################

# 3. Read the data from the raw file 
dtype = np.complex64

signal = read_data(path, count=-1)

###################################################

# 4. Compute the STFT of the signal
stft_matrix, frequency, time = stft_band(signal, frame_size, overlap_size, window_function=np.hanning, f_sampeling=f_sampeling)

###################################################

# 5. Call signal power function
power, noise_floor, sig_power_median = new_signal_noise_power(stft_matrix)
print(power.shape)
###################################################

# 6. LOAD BEST-FIT DATA
line_data = np.loadtxt(path2, delimiter=None, skiprows=1)
bf_time = line_data[:, 0]
bf_frequency = line_data[:, 1]

###################################################

# 7. Generate the spectrogram base
# Note: Ensure plot_spectrogram arguments match your swap
plot_spectrogram(power, time, frequency, noise_floor=noise_floor, sig_power_median=sig_power_median)


###################################################
noise_floor = noise_floor - 20  # Adjust noise floor for better visualization, if needed
###################################################

# 8. Plotting the best fit scatter points on top of the spectrogram
ax = plt.gca()

timei, frequencyi = interp(time, bf_time, bf_frequency, f_tuning)

ax.scatter(frequencyi, timei, color='red', marker='x', s=10, label='Raw .dat Points')

ax.set_ylim(max(timei), 0) 

ax.set_xlabel("frequency (Hz)")
ax.set_ylabel("time (s)")
ax.legend()
#plt.show() 

###################################################

# SNR CALCULATION AND PRINTING
snr_db = newstrest(stft_matrix, frequency, noise_floor, frequencyi)

snr_smoothed = np.copy(snr_db)
valid_idx = ~np.isnan(snr_db)

if np.sum(valid_idx) > 51: # Ensure we have enough points to filter
    snr_smoothed[valid_idx] = savgol_filter(snr_db[valid_idx], window_length=51, polyorder=3)

# PLOTTING SNR
plt.figure(figsize=(10, 5))
plt.plot(timei, snr_db, color='lightgray', alpha=0.5, label='Raw SNR (Jittery)')
plt.plot(timei, snr_smoothed, color='red', linewidth=1, label='Smoothed SNR')
plt.ylim(min(snr_db)-10, max(snr_db)+10)
plt.title("Satellite Signal Strength (Filtered)")
plt.xlabel("Time (s)")
plt.ylabel("SNR (dB)")
plt.legend()
plt.grid(True, alpha=0.3)
#plt.show()
print("Done plotting SNR")
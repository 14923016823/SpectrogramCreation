from Read_data import read_data
from STFT import stft_band
from Signal_Power import signal_noise_power
from Plot_Spectrogram import plot_spectrogram
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
from interp import interp
from strest import strest


np.set_printoptions(threshold=np.inf)
# 1. SETUP PATHS
path_base = r"C:\Users\karol\Desktop\Project Python Folder\Data Dopptrack"
data_aid = r"\Delfi-C3_32789_201601301014"
path = path_base + data_aid + ".fc32"
path2 = path_base + data_aid + ".dat.txt"

# Define Macros
f_tuning = 145869000
f_sampeling = 25000
frame_size = 2**10
overlap_size = frame_size // 2

# Define read parameters
dtype = np.complex64
read_count = -1

signal = read_data(path, dtype=np.complex64, count=-1)
# --- 2. THE SWAP ---
# Swapping 'time' and 'frequency' here as requested
stft_matrix, frequency, time = stft_band(signal, frame_size, overlap_size, window_function=np.hanning, f_sampeling=f_sampeling)



# Call signal power function
power, noise_floor, sig_power_median = signal_noise_power(stft_matrix)

# 2. READ RAW SIGNAL
#signal = read_data(path, dtype=np.complex64, count=-1)


# 4. LOAD BEST-FIT DATA
line_data = np.loadtxt(path2, delimiter=None, skiprows=1)
bf_time = line_data[:, 0]
bf_frequency = line_data[:, 1]

# 1. Generate the spectrogram base
# Note: Ensure plot_spectrogram arguments match your swap
plot_spectrogram(power, time, frequency, noise_floor=noise_floor, sig_power_median=sig_power_median)

noise_floor = noise_floor - 20  # Adjust noise floor for better visualization, if needed
# 2. Get the current axes
ax = plt.gca()
# 1. SETUP DATA AND AXES
# Get the Doppler path and SNR
timei, frequencyi = interp(time, bf_time, bf_frequency, f_tuning)
snr_db = strest(stft_matrix, frequency, noise_floor, frequencyi)

# 2. SMOOTHING
snr_smoothed = np.copy(snr_db)
valid_idx = ~np.isnan(snr_db)
if np.sum(valid_idx) > 51:
    snr_smoothed[valid_idx] = savgol_filter(snr_db[valid_idx], window_length=51, polyorder=3)

# 3. CREATE SIDE-BY-SIDE PLOT
# figsize 15x8 gives enough room for both. width_ratios 2:1 keeps spectrogram larger.
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8), gridspec_kw={'width_ratios': [2, 1]}, constrained_layout=True)

# --- LEFT PLOT: Spectrogram ---
im = ax1.imshow(power, cmap='viridis', aspect='auto', origin='upper',
                vmin=noise_floor, vmax=sig_power_median,
                extent=[frequency.min(), frequency.max(), time.max(), 0])

ax1.scatter(frequencyi, timei, color='red', marker='x', s=5, label='Best-Fit Path', alpha=0.6)
ax1.set_title(f"Spectrogram: {data_aid.strip('/')}")
ax1.set_xlabel("Frequency (Hz)")
ax1.set_ylabel("Time (s)")
ax1.legend(loc='upper right')

# --- RIGHT PLOT: Signal Strength (SNR) ---
# We swap X and Y here: SNR on X-axis, Time on Y-axis to align with Spectrogram
ax2.plot(snr_db, timei, color='lightgray', alpha=0.4, label='Raw SNR')
ax2.plot(snr_smoothed, timei, color='red', linewidth=2, label='Smoothed SNR')
ax2.set_title("Signal Strength (SNR)")
ax2.set_xlabel("SNR (dB)")
ax2.set_ylabel("Time (s)")
ax2.grid(True, linestyle='--', alpha=0.3)
ax2.legend()

# IMPORTANT: Sync the Y-axis (Time) with the spectrogram
ax2.set_ylim(max(time), 0) 

plt.show()
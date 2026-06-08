from Read_data import read_data
from STFT import stft
from Signal_Power import signal_noise_power
from Plotting import plot_spectrogram, plot_snr
from interp import interp
from strest import strest
from scipy.signal import medfilt, savgol_filter
from BandwidthEstimation import get_bandwidth
import matplotlib.pyplot as plt
import numpy as np

np.set_printoptions(threshold=np.inf)

# --- Paths ---
base = r"C:\Users\glute\Desktop\Project Python Folder\Data Dopptrack"
aid  = r"\Delfi-C3_32789_201512251049"
path_iq  = base + aid + ".fc32"   # raw IQ samples (binary)
path_dat = base + aid + ".dat.txt" # best-fit Doppler curve (time, freq)

# --- Parameters ---
f_tune  = 145_869_000   # receiver tuning frequency [Hz]
f_s     = 25_000        # sampling frequency [Hz]
N       = 2**12         # STFT frame size [samples]
N_ol    = N // 2        # overlap between frames [samples]
hop = N-N_ol 

# --- Load data ---
sig, t_bf, f_bf = read_data(path_iq, path_dat, dtype=np.complex64, count=-1)
# sig  : complex IQ samples
# t_bf : timestamps of best-fit Doppler points [s]
# f_bf : frequencies of best-fit Doppler points [Hz], absolute

# --- Compute spectrogram --- 
t_ax, f_ax, S = stft(sig, N, hop, f_s)
# S    : complex STFT matrix [frames x bins]
# f_ax : frequency axis (baseband, centred at 0) [Hz]
# t_ax : time axis [s]

# --- Power and noise floor ---
pwr, n0 = signal_noise_power(S)
# pwr     : power spectrogram [dB], shape [frames x bins]
# noise   : estimated noise floor [dB]
# sig_med : median power of signal region [dB], used as colormap ceiling

# 1. Baseband conversion
f_bf = f_bf - f_tune

# --- Interpolate best-fit curve onto STFT time axis ---
t_interp, f_interp = interp(t_ax, t_bf, f_bf)
# f_interp: baseband Doppler path at each STFT frame; NaN outside valid segments

bw = get_bandwidth(pwr, f_ax, f_interp) or 1200

# --- Estimate SNR along the Doppler path ---
snr_lin, snr_db = strest(pwr, n0, f_ax, f_interp, bw, frame_size=N)

# --- Smooth SNR (Savitzky-Golay) ---
snr_sm = np.copy(snr_db)
valid = ~np.isnan(snr_db)
if np.sum(valid) > 101:
    snr_sm[valid] = medfilt(snr_db[valid], kernel_size=7)
    snr_sm[valid] = savgol_filter(snr_sm[valid], window_length=101, polyorder=3)

# --- Plot ---
fig, (ax1, ax2) = plt.subplots(2, figsize=(16, 12))

plot_spectrogram(pwr, t_ax, f_ax, ax=ax1)
ax1.plot(f_interp, t_interp, color='red', marker=',', linestyle='none', label='S-Curve')
ax1.set_ylim(max(t_interp), 0)  # flip time axis so t=0 is at top

plot_snr(snr_db, t_ax, snrsmoothed=snr_sm, ax=ax2)

plt.tight_layout()
plt.show()


#------------- TEST -------------#
##################################


# figt, axt = plt.subplots()

# mask = np.isin(t_interp, t_bf, invert=True)  # True where t1 is NOT in t2
# t_filtered = t_interp[mask]
# f_filtered = f_interp[mask]

# axt.plot(f_filtered, t_filtered, color='red', marker=',', linestyle='none', label='S-Curve')

# plt.show()





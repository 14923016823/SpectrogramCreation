from Read_data import read_data
from STFT import stft_band
from Signal_Power import signal_noise_power
from Plotting import plot_spectrogram, plot_snr
from interp import interp
from strest import strest
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import numpy as np

np.set_printoptions(threshold=np.inf)

# --- Paths ---
base = r"C:\Users\glute\Desktop\Project Python Folder\Data Dopptrack"
aid  = r"\Delfi-C3_32789_202003231040"
path_iq  = base + aid + ".fc32"   # raw IQ samples (binary)
path_dat = base + aid + ".dat.txt" # best-fit Doppler curve (time, freq)

# --- Parameters ---
f_tune  = 145_869_000   # receiver tuning frequency [Hz]
f_s     = 25_000        # sampling frequency [Hz]
N       = 2**12         # STFT frame size [samples]
N_ol    = N // 2        # overlap between frames [samples]
bw      = 1200          # integration bandwidth around signal path [Hz]

# --- Load data ---
sig, t_bf, f_bf = read_data(path_iq, path_dat, dtype=np.complex64, count=-1)
# sig  : complex IQ samples
# t_bf : timestamps of best-fit Doppler points [s]
# f_bf : frequencies of best-fit Doppler points [Hz], absolute

# --- Compute spectrogram ---
S, f_ax, t_ax = stft_band(sig, N, N_ol, window_function=np.hanning, f_sampeling=f_s)
# S    : complex STFT matrix [frames x bins]
# f_ax : frequency axis (baseband, centred at 0) [Hz]
# t_ax : time axis [s]

# --- Power and noise floor ---
pwr, noise, sig_med = signal_noise_power(S)
# pwr     : power spectrogram [dB], shape [frames x bins]
# noise   : estimated noise floor [dB]
# sig_med : median power of signal region [dB], used as colormap ceiling

# --- Interpolate best-fit curve onto STFT time axis ---
t_interp, f_interp = interp(t_ax, t_bf, f_bf, f_tune)
# f_interp: baseband Doppler path at each STFT frame; NaN outside valid segments

# --- Estimate SNR along the Doppler path ---
snr = strest(S, f_ax, noise, f_interp, frame_size=N)

# --- Smooth SNR (Savitzky-Golay) ---
snr_sm = np.copy(snr)
valid = ~np.isnan(snr)
if np.sum(valid) > 51:
    snr_sm[valid] = savgol_filter(snr[valid], window_length=51, polyorder=3)

# --- Plot ---
fig, (ax1, ax2) = plt.subplots(2, figsize=(16, 12))

plot_spectrogram(pwr, t_ax, f_ax, noise_floor=noise, sig_power_median=sig_med, ax=ax1)
ax1.plot(f_interp, t_interp, color='red', marker=',', linestyle='none', label='S-Curve')
ax1.set_ylim(max(t_interp), 0)  # flip time axis so t=0 is at top

plot_snr(snr, t_ax, snrsmoothed=snr_sm, ax=ax2)

plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
import numpy as np

def plot_snr(snr, t_ax, snrsmoothed=None, ax=None):
    """Plot raw and optionally smoothed SNR over time."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(t_ax, snr, color='lightgray', alpha=0.5, label='Raw SNR')

    if snrsmoothed is not None:
        ax.plot(t_ax, snrsmoothed, color='red', linewidth=2, label='Smoothed SNR')

    ax.set_ylim(np.nanmin(snr) - 3, np.nanmax(snr) + 3)
    ax.set_title('Signal Strength Estimation')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('SNR (dB)')
    ax.legend()
    ax.grid(True, alpha=0.3)


def plot_spectrogram(pwr, t_ax, f_ax, noise_floor=None, sig_power_median=None, ax=None):
    """Plot power spectrogram with noise floor as colormap floor and signal median as ceiling."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    im = ax.imshow(pwr, cmap='turbo', aspect='auto', origin='upper',
                   vmin=noise_floor, vmax=sig_power_median,
                   extent=[f_ax.min(), f_ax.max(), t_ax.max(), t_ax.min()])

    plt.colorbar(im, ax=ax, label='Power/Frequency (dB/Hz)')
    ax.set_title('Spectrogram')
    ax.set_ylabel('Time (s)')
    ax.set_xlabel('Frequency (Hz)')

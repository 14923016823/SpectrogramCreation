import matplotlib.pyplot as plt
import numpy as np

def plot_snr(snr, t_ax, snrsmoothed=None, ax=None, ref=None):
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

def plot_spectrogram(pwr, t_ax, f_ax, ax=None, cmap='turbo'):
    """Waterfall spectrogram — frequency on x-axis."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    pwr_db = 10 * np.log10(pwr + np.finfo(float).tiny)
    vmin = np.percentile(pwr_db, 60)
    vmax = np.percentile(pwr_db, 99.5)

    im = ax.imshow(pwr_db.T, cmap=cmap, aspect='auto', origin='lower',
                   vmin=vmin, vmax=vmax,
                   extent=[f_ax.min(), f_ax.max(), t_ax.min(), t_ax.max()])

    plt.colorbar(im, ax=ax, label='Power (dB)')
    ax.set_title('Spectrogram')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')
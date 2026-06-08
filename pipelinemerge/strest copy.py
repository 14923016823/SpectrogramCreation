import numpy as np

def strest(S, f_ax, noise_floor, f_path, bw=1200, frame_size=1024):
    """
    Estimate signal power by integrating the STFT along the Doppler path.
    Returns SNR [dB] at each time step; NaN where the path is undefined.
    """
    half_bw = bw / 2

    # Normalise STFT to physical power units (accounts for window coherent gain)
    win  = np.hanning(frame_size)
    gain = np.mean(win)                                      # coherent gain ≈ 0.5 for Hanning
    P    = (np.abs(S) / (frame_size * gain)).T ** 2          # shape: [freq_bins x time_steps]

    n_steps  = len(f_path)
    pwr_path = np.zeros(n_steps)

    for i in range(n_steps):
        fc = f_path[i]

        if np.isnan(fc):
            continue  # skip frames where the Doppler path is undefined

        mask = (f_ax >= fc - half_bw) & (f_ax <= fc + half_bw)  # bins within bandwidth

        if np.any(mask):
            pwr_path[i] = max(0, np.sum(P[mask, i]))  # integrate power in window

    snr = 10 * np.log10(pwr_path + 1e-15)  # convert to dB; small offset avoids log(0)
    snr[np.isnan(f_path)] = np.nan          # restore NaN where path was undefined

    return snr
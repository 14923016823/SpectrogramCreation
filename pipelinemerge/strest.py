import numpy as np

def strest(pwr, n0, f_ax, f_path, bw, frame_size=1024):
    """
    Estimate signal power by integrating the STFT along the Doppler path.
    Returns SNR [dB] at each time step; NaN where the path is undefined.
    """
    if isinstance(bw, tuple):
        offset_lo, offset_hi = bw
    else:
        offset_lo, offset_hi = -bw / 2, bw / 2

    n_steps  = len(f_path)
    snr_linear = np.full(n_steps, np.nan)

    for i in range(n_steps):
        fc = f_path[i]

        if not np.isfinite(fc) or not np.isfinite(n0[i]) or n0[i] <= 0:
            continue

        band = (f_ax >= fc + offset_lo) & (f_ax <= fc + offset_hi)
        if not band.any():
            continue
        excess = pwr[band, i] - n0[i]
        snr_linear[i] = float(np.sum(excess) / (band.sum() * n0[i]))

    with np.errstate(invalid="ignore", divide="ignore"):
        snr_db = 10 * np.log10(np.where(snr_linear > 0, snr_linear, np.nan))


    return snr_linear, snr_db

import numpy as np
from scipy.interpolate import interp1d
 
def interp(time_axis, bf_time, bf_freq, f_tuning):
    """
    Returns (time_axis, freq) arrays of the same length as time_axis.
    Frequency values outside the valid pass window are set to NaN.
    """
    threshold = 7.0  # seconds — max gap in .dat data before treating as invalid
 

    # 1. Baseband conversion
    relative_bf_freq = bf_freq - f_tuning
 

    # 2. Linear interpolation over full time axis
    f_map = interp1d(bf_time, relative_bf_freq, kind='linear', fill_value="extrapolate")
    target_freqs = f_map(time_axis)
 

    # 3. Build valid mask — only True between consecutive bf_time points with gap <= threshold
    valid_mask = np.zeros(len(time_axis), dtype=bool)
    for i in range(1, len(bf_time)):
        dt = bf_time[i] - bf_time[i-1]
        if dt <= threshold:
            segment = (time_axis >= bf_time[i-1]) & (time_axis <= bf_time[i])
            valid_mask[segment] = True
 

    # 4. Set invalid frequencies to NaN
    target_freqs[~valid_mask] = np.nan
 
    return time_axis, target_freqs
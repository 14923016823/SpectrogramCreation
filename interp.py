import numpy as np
from scipy.interpolate import interp1d

def interp(time_axis, bf_time, bf_freq, f_tuning):
    """
    Returns freq array of the exact same length as time_axis.
    Uses NaN for gaps and for any time outside the .dat file range.
    """
    threshold = 20.0  # seconds
    
    # 1. Baseband conversion
    relative_bf_freq = bf_freq - f_tuning
    
    # 2. Linear Interpolation across the high-res STFT grid
    f_map = interp1d(bf_time, relative_bf_freq, kind='linear', fill_value="extrapolate")
    target_freqs = f_map(time_axis)

    # 3. Gap & Domain Validation
    # We initialize a mask as False (everything is invalid by default)
    valid_mask = np.zeros(len(time_axis), dtype=bool)
    
    # Mark segments as valid only if they are within the jump threshold
    for i in range(1, len(bf_time)):
        dt = bf_time[i] - bf_time[i-1]
        if dt <= threshold:
            # Mark the time steps between these two 'truth' points as valid
            segment = (time_axis >= bf_time[i-1]) & (time_axis <= bf_time[i])
            valid_mask[segment] = True
            
    # 4. Final Cleanup
    # Anything not in a 'valid' short segment becomes NaN
    # This automatically handles the "Time Clipping" because valid_mask 
    # can only be True between min(bf_time) and max(bf_time).
    target_freqs[~valid_mask] = np.nan
    
    # Return the full, un-clipped time axis and the NaN-filled freq path
    return time_axis, target_freqs
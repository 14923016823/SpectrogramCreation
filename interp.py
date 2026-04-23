import numpy as np
from scipy.interpolate import interp1d

def interp(time_axis, bf_time, bf_freq, f_tuning):
    """
    Returns (time, freq) arrays clipped strictly to the .dat file's time range.
    """
    threshold = 15.0  # seconds, adjust as needed based on expected gaps in .dat data
    # 1. Baseband conversion
    relative_bf_freq = bf_freq - f_tuning
    
    # 2. Linear Interpolation
    # We use 'extrapolate' here just to ensure we catch the very edges, 
    # but we will manually cut the "tails" off in step 3.
    f_map = interp1d(bf_time, relative_bf_freq, kind='linear', fill_value="extrapolate")
    target_freqs = f_map(time_axis)

    valid_mask = np.zeros(len(time_axis), dtype=bool)
    
    for i in range(1, len(time_axis)):
        if i>=len(bf_time):
            continue

        dt = bf_time[i] - bf_time[i-1]
        
        # If the jump between two points is acceptable, mark that range as valid
        if dt <= threshold:
            segment = (time_axis >= bf_time[i-1]) & (time_axis <= bf_time[i])
            valid_mask[segment] = True
            
    # 4. Set "Invalid" frequencies to NaN
    target_freqs[~valid_mask] = np.nan
    
    # 3. Time Clipping Logic
    # Create a mask that is only True if the STFT time is within the .dat time range
    t_min, t_max = np.min(bf_time), np.max(bf_time)
    time_mask = (time_axis >= t_min) & (time_axis <= t_max)
    
    
        
            

    return time_axis[time_mask], target_freqs[time_mask]

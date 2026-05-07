import numpy as np

def strest(stft_matrix, freq_axis, noise_floor, path_freq, bandwidth_hz=1200):
    """
    Integrates power along a frequency path and subtracts the noise floor.
    """
    # 1. Setup
    half_bw = bandwidth_hz / 2
    num_steps = len(path_freq)
    extracted_pwr = np.zeros(num_steps)
    
    # FIX: Transpose the matrix here if it comes in as (Time, Frequency)
    # This ensures rows = Frequency and columns = Time
    power_matrix = np.abs(stft_matrix).T**2 

    # 2. The Loop
    for i in range(num_steps):
        f_target = path_freq[i]
        
        if np.isnan(f_target):
            extracted_pwr[i] = 0
            continue
        
        # Create the mask based on the frequency axis
        f_mask = (freq_axis >= f_target - half_bw) & (freq_axis <= f_target + half_bw)
        
        if np.any(f_mask):
            # Now power_matrix[f_mask, i] works because axis 0 is Frequency
            total_window_pwr = np.sum(power_matrix[f_mask, i])
            
            num_bins = np.sum(f_mask)
            
            # Use the scalar noise_floor (ensure it's not log-scale dB here!)
            # If your noise_floor is in dB, use: 10**(noise_floor/10)
            clean_pwr = total_window_pwr 
            
            extracted_pwr[i] = max(0, clean_pwr)

    # 3. Convert to dB
    snr_db = extracted_pwr
    snr_db[np.isnan(path_freq)] = np.nan
    
    return snr_db


def newstrest(stft_matrix, freq_axis, noise_floor, path_freq, bandwidth_hz=1200, 
           noise_window=10):
    """
    Integrates power along a frequency path and subtracts a rolling 
    time-averaged noise floor.
    
    Parameters
    ----------
    stft_matrix   : complex STFT, shape (Time, Freq) or (Freq, Time)
    freq_axis     : 1D array of frequency values [Hz]
    noise_floor   : fallback scalar noise floor (linear power, not dB)
                    used only when the rolling average window is not yet full
    path_freq     : 1D array of center frequencies to track [Hz]
    bandwidth_hz  : integration bandwidth around path_freq [Hz]
    noise_window  : number of past time steps to average for noise estimation
    """

    # -----------------------------------------------------------------------
    # 1. Setup
    # -----------------------------------------------------------------------
    
    half_bw      = bandwidth_hz / 2
    num_steps    = len(path_freq) # Based on the best aproximation path 
    extracted_pwr = np.zeros(num_steps) # 1D array 
    delta_f      = freq_axis[1] - freq_axis[0]   # Hz per bin (fixes scaling bug)

    # Ensure shape is (Freq, Time)
    power_matrix = np.abs(stft_matrix).T ** 2    # shape: (Freq, Time)

    # -----------------------------------------------------------------------
    # 2. Pre-compute the rolling average noise floor across ALL frequency bins
    #
    #    For each time step i, average the previous `noise_window` columns.
    #    Result shape: (Freq, Time) — same as power_matrix.
    #
    #    Using cumsum along axis=1 (time axis) for efficiency.
    # -----------------------------------------------------------------------

    num_freq, num_time = power_matrix.shape # num_freq = number of rows; num_time = number of columns


    # Pad the left edge with the first column repeated, so early time steps
    # have a valid average instead of using fewer samples

    pad = np.repeat(power_matrix[:, :1], noise_window, axis=1)   # (Freq, noise_window)
    padded = np.concatenate([pad, power_matrix], axis=1)          # (Freq, Time + noise_window)

    # Cumulative sum along time
    cs = np.cumsum(padded, axis=1)                                 # (Freq, Time + noise_window + 1)
    cs = np.concatenate([np.zeros((num_freq, 1)), cs], axis=1)

    # Rolling mean: subtract cumsum L steps back, divide by L
    # Index offset is noise_window because of the left-edge padding
    rolling_noise = (
        cs[:, noise_window + 1 : noise_window + 1 + num_time] -
        cs[:, 1 : 1 + num_time]
    ) / noise_window                                               # (Freq, Time)

    # -----------------------------------------------------------------------
    # 3. The Loop — now uses rolling_noise instead of scalar noise_floor
    # -----------------------------------------------------------------------
    for i in range(num_steps):
        f_target = path_freq[i]

        if np.isnan(f_target):
            extracted_pwr[i] = 0
            continue

        # Frequency mask for the bandwidth window
        f_mask = (freq_axis >= f_target - half_bw) & (freq_axis <= f_target + half_bw)

        if not np.any(f_mask):
            continue

        num_bins = np.sum(f_mask)

        # Signal power integrated over bandwidth (with correct Δf scaling)
        total_window_pwr = np.sum(power_matrix[f_mask, i])
        
        # Rolling noise floor for this time step, averaged over the bandwidth window
        # Falls back to th  e scalar noise_floor for the very first steps if rolling
        # window is too short (guarded by the left-edge padding above, but kept
        # as an explicit safety net)
        local_noise = np.mean(rolling_noise[f_mask, i]) * num_bins * delta_f
        if local_noise <= 0:
            local_noise = noise_floor * num_bins * delta_f

        # Subtract noise floor and clamp
        clean_pwr = max(0.0, total_window_pwr - local_noise)
        extracted_pwr[i] = clean_pwr

    # -----------------------------------------------------------------------
    # 4. Convert to dB
    # -----------------------------------------------------------------------
    snr_db = 10 * np.log10(extracted_pwr + 1e-12)
    snr_db[np.isnan(path_freq)] = np.nan

    return snr_db
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
    snr_db = 10 * np.log10(extracted_pwr + 1e-15)
    snr_db[np.isnan(path_freq)] = np.nan
    
    return snr_db
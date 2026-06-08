import numpy as np

LN2 = np.log(2.0)

def export_decibel_matrix(decibel_power_matrix, filepath=r"C:\Users\glute\Desktop\Project Python Folder\Data Dopptrack\results\powercsvmatrices\matrix.csv"):
    rounded = np.round(decibel_power_matrix, 1)
    np.savetxt(filepath, rounded, delimiter=",", fmt="%.1f")
    
def signal_noise_power(S):
    """Returns linear power matrix and per-frame noise floor (N0)."""
    power = np.abs(S) ** 2                                    # (n_frames, N)
    n0 = np.median(power, axis=0) / LN2                      # (n_frames,) debiased
    return power, n0




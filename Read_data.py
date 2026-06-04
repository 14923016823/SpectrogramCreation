import numpy as np

def read_data(path_iq, path_dat, dtype=np.complex64, count=2_000_000):
    """Read IQ samples and best-fit Doppler curve from disk."""
    try:
        sig = np.fromfile(path_iq, dtype=dtype, count=count)  # complex IQ samples
        print(f"Read IQ data from {path_iq}")

        dat = np.loadtxt(path_dat, delimiter=None, skiprows=1)
        t_bf = dat[:, 0]  # best-fit timestamps [s]
        f_bf = dat[:, 1]  # best-fit frequencies [Hz], absolute
        print(f"Read line data from {path_dat}")

        return sig, t_bf, f_bf

    except Exception as e:
        print(f"Error reading data: {e}")
        return None

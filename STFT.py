import numpy as np

def stft_band(sig, N, N_ol, window_function=np.hanning, f_sampeling=25_000):
    """Compute the STFT of a complex IQ signal and return the full baseband matrix."""
    try:
        step   = N - N_ol                                   # hop size [samples]
        n_fr   = (len(sig) - N) // step + 1                 # number of frames
        df     = f_sampeling / N                             # frequency resolution [Hz/bin]
        center = N // 2                                      # index of DC bin

        S = np.empty((n_fr, N), dtype=np.complex64)         # output STFT matrix

        try:
            win = window_function(N)
        except Exception:
            print("Invalid window function, defaulting to Hanning.")
            win = np.hanning(N)

        for i in range(n_fr):
            frame = sig[i*step : i*step + N] * win          # window the frame
            S[i, :] = np.fft.fftshift(np.fft.fft(frame))   # FFT, shift DC to centre

        f_ax = (np.arange(N) - center) * df                 # frequency axis [Hz], centred at 0
        t_ax = (np.arange(n_fr) * step + N / 2) / f_sampeling  # time axis [s]

        return S, f_ax, t_ax

    except Exception as e:
        print(f"STFT failed: {e}")
        return None

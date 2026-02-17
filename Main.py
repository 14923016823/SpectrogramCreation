import numpy as np
import matplotlib.pyplot as plt

def read_complex64(path, num_samples=None, offset_samples=0, scale=1.0):
    with open(path, "rb") as f:
        f.seek(offset_samples * np.dtype(np.complex64).itemsize)
        x = np.fromfile(f, dtype=np.complex64, count=num_samples)
    return x * scale

def stft_power_db(x, fs, nperseg=4096, hop=1024):
    #x = x - np.mean(x)  # remove DC
    win = np.hanning(nperseg).astype(np.float32)

    n_frames = 1 + (len(x) - nperseg) // hop
    frames = np.lib.stride_tricks.as_strided(
        x,
        shape=(n_frames, nperseg),
        strides=(x.strides[0] * hop, x.strides[0]),
        writeable=False
    )

    X = np.fft.fft(frames * win, axis=1)
    X = np.fft.fftshift(X, axes=1)

    P = (X.real * X.real + X.imag * X.imag)
    P_db = 10.0 * np.log10(P)

    t = (np.arange(n_frames) * hop) / fs
    f = np.fft.fftshift(np.fft.fftfreq(nperseg, d=1/fs))  # Hz
    return P_db, t, f

def waterfall(P_db, t, f, f_lim=15000, title="Waterfall"):
    # Crop frequency bins to ±f_lim
    idx = np.where((f >= -f_lim) & (f <= f_lim))[0]
    Pc = P_db[:, idx]
    fc = f[idx]

    # Contrast like “team plot”: percentile clipping
    vmin = np.percentile(Pc, 5)
    vmax = np.percentile(Pc, 99.5)

    plt.figure(figsize=(12, 6))
    # Time on y (like your team image), frequency on x
    plt.imshow(
        Pc,
        aspect="auto",
        origin="upper",
        extent=[fc[0], fc[-1], t[-1], t[0]],
        vmin=vmin, vmax=vmax,
        cmap="viridis"
    )
    plt.colorbar(label="Power (dB)")
    plt.xlabel("Frequency [Hz] (relative to center)")
    plt.ylabel("Time [s]")
    plt.title(title)
    plt.tight_layout()
    plt.show()

path = "/home/nziubrys/Linux/GitHub/FFT_DATA/FUNcube-1_39444_202601040540.32fc"
fs = 250000

# Use a longer chunk than 2e6 if your pass is minutes long.
# Or jump to where the pass is using offset_samples.
x = read_complex64(path, num_samples=20_000_000, scale=1.0)

P_db, t, f = stft_power_db(x, fs, nperseg=4*4096, hop=1024)
waterfall(P_db, t, f, f_lim=15000, title="FUNcube waterfall (±15 kHz)")
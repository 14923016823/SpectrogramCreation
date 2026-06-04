import numpy as np

def signal_noise_power(S):
    """Estimate per-frame noise floor and signal power from the STFT matrix."""
    pwr = 10 * np.log10(np.abs(S) ** 2)  # power spectrogram [dB]

    noise_acc = 0   # accumulator for noise floor across frames
    sig_acc   = 0   # accumulator for signal region median across frames

    for i in range(pwr.shape[0]):
        p_max = np.max(pwr[i, :])
        mask  = pwr[i, :] <= p_max - 10  # bins more than 10 dB below peak = noise
        noise_acc += np.median(pwr[i,  mask])
        sig_acc   += np.median(pwr[i, ~mask])

    noise   = noise_acc / pwr.shape[0]   # mean noise floor across all frames [dB]
    sig_med = sig_acc   / pwr.shape[0]   # mean signal median across all frames [dB]

    return pwr, noise, sig_med

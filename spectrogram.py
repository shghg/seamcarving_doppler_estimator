import numpy as np
import librosa

def compute_spectrogram(x, sr, target_sr=1024, nfft=2048, hop=128):
    x_ds = librosa.resample(x, orig_sr=sr, target_sr=target_sr)
    S = np.abs(librosa.stft(x_ds, n_fft=nfft, hop_length=hop))
    freq = np.linspace(0, target_sr/2, S.shape[0])
    time = np.arange(S.shape[1]) * hop / target_sr
    return S, freq, time, x_ds, target_sr

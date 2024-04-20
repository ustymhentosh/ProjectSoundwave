from librosa import load
import numpy as np
from scipy.fftpack import dct


def our_mfcc(audio_file, window_size_ms=25, step_ms=10, n_mfcc=13):
    """
    Extracts Mel-frequency cepstral coefficients (MFCCs) from an audio file.
    Args:
        audio_file (str): Path to the audio file.
        window_size_ms (int, optional): Size of the analysis window in milliseconds. Defaults to 25.
        step_ms (int, optional): Step size between consecutive frames in milliseconds. Defaults to 10.
        n_mfcc (int, optional): Number of MFCCs to extract. Defaults to 13.
    Returns:
        numpy.ndarray: MFCC matrix, where rows represent frames and columns represent MFCC coefficients.
    """
    y, sr = load(audio_file, sr=None)

    # Pre-emphasis
    y = np.append(y[0], y[1:] - 0.97 * y[:-1])
    # Parameters for framing
    frame_length = window_size_ms / 1000  # frame length in seconds
    frame_step = step_ms / 1000  # frame step in seconds
    frame_length_samples = int(sr * frame_length)
    frame_step_samples = int(sr * frame_step)

    # Framing
    num_frames = (
        int(np.ceil(float(np.abs(len(y) - frame_length_samples)) / frame_step_samples))
        + 1
    )
    pad_width = frame_length_samples + frame_step_samples * (num_frames - 1) - len(y)
    y_padded = np.pad(y, (0, pad_width), mode="constant")
    indices = (
        np.tile(np.arange(0, frame_length_samples), (num_frames, 1))
        + np.tile(
            np.arange(0, num_frames * frame_step_samples, frame_step_samples),
            (frame_length_samples, 1),
        ).T
    )
    frames = y_padded[indices].astype(np.float32)

    # Windowing
    frames *= np.hamming(frame_length_samples)

    # FFT and power spectrum
    NFFT = 512
    mag_frames = np.absolute(np.fft.rfft(frames, NFFT))  # Magnitude of the FFT
    pow_frames = (1.0 / NFFT) * ((mag_frames) ** 2)  # Power Spectrum

    # Filter banks
    low_freq_mel = 0
    high_freq_mel = 2595 * np.log10(1 + (sr / 2) / 700)  # Convert Hz to Mel
    mel_points = np.linspace(
        low_freq_mel, high_freq_mel, n_mfcc + 2
    )  # Equally spaced in Mel scale
    hz_points = 700 * (10 ** (mel_points / 2595) - 1)  # Convert Mel to Hz
    bins = np.floor((NFFT + 1) * hz_points / sr)
    fbank = np.zeros((n_mfcc, int(np.floor(NFFT / 2 + 1))))
    for m in range(1, n_mfcc + 1):
        f_m_minus = int(bins[m - 1])
        f_m = int(bins[m])
        f_m_plus = int(bins[m + 1])
        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bins[m - 1]) / (bins[m] - bins[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bins[m + 1] - k) / (bins[m + 1] - bins[m])

    filter_banks = np.dot(pow_frames, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
    log_filter_banks = np.log(filter_banks)

    # DCT
    mfcc = dct(log_filter_banks, type=2, axis=1, norm="ortho")[
        :, 1 : (n_mfcc + 1)
    ]  # Keep 2-13

    # Mean normalization
    mfcc -= np.mean(mfcc, axis=0) + 1e-8

    return mfcc
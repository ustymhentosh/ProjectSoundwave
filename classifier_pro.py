import librosa
import numpy as np
import librosa.display
import json
from sklearn.neighbors import KNeighborsClassifier
import os
from tqdm import tqdm
import numpy as np
from scipy.fftpack import dct

def compute_mfcc(signal, sr, n_fft, hop_length, n_mfcc=13, n_filters=26):
    # Apply pre-emphasis to boost high frequencies
    pre_emphasis = 0.97
    emphasized_signal = np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])

    # Framing
    frame_length, frame_step = n_fft, hop_length
    signal_length = len(emphasized_signal)
    frame_num = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step)) + 1
    pad_signal_length = frame_num * frame_step + frame_length
    z = np.zeros((pad_signal_length - signal_length))
    pad_signal = np.append(emphasized_signal, z)
    indices = np.tile(np.arange(0, frame_length), (frame_num, 1)) + \
              np.tile(np.arange(0, frame_num * frame_step, frame_step), (frame_length, 1)).T
    frames = pad_signal[indices.astype(np.int32, copy=False)]

    # Hamming window
    frames *= np.hamming(frame_length)

    # FFT and power spectrum
    mag_frames = np.absolute(np.fft.rfft(frames, n_fft))
    pow_frames = ((1.0 / n_fft) * (mag_frames ** 2))

    # Mel filterbanks
    low_freq_mel = 0
    high_freq_mel = (2595 * np.log10(1 + (sr / 2) / 700))  # Convert Hz to Mel
    mel_points = np.linspace(low_freq_mel, high_freq_mel, n_filters + 2)  # Equally spaced in Mel scale
    hz_points = (700 * (10**(mel_points / 2595) - 1))  # Convert Mel to Hz
    bin = np.floor((n_fft + 1) * hz_points / sr)
    fbank = np.zeros((n_filters, int(np.floor(n_fft / 2 + 1))))
    for m in range(1, n_filters + 1):
        f_m_minus = int(bin[m - 1])   # left
        f_m = int(bin[m])             # center
        f_m_plus = int(bin[m + 1])    # right

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])

    filter_banks = np.dot(pow_frames, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # Numerical Stability
    log_filter_banks = np.log(filter_banks)

    # DCT to get MFCCs
    mfcc = dct(log_filter_banks, type=2, axis=1, norm='ortho')[:, :n_mfcc]  # Keep 2-13

    return mfcc

# Update the extract_mfcc function to use the custom compute_mfcc function
def extract_mfcc(audio_file, window_size_ms=25, step_ms=10, n_mfcc=13):
    """Extracts Mel-frequency cepstral coefficients (MFCCs) from an audio file.

    Args:
        audio_file (str): Path to the audio file.
        window_size_ms (int, optional): Size of the analysis window in milliseconds. Defaults to 25.
        step_ms (int, optional): Step size between consecutive frames in milliseconds. Defaults to 10.
        n_mfcc (int, optional): Number of MFCCs to extract. Defaults to 13.

    Returns:
        numpy.ndarray: MFCC matrix, where rows represent frames and columns represent MFCC coefficients.
    """
    y, sr = librosa.load(audio_file, sr=None)
    y = np.array([i for i in y if abs(i) > 0.005])
    n_fft = int(sr * window_size_ms / 1000)
    hop_length = int(sr * step_ms / 1000)
    mfccs = compute_mfcc(y, sr, n_fft, hop_length, n_mfcc=n_mfcc)
    return mfccs.transpose()


def standardized_mfcc(mfccs, fixed_length=89):
    """Standardize MFCC array to a fixed length by padding or truncating."""
    # Reshape to ensure mfccs is two-dimensional (1 frame x n coefficients)
    if mfccs.ndim == 1:
        mfccs = mfccs.reshape(1, -1)
    
    current_length = mfccs.shape[0]
    
    if current_length > fixed_length:
        # Truncate if the number of frames exceeds the fixed length
        mfccs = mfccs[:fixed_length, :]
    elif current_length < fixed_length:
        # Pad with zeros if the number of frames is less than the fixed length
        pad_width = fixed_length - current_length
        mfccs = np.pad(mfccs, pad_width=((0, pad_width), (0, 0)), mode='constant')
    
    return mfccs
class UsLeVoModel:
    """
    Model for classification peoplr by voice mfc coeficients
    """

    def __init__(self, k, win_size, step) -> None:
        self.space = KNeighborsClassifier(n_neighbors=k)
        self.w = win_size
        self.s = step

    def _get_majority(self, lst):
        mj = max(set(lst), key=list(lst).count)
        percent = len([1 for i in lst if i == mj]) / len(lst)
        return mj, round(percent, 3)



    def train(self, json_path, rec_dir):
        with open(json_path, "r", encoding="utf-8") as f:
            dct = json.load(f)
        classes = []
        data = []
        for user, value in dct.items():
            for rec_path in value["train"]:
                rec_mfcc = extract_mfcc(rec_dir + os.sep + rec_path, window_size_ms=self.w, step_ms=self.s)
                if rec_mfcc.size == 0:
                    print(f"Warning: MFCC extraction resulted in empty data for {rec_path}")
                    continue
                mean_mfcc = np.mean(rec_mfcc, axis=0).reshape(1, -1)  # Ensure mean_mfcc is 2D
                standardized_features = standardized_mfcc(mean_mfcc, fixed_length=89)
                if standardized_features.shape[1] != mean_mfcc.shape[1]:  # Check feature count consistency
                    print(f"Error: Standardized MFCC feature count mismatch for {rec_path}")
                    continue
                data.append(standardized_features.flatten())  # Flatten to ensure data is 1D per sample
                classes.append(user)

        data = np.array(data)
        if data.size > 0:
            self.space.fit(data, classes)
        else:
            print("No valid data to train on.")




dct_m = {}
dct_f = {}
for k in tqdm([20]):
    for win_size in [40]:
        # m = UsLeVoModel(k, win_size, int(win_size / 2.5))
        # m.train("src/male.json", "dataset/clips")
        # dct_m[f"k={k}, win_size={win_size}"] = m.test("src/male.json", "dataset/clips")
        f = UsLeVoModel(k, win_size, int(win_size / 2.5))
        f.train("src/female.json", "dataset/clips")
        dct_f[f"k={k}, win_size={win_size}"] = f.test("src/female.json", "dataset/clips")
        

# with open("male_results.json", "w", encoding="utf-8") as file:
#     json.dump(dct_m, file)
with open("female_results.json", "w", encoding="utf-8") as file:
    json.dump(dct_f, file)

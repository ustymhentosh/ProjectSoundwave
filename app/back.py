from librosa import load
from librosa.feature import mfcc
from numpy import array
from sklearn.neighbors import KNeighborsClassifier
from os.path import basename
import csv


def extract_mfcc(audio_file, window_size_ms=50, step_ms=30, n_mfcc=13):
    """Extracts Mel-frequency cepstral coefficients (MFCCs) from an audio file.

    Args:
        audio_file (str): Path to the audio file.
        window_size_ms (int, optional): Size of the analysis window in milliseconds. Defaults to 25.
        step_ms (int, optional): Step size between consecutive frames in milliseconds. Defaults to 10.
        n_mfcc (int, optional): Number of MFCCs to extract. Defaults to 13.

    Returns:
        numpy.ndarray: MFCC matrix, where rows represent frames and columns represent MFCC coefficients.
    """
    y, sr = load(audio_file, sr=None)
    y = array([i for i in y if abs(i) > 0.005])

    n_fft = int(sr * window_size_ms / 1000)
    hop_length = int(sr * step_ms / 1000)

    mfccs = mfcc(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mfcc=n_mfcc)
    return mfccs.transpose()


class UsLeVoModel:
    """
    Model for classification peoplr by voice mfc coeficients
    """

    def __init__(self, k=20, win_size=50, step=30) -> None:
        self.space = KNeighborsClassifier(n_neighbors=k)
        self.w = win_size
        self.s = step

    def _get_majority(self, lst):
        mj = max(set(lst), key=list(lst).count)
        percent = len([1 for i in lst if i == mj]) / len(lst)
        return mj, round(percent, 3)

    def train(self, sounds, csv_file):
        dct = {}
        with open(csv_file, "r") as file:
            reader = csv.DictReader(file)
            for row in reader:
                print(row)
                dct[row["recording"]] = row["user"]

        classes = []
        data = []
        for rec_path in sounds:
            file_name = basename(rec_path)
            rec_mfcc = extract_mfcc(rec_path, window_size_ms=self.w, step_ms=self.s)
            for j in rec_mfcc:
                data.append(j)
                classes.append(dct[file_name])
        self.space.fit(data, classes)

    def test(self, target_path):
        rec_mfcc = extract_mfcc(target_path, window_size_ms=self.w, step_ms=self.s)
        prediction, prc = self._get_majority(self.space.predict(rec_mfcc))
        print(prc)
        return prediction

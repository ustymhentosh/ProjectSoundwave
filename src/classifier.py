import librosa
import numpy as np
import librosa.display
import json
from sklearn.neighbors import KNeighborsClassifier
import os
from mfcc import our_mfcc


def lib_mfcc(audio_file, window_size_ms=25, step_ms=10, n_mfcc=13):
    """
    Proxy for mfcc
    """
    y, sr = librosa.load(audio_file, sr=None)

    y = np.array([i for i in y if abs(i) > 0.005])

    n_fft = int(sr * window_size_ms / 1000)
    hop_length = int(sr * step_ms / 1000)

    mfccs = librosa.feature.mfcc(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mfcc=n_mfcc
    )
    return mfccs.transpose()


class UsLeVoModel:
    """
    Model for classification peoplr by voice mfc coeficients
    """

    def __init__(self, k, win_size, step, mtype="lib") -> None:
        self.space = KNeighborsClassifier(n_neighbors=k)
        self.w = win_size
        self.s = step
        if mtype == "lib":
            self.extract_mfcc = lib_mfcc
        else:
            self.extract_mfcc = our_mfcc

    def _get_majority(self, lst):
        mj = max(set(lst), key=list(lst).count)
        percent = len([1 for i in lst if i == mj]) / len(lst)
        return mj, round(percent, 3)

    def train(self, json_path, rec_dir):
        with open(json_path, "r", encoding="utf-8") as f:
            dct = json.load(f)
        classes = []
        data = []
        count = 0
        for user, value in dct.items():
            count += 1
            for rec_path in value["train"]:
                rec_mfcc = self.extract_mfcc(
                    rec_dir + os.sep + rec_path, window_size_ms=self.w, step_ms=self.s
                )
                for j in rec_mfcc:
                    data.append(j)
                    classes.append(user)
        self.space.fit(data, classes)

    def test(self, json_path, rec_dir):
        with open(json_path, "r", encoding="utf-8") as f:
            dct = json.load(f)

        retults = {"right": [], "wrong": []}

        for user, value in dct.items():
            for rec_path in value["test"]:
                rec_mfcc = self.extract_mfcc(
                    rec_dir + os.sep + rec_path, window_size_ms=self.w, step_ms=self.s
                )
                prediction, prc = self._get_majority(self.space.predict(rec_mfcc))
                if prediction == user:
                    retults["right"].append(prc)
                else:
                    retults["wrong"].append(prc)
        return retults

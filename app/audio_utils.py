import librosa
import numpy as np
import io

def extract_features(audio_bytes):
    audio, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000)

    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    features = np.vstack([mfcc, delta, delta2])
    return np.mean(features, axis=1)

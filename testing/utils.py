import numpy as np
import librosa
import joblib
class Utilities():
    
    def __init__(self):
        pass

    def extract_mfcc(self,signal, sr, n_mfcc=40):
        mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)
        return np.concatenate((mfcc_mean, mfcc_std))

    def extract_zcr(self,signal):
        zcr = librosa.feature.zero_crossing_rate(signal)
        return np.array([np.mean(zcr), np.std(zcr)])

    def extract_teo(self,signal):
        teo = np.zeros(len(signal))
        teo[1:-1] = signal[1:-1]**2 - signal[:-2] * signal[2:]
        return np.array([np.mean(teo), np.std(teo)])

    def extract_hnr(self,signal, sr):
        f0, voiced_flag, voiced_probs = librosa.pyin(signal, fmin=50, fmax=300)
        harmonics = f0[~np.isnan(f0)]
        if len(harmonics) > 0:
            hnr = 10 * np.log10(np.var(harmonics) / (np.var(signal) + 1e-6))
        else:
            hnr = 0.0  # default if pitch not found
        return np.array([hnr])
    
    def feature_extraction_and_concatenation(self,filepath):
        signal, sr = librosa.load(filepath, sr=None)

        # Optional: trim silence
        signal, _ = librosa.effects.trim(signal)

        features = []
        features.extend(self.extract_mfcc(signal, sr))
        features.extend(self.extract_zcr(signal))
        features.extend(self.extract_teo(signal))
        features.extend(self.extract_hnr(signal, sr))
        
        return np.array(features)
    
    def scaling_values(self,audio_features):
        scaling_model=joblib.load(r'D:\Projects\MoodMate\paper_code\models\scaler.pkl')

        audio_features = np.array(audio_features).reshape(1, -1)

        scaled_features = scaling_model.transform(audio_features)

        return scaled_features
    

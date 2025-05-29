import librosa
import numpy as np
import joblib
import os

def extract_mfcc(file_path):
    try:
        # Load audio file
        y, sr = librosa.load(file_path, sr=22050)  # standard sample rate
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfccs_processed = np.mean(mfccs.T, axis=0)  # shape: (40,)
        return mfccs_processed
    except Exception as e:
        print(f"Error processing audio: {e}")
        return None

def predict_emotion(audio_path):
    try:
        features = extract_mfcc(audio_path)
        if features is None:
            return "Could not process audio"

        # Load the model
        model = joblib.load("emotion_model.pkl")  # make sure this exists
        prediction = model.predict([features])[0]
        return prediction
    except Exception as e:
        print(f"Prediction error: {e}")
        return "Prediction failed"
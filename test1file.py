import librosa
import numpy as np
import joblib  # Only if your model is a .pkl (scikit-learn)

# Step 1: Load the audio
y, sr = librosa.load("angry.wav", duration=3, offset=0.5)

# Step 2: Extract features
mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
mfcc_scaled = mfcc.mean(axis=1).reshape(1, -1)

# Step 3: Load your trained model
model = joblib.load("model.pkl")  # or replace with keras load if .h5

# Step 4: Predict
prediction = model.predict(mfcc_scaled)
print("Predicted Emotion:", prediction[0])
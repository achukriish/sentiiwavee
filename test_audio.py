import librosa
import numpy as np
import joblib

# Load the saved model
model = joblib.load('emotion_model.pkl')

# Path to your test audio file
audio_path = 'test_audio.wav'  # <-- Make sure this matches your audio filename

# Function to extract MFCC features from audio
def extract_features(file_name):
    try:
        audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
    except Exception as e:
        print(f"Error loading audio file: {e}")
        return None
    try:
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
        mfccs_scaled = np.mean(mfccs.T, axis=0)
        return mfccs_scaled
    except Exception as e:
        print(f"Feature extraction failed: {e}")
        return None

# Extract features from test audio
features = extract_features(audio_path)

if features is not None:
    # Reshape features for prediction (1 sample, n features)
    features_reshaped = features.reshape(1, -1)

    # Predict emotion
    prediction = model.predict(features_reshaped)

    print(f"Predicted Emotion: {prediction[0]}")
else:
    print("Cannot predict emotion due to previous errors.")


import os
import librosa
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Step 1: Define emotion labels
emotion_map = {
    '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
    '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'
}

X = []
y = []

# Step 2: Loop through all audio files
data_path = r"C:\Users\ARCHANA\Desktop\audiofiles"
for folder in os.listdir(data_path):
    folder_path = os.path.join(data_path, folder)
    if os.path.isdir(folder_path):
        for file in os.listdir(folder_path):
            if file.endswith(".wav"):
                emotion_code = file.split("-")[2]
                if emotion_code in emotion_map:
                    emotion = emotion_map[emotion_code]
                    file_path = os.path.join(folder_path, file)
                    audio, sr = librosa.load(file_path, duration=3, offset=0.5)
                    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
                    mfcc_scaled = np.mean(mfcc.T, axis=0)
                    X.append(mfcc_scaled)
                    y.append(emotion)

# Step 3: Train a simple model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Step 4: Save model
joblib.dump(model, "model.pkl")
print("✅ Model trained and saved as model.pkl")

# Step 5: Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("🎯 Accuracy:", acc)



import numpy as np
import joblib
from sklearn.svm import SVC

# Dummy training data (X = MFCC features, y = emotions)
X = np.random.rand(100, 40)  # 100 audio samples with 40 MFCC features
y = np.random.choice(['Happy', 'Sad', 'Angry', 'Neutral', 'Calm'], 100)

# Train model
model = SVC()
model.fit(X, y)

# Save the model
joblib.dump(model, 'emotion_model.pkl')
print("Model saved as emotion_model.pkl")

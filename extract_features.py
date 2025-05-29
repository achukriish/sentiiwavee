print("Extract Features Script Started")


import os
import librosa
import numpy as np
import csv

# Your dataset folder
dataset_path = r"C:\Users\ARCHANA\Desktop\audiofiles"

# Where to save the CSV file
csv_filename = "features.csv"

# Create or overwrite CSV file
with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    # Write header
    writer.writerow(['mfcc'] + ['label'])

    # Loop through each actor folder
    for actor_folder in os.listdir(dataset_path):
        actor_path = os.path.join(dataset_path, actor_folder)

        # Make sure it's a folder
        if os.path.isdir(actor_path):
            for filename in os.listdir(actor_path):
                if filename.endswith(".wav"):
                    file_path = os.path.join(actor_path, filename)

                    # Load audio
                    audio, sr = librosa.load(file_path, duration=3, offset=0.5)

                    # Extract MFCC
                    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
                    mfccs_processed = np.mean(mfccs.T, axis=0)

                    # Get emotion label from filename (3rd value)
                    emotion_code = int(filename.split("-")[2])
                    emotion_map = {
                        1: "neutral", 2: "calm", 3: "happy", 4: "sad",
                        5: "angry", 6: "fearful", 7: "disgust", 8: "surprised"
                    }
                    label = emotion_map.get(emotion_code, "unknown")

                    # Write to CSV
                    writer.writerow([mfccs_processed.tolist(), label])

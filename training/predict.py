import os
import sys
from unittest import result

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import joblib
import numpy as np
import librosa

from preprocessing.audio_loader import load_audio

# ==================================================
# Load Trained Model and Label Encoder
# ==================================================

BASE_DIR = os.path.dirname(__file__)

MODEL_PATH = os.path.join(BASE_DIR, "random_forest_model.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "label_encoder.pkl")

model = joblib.load(MODEL_PATH)
label_encoder = joblib.load(ENCODER_PATH)


# ==================================================
# Prediction Function
# ==================================================

def predict_audio(audio_path):

    # Load audio
    audio, sample_rate = load_audio(audio_path)

    if audio is None:
        print("Failed to load audio.")
        return

    # Extract MFCC
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=40
    )

    print("\nMFCC Shape:", mfcc.shape)

    # Convert MFCC into fixed-size feature vector
    feature_vector = np.mean(mfcc, axis=1)

    print("Feature Vector Shape:", feature_vector.shape)

    # Reshape for sklearn
    feature_vector = feature_vector.reshape(1, -1)

    # Predict class
    prediction = model.predict(feature_vector)

    # Predict probabilities
    probabilities = model.predict_proba(feature_vector)

    # Decode predicted label
    animal = label_encoder.inverse_transform(prediction)

    # Confidence
    confidence = np.max(probabilities) * 100

    # ===============================
    # Display Results
    # ===============================

    print("\n" + "=" * 45)
    print("Prediction Result")
    print("=" * 45)

    result = {
    "animal": animal[0],
    "confidence": float(confidence)
}

    return result
    print(f"Confidence        : {confidence:.2f}%")

    print("\nClass Probabilities")
    print("-" * 30)

    for label, prob in zip(label_encoder.classes_, probabilities[0]):
        print(f"{label:<10} : {prob*100:.2f}%")

    print("=" * 45)


# ==================================================
# Main
# ==================================================

if __name__ == "__main__":

    PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

    audio_path = os.path.join(
        PROJECT_ROOT,
        "dataset",
        "raw",
        "dog",
        "dog1.wav"
    )

    predict_audio(audio_path)
    
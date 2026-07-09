import os
import joblib
import librosa
import numpy as np

from django.conf import settings
from django.shortcuts import render

# ==========================
# Load ML Model
# ==========================
MODEL_PATH = os.path.join(
    settings.BASE_DIR,
    "ml_models",
    "segmented_random_forest.pkl"
)

LABEL_PATH = os.path.join(
    settings.BASE_DIR,
    "ml_models",
    "segmented_label_encoder.pkl"
)

model = joblib.load(MODEL_PATH)
label_encoder = joblib.load(LABEL_PATH)


# ==========================
# Home Page
# ==========================
def home(request):
    return render(request, "predictor/home.html")


# ==========================
# Upload + Prediction
# ==========================
def upload(request):

    if request.method == "POST":

        uploaded_file = request.FILES.get("audio")

        if uploaded_file:

            # Create Upload Folder
            upload_folder = os.path.join(settings.MEDIA_ROOT, "uploads")
            os.makedirs(upload_folder, exist_ok=True)

            save_path = os.path.join(upload_folder, uploaded_file.name)

            # Save uploaded audio
            with open(save_path, "wb+") as destination:
                for chunk in uploaded_file.chunks():
                    destination.write(chunk)

            # Load Audio
            audio, sample_rate = librosa.load(save_path, sr=22050)

            # Extract MFCC
            mfcc = librosa.feature.mfcc(
                y=audio,
                sr=sample_rate,
                n_mfcc=40
            )

            feature_vector = np.mean(mfcc, axis=1).reshape(1, -1)

            # Predict
            predicted_class = model.predict(feature_vector)[0]

            prediction = label_encoder.inverse_transform(
                [predicted_class]
            )[0]

            probabilities = model.predict_proba(feature_vector)[0]

            confidence = round(np.max(probabilities) * 100, 2)

            # Select Icon
            if prediction == "cat":
                icon = "emoji-smile"

            elif prediction == "dog":
                icon = "shield-fill-check"

            elif prediction == "cow":
                icon = "award-fill"

            else:
                icon = "question-circle"

            # Data for Template
            result = {
                "animal": prediction.title(),
                "confidence": confidence,
                "icon": icon,
                "scientific_name": "Recognized using Random Forest Classifier",
                "audio_url": settings.MEDIA_URL + "uploads/" + uploaded_file.name,
            }

            return render(
                request,
                "predictor/result.html",
                {
                    "result": result
                }
            )

    return render(request, "predictor/upload.html")


# ==========================
# History Page
# ==========================
def history(request):
    return render(request, "predictor/history.html")


# ==========================
# About Page
# ==========================
def about(request):
    return render(request, "predictor/about.html")
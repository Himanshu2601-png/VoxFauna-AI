from .models import PredictionHistory
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

        print("POST REQUEST RECEIVED")

        uploaded_file = request.FILES.get("audio")

        print("Uploaded File:", uploaded_file)

        if uploaded_file:

            print("File found")

            upload_folder = os.path.join(settings.MEDIA_ROOT, "uploads")
            os.makedirs(upload_folder, exist_ok=True)

            save_path = os.path.join(upload_folder, uploaded_file.name)

            with open(save_path, "wb+") as destination:
                for chunk in uploaded_file.chunks():
                    destination.write(chunk)

            print("Saved:", save_path)

            audio, sample_rate = librosa.load(save_path, sr=22050)

            mfcc = librosa.feature.mfcc(
                y=audio,
                sr=sample_rate,
                n_mfcc=40
            )

            feature_vector = np.mean(mfcc, axis=1).reshape(1, -1)

            predicted_class = model.predict(feature_vector)[0]

            prediction = label_encoder.inverse_transform([predicted_class])[0]

            probabilities = model.predict_proba(feature_vector)[0]

            confidence = round(np.max(probabilities) * 100, 2)

           

            return render(request, "predictor/result.html", {
                "prediction": prediction,
                "confidence": confidence,
                "uploaded_filename": uploaded_file.name,
                "animal_image": f"images/{prediction}.jpg",
            })

    print("Returning upload page")

    return render(request, "predictor/upload.html")


# ==========================
# History Page
# ==========================
from .models import PredictionHistory


def history(request):

    history = PredictionHistory.objects.order_by("-uploaded_at")

    return render(
        request,
        "predictor/history.html",
        {
            "prediction_history": history,
            "total_predictions": history.count(),
            "unique_species": history.values("prediction").distinct().count(),
            "avg_confidence": round(
                sum(x.confidence for x in history) / history.count(),
                2
            ) if history.count() else 0,
            "last_upload": history.first().uploaded_at.strftime("%d %b %Y")
            if history.exists()
            else "No Upload"
        }
    )


# ==========================
# About Page
# ==========================
def about(request):
    return render(request, "predictor/about.html")
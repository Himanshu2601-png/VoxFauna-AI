import os
import joblib
import numpy as np
import soundfile as sf

from python_speech_features import mfcc

from django.conf import settings
from django.shortcuts import render

from .models import PredictionHistory

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

        if uploaded_file:

            print("Uploaded:", uploaded_file.name)

            upload_folder = os.path.join(settings.MEDIA_ROOT, "uploads")
            os.makedirs(upload_folder, exist_ok=True)

            save_path = os.path.join(upload_folder, uploaded_file.name)

            with open(save_path, "wb+") as destination:
                for chunk in uploaded_file.chunks():
                    destination.write(chunk)

            print("Saved:", save_path)

            try:
                print("Loading audio...")

                audio, sample_rate = sf.read(save_path)

                # Stereo -> Mono
                if len(audio.shape) > 1:
                    audio = np.mean(audio, axis=1)

                print("Extracting MFCC...")

                mfcc_features = mfcc(
                    signal=audio,
                    samplerate=sample_rate,
                    numcep=40
                )

                feature_vector = np.mean(
                    mfcc_features,
                    axis=0
                ).reshape(1, -1)

                print("Predicting...")

                predicted_class = model.predict(feature_vector)[0]

                prediction = label_encoder.inverse_transform(
                    [predicted_class]
                )[0]

                probabilities = model.predict_proba(feature_vector)[0]

                confidence = round(
                    np.max(probabilities) * 100,
                    2
                )

                print("Prediction:", prediction)

                # Save history
                PredictionHistory.objects.create(
                    filename=uploaded_file.name,
                    prediction=prediction,
                    confidence=confidence
                )

                return render(
                    request,
                    "predictor/result.html",
                    {
                        "prediction": prediction,
                        "confidence": confidence,
                        "uploaded_filename": uploaded_file.name,
                        "animal_image": f"images/{prediction}.jpg",
                    }
                )

            except Exception as e:
                print("ERROR:", e)

                return render(
                    request,
                    "predictor/upload.html",
                    {
                        "error": str(e)
                    }
                )

    return render(request, "predictor/upload.html")


# ==========================
# History
# ==========================

def history(request):

    history = PredictionHistory.objects.order_by("-uploaded_at")

    return render(
        request,
        "predictor/history.html",
        {
            "prediction_history": history,
            "total_predictions": history.count(),
            "unique_species": history.values(
                "prediction"
            ).distinct().count(),
            "avg_confidence": round(
                sum(x.confidence for x in history) / history.count(),
                2
            ) if history.count() else 0,
            "last_upload": history.first().uploaded_at.strftime("%d %b %Y")
            if history.exists()
            else "No Upload",
        },
    )


# ==========================
# About
# ==========================

def about(request):
    return render(request, "predictor/about.html")
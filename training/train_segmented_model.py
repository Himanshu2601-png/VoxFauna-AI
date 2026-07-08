from prepare_segmented_dataset import prepare_segmented_dataset

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import joblib


def train_model():

    # Load Dataset
    X, y = prepare_segmented_dataset()

    print("\nDataset Loaded Successfully!")
    print("Features :", X.shape)
    print("Labels   :", y.shape)

    # Encode Labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    print("\nClasses :", label_encoder.classes_)

    # Split Dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded
    )

    print("\nTraining Samples :", X_train.shape[0])
    print("Testing Samples  :", X_test.shape[0])

    # Train Model
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )

    print("\nTraining Model...")
    model.fit(X_train, y_train)

    print("Training Completed!")

    # Prediction
    y_pred = model.predict(X_test)

    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)

    print("\n" + "=" * 50)
    print("MODEL PERFORMANCE")
    print("=" * 50)

    print(f"Accuracy : {accuracy * 100:.2f}%")

    print("\nClassification Report")
    print(classification_report(
        y_test,
        y_pred,
        target_names=label_encoder.classes_,
        zero_division=0
    ))

    print("\nConfusion Matrix")
    print(confusion_matrix(y_test, y_pred))

    # Save Model
    joblib.dump(model, "segmented_random_forest.pkl")
    joblib.dump(label_encoder, "segmented_label_encoder.pkl")

    print("\nModel Saved Successfully!")
    print("segmented_random_forest.pkl")
    print("segmented_label_encoder.pkl")


if __name__ == "__main__":
    train_model()
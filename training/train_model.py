from prepare_dataset import prepare_dataset

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import joblib


def train_model():
     X,y = prepare_dataset()
     print(X.shape)
     print(y.shape)
     
     label_encoder = LabelEncoder()
     y_encoded = label_encoder.fit_transform(y)
     print("classes:", label_encoder.classes_)
     print("Encoded Labels:", y_encoded[:10])
     
     
     X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
     
    )
     
     print("Training set shape:", X_train.shape, y_train.shape)
     print("Testing set shape:", X_test.shape, y_test.shape)
     
     model = RandomForestClassifier(n_estimators=100, random_state=42)
     model.fit(X_train, y_train)
     
     y_pred = model.predict(X_test)
     
     accuracy = accuracy_score(y_test, y_pred)
     print(f"Test Accuracy: {accuracy * 100:.2f}%")
     
     print("\nClassification Report:")
     print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
     
     print("\nConfusion Matrix:")
     print(confusion_matrix(y_test, y_pred))
     
     joblib.dump(model, "random_forest_model.pkl")
     joblib.dump(label_encoder, "label_encoder.pkl")
     
     
     
     
if __name__ == "__main__":
    train_model()
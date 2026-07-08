import os
import numpy as np


def prepare_dataset():
    """
    Prepare the dataset for machine learning.

    Returns:
        X (numpy.ndarray): Feature matrix
        y (numpy.ndarray): Labels
    """

    X = []
    y = []

    processed_path = "../dataset/processed"

    print("=" * 50)
    print("Preparing Dataset...")
    print("=" * 50)

    for animal in os.listdir(processed_path):

        animal_folder = os.path.join(processed_path, animal)

        if not os.path.isdir(animal_folder):
            continue

        print(f"\nProcessing Animal: {animal}")

        for file in os.listdir(animal_folder):

            if not file.endswith(".npy"):
                continue

            input_path = os.path.join(animal_folder, file)

            # Load MFCC features
            mfcc_features = np.load(input_path)

            # Convert (40, N) -> (40,)
            feature_vector = np.mean(mfcc_features, axis=1)

            X.append(feature_vector)
            y.append(animal)

    # Convert lists to NumPy arrays AFTER processing all files
    X = np.array(X)
    y = np.array(y)

    print("\n" + "=" * 50)
    print("Dataset Prepared Successfully!")
    print("=" * 50)
    print("Feature Shape :", X.shape)
    print("Label Shape   :", y.shape)

    return X, y


if __name__ == "__main__":
    X, y = prepare_dataset()
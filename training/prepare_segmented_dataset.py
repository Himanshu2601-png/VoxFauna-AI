import os
import numpy as np

def prepare_segmented_dataset():

    X = []
    y = []

    processed_path = "../dataset/processed_segmented"

    print("=" * 50)
    print("Preparing Segmented Dataset...")
    print("=" * 50)

    for animal in os.listdir(processed_path):

        animal_folder = os.path.join(processed_path, animal)

        if not os.path.isdir(animal_folder):
            continue

        print(f"\nProcessing Animal : {animal}")

        for file in os.listdir(animal_folder):

            if not file.endswith(".npy"):
                continue

            file_path = os.path.join(animal_folder, file)

            mfcc = np.load(file_path)

            # Convert (40, time_frames) → (40,)
            feature_vector = np.mean(mfcc, axis=1)

            X.append(feature_vector)
            y.append(animal)

    X = np.array(X)
    y = np.array(y)

    print("\n" + "=" * 50)
    print("Segmented Dataset Ready!")
    print("=" * 50)
    print("Feature Shape :", X.shape)
    print("Label Shape   :", y.shape)

    return X, y


if __name__ == "__main__":
    X, y = prepare_segmented_dataset()
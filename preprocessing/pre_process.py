import os

from audio_loader import load_audio
from mfcc import extract_mfcc

# Dataset paths
RAW_DATASET = "../dataset/raw"
PROCESSED_DATASET = "../dataset/processed"


def preprocess_dataset():
    """
    Process all audio files in the dataset and save their MFCC features.
    """

    print("=" * 60)
    print("Starting Dataset Preprocessing...")
    print("=" * 60)

    # Iterate over each animal folder
    for animal in os.listdir(RAW_DATASET):

        animal_folder = os.path.join(RAW_DATASET, animal)

        if not os.path.isdir(animal_folder):
            continue

        print(f"\nProcessing Animal : {animal}")

        # Create processed folder
        processed_folder = os.path.join(PROCESSED_DATASET, animal)
        os.makedirs(processed_folder, exist_ok=True)

        # Process every audio file
        for file in os.listdir(animal_folder):

            if not file.endswith(".wav"):
                continue

            input_path = os.path.join(animal_folder, file)

            output_file = os.path.splitext(file)[0] + ".npy"

            output_path = os.path.join(processed_folder, output_file)

            print(f"\nReading : {file}")

            audio, sample_rate = load_audio(input_path)

            if audio is None:
                print("Skipping corrupted file...")
                continue

            extract_mfcc(audio, sample_rate, output_path)

            print("Saved :", output_path)

    print("\n" + "=" * 60)
    print("Dataset Preprocessing Completed Successfully!")
    print("=" * 60)


if __name__ == "__main__":
    preprocess_dataset()
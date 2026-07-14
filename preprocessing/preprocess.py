import os

from audio_loader import load_audio
from mfcc import extract_mfcc

RAW_DATASET = "../dataset/raw"
PROCESSED_DATASET = "../dataset/processed"


def preprocess_dataset():

    print("=" * 60)
    print("Starting Dataset Preprocessing...")
    print("=" * 60)

    for animal in os.listdir(RAW_DATASET):

        animal_folder = os.path.join(RAW_DATASET, animal)

        if not os.path.isdir(animal_folder):
            continue

        print(f"\nProcessing {animal}")

        processed_folder = os.path.join(PROCESSED_DATASET, animal)
        os.makedirs(processed_folder, exist_ok=True)

        for file in os.listdir(animal_folder):

            if not file.lower().endswith(".wav"):
                continue

            input_path = os.path.join(animal_folder, file)

            output_path = os.path.join(
                processed_folder,
                os.path.splitext(file)[0] + ".npy"
            )

            print("Reading:", input_path)

            audio, sample_rate = load_audio(input_path)

            if audio is None:
                print("Skipped.")
                continue

            # Convert stereo to mono
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)

            extract_mfcc(
                audio,
                sample_rate,
                output_path
            )

            print("Saved:", output_path)

    print("\nDataset preprocessing completed.")


if __name__ == "__main__":
    preprocess_dataset()
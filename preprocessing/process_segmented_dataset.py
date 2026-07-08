import os
import numpy as np
import librosa

# ==========================================
# Configuration
# ==========================================

INPUT_PATH = "../dataset/segmented"
OUTPUT_PATH = "../dataset/processed_segmented"

N_MFCC = 40

print("=" * 60)
print("PROCESSING SEGMENTED DATASET")
print("=" * 60)

total_processed = 0

# ==========================================
# Process every animal
# ==========================================

for animal in os.listdir(INPUT_PATH):

    animal_input = os.path.join(INPUT_PATH, animal)

    if not os.path.isdir(animal_input):
        continue

    animal_output = os.path.join(OUTPUT_PATH, animal)
    os.makedirs(animal_output, exist_ok=True)

    print(f"\nProcessing Animal : {animal}")

    for file in os.listdir(animal_input):

        if not file.endswith(".wav"):
            continue

        input_file = os.path.join(animal_input, file)

        try:
            # Load audio
            audio, sample_rate = librosa.load(
                input_file,
                sr=22050
            )

            # Extract MFCC
            mfcc = librosa.feature.mfcc(
                y=audio,
                sr=sample_rate,
                n_mfcc=N_MFCC
            )

            # Save MFCC
            filename = os.path.splitext(file)[0] + ".npy"

            output_file = os.path.join(
                animal_output,
                filename
            )

            np.save(output_file, mfcc)

            total_processed += 1

            print(f"Saved : {filename}")

        except Exception as e:

            print(f"Error processing {file}")
            print(e)

print("\n" + "=" * 60)
print("PROCESS COMPLETED")
print("=" * 60)
print(f"Total MFCC Files Created : {total_processed}")

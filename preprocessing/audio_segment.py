import os
import librosa
import soundfile as sf

# ==========================
# Configuration
# ==========================

INPUT_DATASET = "../dataset/raw"
OUTPUT_DATASET = "../dataset/segmented"

SAMPLE_RATE = 22050
SEGMENT_DURATION = 3  # seconds
SEGMENT_SAMPLES = SAMPLE_RATE * SEGMENT_DURATION

print("=" * 60)
print("AUDIO SEGMENTATION STARTED")
print("=" * 60)

total_segments = 0

# ==========================
# Process each animal folder
# ==========================

for animal in os.listdir(INPUT_DATASET):

    animal_input_folder = os.path.join(INPUT_DATASET, animal)

    if not os.path.isdir(animal_input_folder):
        continue

    animal_output_folder = os.path.join(OUTPUT_DATASET, animal)
    os.makedirs(animal_output_folder, exist_ok=True)

    print(f"\nProcessing Animal : {animal}")

    for file in os.listdir(animal_input_folder):

        if not file.endswith(".wav"):
            continue

        input_path = os.path.join(animal_input_folder, file)

        try:
            audio, sample_rate = librosa.load(
                input_path,
                sr=SAMPLE_RATE
            )

            filename = os.path.splitext(file)[0]

            segment_number = 1

            for start in range(0, len(audio), SEGMENT_SAMPLES):

                end = start + SEGMENT_SAMPLES

                segment = audio[start:end]

                # Skip very short segments
                if len(segment) < SEGMENT_SAMPLES:
                    continue

                output_filename = f"{filename}_{segment_number}.wav"

                output_path = os.path.join(
                    animal_output_folder,
                    output_filename
                )

                sf.write(output_path, segment, SAMPLE_RATE)

                print(f"Saved : {output_filename}")

                segment_number += 1
                total_segments += 1

        except Exception as e:
            print(f"Error processing {file}")
            print(e)

print("\n" + "=" * 60)
print("SEGMENTATION COMPLETED")
print("=" * 60)
print(f"Total Segments Created : {total_segments}")
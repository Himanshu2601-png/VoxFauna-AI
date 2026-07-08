import os
import librosa

dataset_path = "../dataset/raw"

print("=" * 50)
print("DATASET ANALYSIS")
print("=" * 50)

total_files = 0
total_duration = 0

shortest_duration = float("inf")
longest_duration = 0

shortest_file = ""
longest_file = ""

for animal in os.listdir(dataset_path):

    animal_folder = os.path.join(dataset_path, animal)

    if not os.path.isdir(animal_folder):
        continue

    print(f"\nAnimal : {animal}")

    file_count = 0

    for file in os.listdir(animal_folder):

        if file.endswith(".wav"):

            file_path = os.path.join(animal_folder, file)

            try:
                audio, sample_rate = librosa.load(file_path, sr=22050)

                duration = len(audio) / sample_rate

                print(f"{file:<15} {duration:.2f} sec")

                total_duration += duration
                total_files += 1
                file_count += 1

                if duration < shortest_duration:
                    shortest_duration = duration
                    shortest_file = file

                if duration > longest_duration:
                    longest_duration = duration
                    longest_file = file

            except Exception as e:
                print(f"Could not read {file}")
                print(e)

    print(f"Number of Files : {file_count}")

print("\n" + "=" * 50)
print("DATASET SUMMARY")
print("=" * 50)

print(f"Total Files      : {total_files}")

if total_files > 0:
    average_duration = total_duration / total_files

    print(f"Average Duration : {average_duration:.2f} sec")
    print(f"Shortest Audio   : {shortest_file} ({shortest_duration:.2f} sec)")
    print(f"Longest Audio    : {longest_file} ({longest_duration:.2f} sec)")
import os
import matplotlib.pyplot as plt
import librosa.display

from audio_loader import load_audio


def plot_waveform(audio, sample_rate, save_path):
    """
    Plot and save waveform.
    """

    plt.figure(figsize=(12, 4))

    librosa.display.waveshow(audio, sr=sample_rate)

    plt.title("Waveform of Audio")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.grid(True)

    plt.savefig(save_path)

    plt.show()

    plt.close()


if __name__ == "__main__":

    audio_path = "../dataset/raw/dog/dog1.wav"

    output_path = "../outputs"

    os.makedirs(output_path, exist_ok=True)

    audio, sample_rate = load_audio(audio_path)

    if audio is not None:
        save_file = os.path.join(output_path, "dog1_waveform.png")
        plot_waveform(audio, sample_rate, save_file)

        print(f"\nWaveform saved successfully!")
        print(f"Location : {save_file}")
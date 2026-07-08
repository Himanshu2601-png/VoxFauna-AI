import os
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display

from audio_loader import load_audio

def plot_spectrogram(audio, sample_rate, save_path):
   
    mel = librosa.feature.melspectrogram(
    y=audio,
    sr=sample_rate
)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    plt.figure(figsize=(12,5))
    librosa.display.specshow(
    mel_db,
    sr=sample_rate,
    x_axis="time",
    y_axis="mel"
)
    plt.title("Mel Spectrogram of Audio")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Mel Frequency (Hz)")
    plt.colorbar(format="%+2.0f dB")
    plt.savefig(save_path)
    plt.show()
    plt.close()


    
    
if __name__ == "__main__":
    audio_path = "../dataset/raw/dog/dog1.wav"

    output_path = "../outputs"

    os.makedirs(output_path, exist_ok=True)

    audio, sample_rate = load_audio(audio_path)

    if audio is not None:
        save_file = os.path.join(output_path, "dog1_spectrogram.png")
        plot_spectrogram(audio, sample_rate, save_file)
        plt.tight_layout()
        plt.savefig(save_file)

        print(f"\nSpectrogram saved successfully!")
        print(f"Location : {save_file}")
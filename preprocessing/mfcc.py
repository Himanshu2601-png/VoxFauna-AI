import os
import numpy as np
import librosa
from audio_loader import load_audio


def extract_mfcc(audio, sample_rate, save_path):
    
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    print("MFCC shape:", mfcc.shape)
    np.save(save_path, mfcc)
    return mfcc


if __name__ == "__main__":
    audio_path = "../dataset/raw/dog/dog1.wav"
    output_path = "../outputs"
    os.makedirs(output_path, exist_ok=True)
    audio, sample_rate = load_audio(audio_path)
    if audio is not None:
        save_file = os.path.join(output_path, "dog1_mfcc.npy")
        mfcc = extract_mfcc(audio, sample_rate, save_file)

        print(f"\nMFCC saved successfully!")
        print(f"Location : {save_file}")
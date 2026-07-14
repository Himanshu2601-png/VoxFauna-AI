import os
import numpy as np
import soundfile as sf

from python_speech_features import mfcc


def extract_mfcc(audio, sample_rate, save_path):

    mfcc_features = mfcc(
        signal=audio,
        samplerate=sample_rate,
        numcep=40
    )

    print("MFCC shape:", mfcc_features.shape)

    np.save(save_path, mfcc_features)

    return mfcc_features


if __name__ == "__main__":

    audio_path = "../dataset/raw/dog/dog1.wav"

    output_path = "../outputs"

    os.makedirs(output_path, exist_ok=True)

    audio, sample_rate = sf.read(audio_path)

    # Stereo → Mono
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)

    save_file = os.path.join(
        output_path,
        "dog1_mfcc.npy"
    )

    extract_mfcc(audio, sample_rate, save_file)

    print("\nMFCC saved successfully!")
    print(f"Location : {save_file}")
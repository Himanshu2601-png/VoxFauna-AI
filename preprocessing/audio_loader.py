import soundfile as sf
import numpy as np
import os


def load_audio(path):
    try:
        audio, sample_rate = sf.read(path)

        # Convert stereo to mono
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)

        print("=" * 40)
        print("Audio Information")
        print("=" * 40)
        print("File Name      :", os.path.basename(path))
        print("Sample Rate    :", sample_rate)
        print("Duration       :", len(audio) / sample_rate)
        print("Total Samples  :", len(audio))
        print("Audio Shape    :", audio.shape)
        print("=" * 40)

        return audio, sample_rate

    except Exception as e:
        print("Error:", e)
        return None, None
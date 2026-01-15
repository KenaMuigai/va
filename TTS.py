import numpy as np
import sounddevice as sd
from piper.voice import PiperVoice

# Load Piper voice model once
MODEL_PATH = "models/en_US-kristin-medium.onnx"
voice = PiperVoice.load(MODEL_PATH)

def text_to_speech_stream(text: str):
    """
    Convert text to speech using Piper TTS and stream playback
    (plays audio as it’s generated, no truncation).
    """
    # Buffer to accumulate samples if needed
    sample_rate = None

    # Iterate over AudioChunk objects
    for chunk in voice.synthesize(text):
        # First chunk contains sample rate + audio info
        if sample_rate is None:
            sample_rate = chunk.sample_rate

        # raw bytes already int16 PCM
        audio_int16 = np.frombuffer(chunk.audio_int16_bytes, dtype=np.int16)

        # play chunk
        sd.play(audio_int16, samplerate=sample_rate)
        sd.wait()

# Example usage
if __name__ == "__main__":
    text_to_speech_stream(
        "this is a tts demo for you"
    )

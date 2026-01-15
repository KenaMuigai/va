import sounddevice as sd
import numpy as np
import queue
import tempfile
import os
from scipy.io.wavfile import write
from faster_whisper import WhisperModel

SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_DURATION = 5
MODEL_SIZE = "small"

model = WhisperModel(
    MODEL_SIZE,
    device="cpu",
    compute_type="int8"
)

audio_queue = queue.Queue()

def audio_callback(indata, frames, time, status):
    if status:
        print(status)
    audio_queue.put(indata.copy())


def listen_once() -> str:
    """Record audio and return transcribed text"""
    buffer = np.empty((0, CHANNELS), dtype=np.float32)

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        callback=audio_callback
    ):
        while True:
            data = audio_queue.get()
            buffer = np.concatenate((buffer, data))

            if len(buffer) >= SAMPLE_RATE * CHUNK_DURATION:
                audio_chunk = buffer[: SAMPLE_RATE * CHUNK_DURATION]

                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    write(f.name, SAMPLE_RATE, audio_chunk)
                    temp_filename = f.name

                segments, _ = model.transcribe(
                    temp_filename,
                    beam_size=5,
                    vad_filter=True
                )

                os.remove(temp_filename)

                text = ""
                for segment in segments:
                    text += segment.text.strip() + " "

                if text.strip():
                    return text.strip()

if __name__ == "__main__":
    print("Testing ASR. Speak now...")
    text = listen_once()
    print("Transcribed:", text)
import numpy as np
import sounddevice as sd
from queue import Queue
from threading import Thread
from piper.voice import PiperVoice

MODEL_PATH = "models/en_US-kristin-medium.onnx"
voice = PiperVoice.load(MODEL_PATH)


def text_to_speech_stream(text: str, end_silence_seconds: float = 2.0):
    audio_queue = Queue()
    sample_rate = None

    # Producer: generates audio and pushes into queue
    def synthesize():
        nonlocal sample_rate
        for chunk in voice.synthesize(text):
            if sample_rate is None:
                sample_rate = chunk.sample_rate

            audio_int16 = np.frombuffer(chunk.audio_int16_bytes, dtype=np.int16)
            audio_float32 = audio_int16.astype(np.float32) / 32768.0
            audio_queue.put(audio_float32)

        # Add silence at the end
        silence = np.zeros(int(end_silence_seconds * sample_rate), dtype=np.float32)
        audio_queue.put(silence)

        # End signal
        audio_queue.put(None)

    # Consumer callback: fills output buffer
    buffer = np.empty((0,), dtype=np.float32)

    def callback(outdata, frames, time, status):
        nonlocal buffer

        if status:
            print("Stream status:", status)

        # fill buffer if not enough
        while len(buffer) < frames:
            data = audio_queue.get()
            if data is None:
                raise sd.CallbackStop()
            buffer = np.concatenate((buffer, data))

        # write exactly 'frames' samples
        outdata[:] = buffer[:frames].reshape(-1, 1)
        buffer = buffer[frames:]


    # Start synth thread
    synth_thread = Thread(target=synthesize, daemon=True)
    synth_thread.start()

    # wait for sample rate
    while sample_rate is None:
        pass

    # Start stream
    with sd.OutputStream(
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
        callback=callback
    ):
        sd.sleep(int((len(text) / 10 + end_silence_seconds) * 1000))


if __name__ == "__main__":
    text_to_speech_stream("This is a TTS demo for you.")

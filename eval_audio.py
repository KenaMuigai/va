import time
import os
import warnings
import numpy as np
import librosa
from TTS import voice               # Import from your tts.py
from ASR import model, SAMPLE_RATE  # Import from your asr.py

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

try:
    import Levenshtein
except ImportError:
    print("Please run: pip install python-Levenshtein")
    exit(1)

# ---------------------------------------------------------
# A. TTS LATENCY TEST
# ---------------------------------------------------------
def evaluate_tts_latency():
    print("\n" + "="*40)
    print("       A. TTS LATENCY EVALUATION       ")
    print("="*40)
    
    test_sentences = [
        "The weather in Marburg is sunny.",
        "Appointment added successfully.",
        "I could not find that event."
    ]
    
    total_latency = 0
    count = 0
    
    print(f"{'Text':<30} | {'Latency':<10} | {'RTF':<8}")
    print("-" * 55)

    for text in test_sentences:
        start_time = time.time()
        
        # Access generator to measure time-to-first-byte
        stream = voice.synthesize(text)
        
        try:
            # Measure time to get FIRST chunk (Latency)
            first_chunk = next(stream)
            first_chunk_time = time.time()
            latency_ms = (first_chunk_time - start_time) * 1000
            
            # Consume rest to measure total processing speed (RTF)
            audio_len = len(first_chunk.audio_int16_bytes) // 2
            for chunk in stream:
                audio_len += len(chunk.audio_int16_bytes) // 2
                
            total_duration = audio_len / first_chunk.sample_rate
            proc_time = time.time() - start_time
            rtf = proc_time / total_duration if total_duration > 0 else 0
            
            print(f"{text[:27]+'...':<30} | {latency_ms:6.2f} ms | {rtf:.4f}")
            total_latency += latency_ms
            count += 1
        except Exception as e:
            print(f"Error on '{text}': {e}")

    if count > 0:
        print("-" * 55)
        print(f"AVERAGE LATENCY: {total_latency / count:.2f} ms")

# ---------------------------------------------------------
# B. ASR ACCURACY TEST
# ---------------------------------------------------------
def normalize(text):
    import re
    return re.sub(r'[^\w\s]', '', text.lower())

def evaluate_asr_wer():
    print("\n" + "="*40)
    print("       B. ASR ACCURACY (WER)       ")
    print("="*40)
    
    # Ground Truths (Based on your files)
    dataset = [
        {"file": "1_marburg_weather.mp3",      "text": "what will the weather be like today in marburg"},
        {"file": "2_frankfurt_weather.mp3",    "text": "what will the weather be on friday in frankfurt"},
        {"file": "3_there_saturday.mp3",       "text": "will it rain there on saturday"},
        {"file": "4_get_next appointment.mp3", "text": "where is my next appointment"},
        {"file": "5_add_appointment.mp3",      "text": "add an appointment titled final exam for the 30th of january"},
        {"file": "6_change_appointment.mp3",   "text": "change the place for my appointment tomorrow"},
        {"file": "7_delete_appointment.mp3",   "text": "delete the previously created appointment"},
        {"file": "exit.mp3",                   "text": "exit"}
    ]
    
    total_wer = 0
    valid_samples = 0
    base_dir = "audio_inputs"
    
    print(f"{'File':<25} | {'WER':<8} | {'Hypothesis'}")
    print("-" * 70)

    for data in dataset:
        path = os.path.join(base_dir, data["file"])
        if not os.path.exists(path):
            print(f"{data['file']:<25} | MISSING  |")
            continue
            
        # Transcribe
        audio, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
        segments, _ = model.transcribe(audio, beam_size=1, vad_filter=False)
        hypothesis = " ".join(s.text.strip() for s in segments).strip()
        
        # Calculate WER
        ref = normalize(data["text"]).split()
        hyp = normalize(hypothesis).split()
        dist = Levenshtein.distance(ref, hyp)
        wer = dist / len(ref) if ref else 0
        
        print(f"{data['file'][:25]:<25} | {wer:.2%}   | {hypothesis}")
        total_wer += wer
        valid_samples += 1

    if valid_samples > 0:
        print("-" * 70)
        print(f"AVERAGE WER: {total_wer / valid_samples:.2%}")

if __name__ == "__main__":
    evaluate_tts_latency()
    evaluate_asr_wer()